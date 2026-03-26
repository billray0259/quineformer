from pathlib import Path

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM

from quineformer.bias_absorption import extract_non_bert_params
from quineformer.experiment_utils import (
    build_functional_mlm_params,
    sample_masked_mlm_batch_from_token_ids,
    get_extended_attention_mask,
    load_frozen_bias_projection,
    load_serialized_models,
    run_functional_mlm_logits,
    run_functional_mlm_loss,
)
from quineformer.serialization import serialize


BERT_BASE_CONFIG = dict(
    vocab_size=100,
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=32,
    max_position_embeddings=16,
    type_vocab_size=2,
)


class DummyTokenizer:
    cls_token_id = 101
    sep_token_id = 102
    mask_token_id = 103


@pytest.fixture(scope="session")
def tiny_bert():
    config = BertConfig(**BERT_BASE_CONFIG)
    return BertForMaskedLM(config).eval()


class TestLoadSerializedModels:
    def test_loads_requested_seed_files(self, tmp_path):
        config = BertConfig(**BERT_BASE_CONFIG)
        for seed in (0, 1):
            torch.save(torch.full((2, 3), float(seed)), tmp_path / f"seed_{seed}.pt")

        serialized, loaded_config = load_serialized_models([0, 1], tmp_path, config=config)

        assert set(serialized) == {0, 1}
        assert torch.equal(serialized[0], torch.zeros(2, 3))
        assert loaded_config is config

    def test_raises_for_missing_seed_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_serialized_models([0], tmp_path)


class TestLoadFrozenBiasProjection:
    def test_loads_and_freezes_checkpoint(self, tmp_path):
        config = BertConfig(**BERT_BASE_CONFIG)
        model = BertForMaskedLM(config).eval()
        projection = load_frozen_bias_projection(
            _write_projection_checkpoint(tmp_path, config.hidden_size),
            config.hidden_size,
        )

        assert projection.training is False
        assert all(not parameter.requires_grad for parameter in projection.parameters())
        assert next(projection.parameters()).shape[0] == model.config.hidden_size


class TestSampleMaskedMlmBatchFromTokenIds:
    def test_builds_expected_shapes(self):
        batch = sample_masked_mlm_batch_from_token_ids(
            token_ids=list(range(500)),
            tokenizer=DummyTokenizer(),
            num_samples=3,
            max_length=8,
            seed=0,
        )

        assert batch["input_ids"].shape == (3, 8)
        assert batch["attention_mask"].shape == (3, 8)
        assert batch["labels"].shape == (3, 8)

    def test_keeps_special_tokens_unmasked(self):
        batch = sample_masked_mlm_batch_from_token_ids(
            token_ids=list(range(500)),
            tokenizer=DummyTokenizer(),
            num_samples=2,
            max_length=8,
            seed=0,
        )

        assert torch.equal(batch["input_ids"][:, 0], torch.full((2,), DummyTokenizer.cls_token_id))
        assert torch.equal(batch["input_ids"][:, -1], torch.full((2,), DummyTokenizer.sep_token_id))
        assert torch.equal(batch["labels"][:, 0], torch.full((2,), -100))
        assert torch.equal(batch["labels"][:, -1], torch.full((2,), -100))

    def test_samples_from_full_token_stream(self):
        batch = sample_masked_mlm_batch_from_token_ids(
            token_ids=list(range(500)),
            tokenizer=DummyTokenizer(),
            num_samples=3,
            max_length=8,
            seed=0,
        )

        reconstructed = torch.where(batch["labels"] == -100, batch["input_ids"], batch["labels"])
        sequential = torch.tensor(
            [
                [DummyTokenizer.cls_token_id, 0, 1, 2, 3, 4, 5, DummyTokenizer.sep_token_id],
                [DummyTokenizer.cls_token_id, 6, 7, 8, 9, 10, 11, DummyTokenizer.sep_token_id],
                [DummyTokenizer.cls_token_id, 12, 13, 14, 15, 16, 17, DummyTokenizer.sep_token_id],
            ],
            dtype=torch.long,
        )

        assert not torch.equal(reconstructed, sequential)

    def test_rejects_insufficient_token_stream(self):
        with pytest.raises(ValueError):
            sample_masked_mlm_batch_from_token_ids(
                token_ids=list(range(4)),
                tokenizer=DummyTokenizer(),
                num_samples=2,
                max_length=8,
                seed=0,
            )


class TestAttentionMaskAndFunctionalLoss:
    def test_get_extended_attention_mask_preserves_batch(self, tiny_bert):
        attention_mask = torch.ones(2, 8, dtype=torch.long)
        extended = get_extended_attention_mask(tiny_bert, attention_mask, torch.float32)
        assert extended.shape[0] == 2

    def test_run_functional_mlm_loss_matches_direct_forward(self, tiny_bert):
        batch = {
            "input_ids": torch.randint(0, BERT_BASE_CONFIG["vocab_size"], (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
            "labels": torch.randint(0, BERT_BASE_CONFIG["vocab_size"], (2, 8)),
        }
        bert_params = tiny_bert.bert.state_dict()
        extra_params = extract_non_bert_params(tiny_bert)

        direct = tiny_bert(**batch).loss
        functional = run_functional_mlm_loss(
            tiny_bert,
            bert_params,
            extra_params,
            batch,
            torch.device("cpu"),
        )

        assert torch.allclose(direct, functional)

    def test_build_functional_mlm_params_includes_pretrained_fallbacks(self, tiny_bert):
        bert_params = tiny_bert.bert.state_dict()
        extra_params = extract_non_bert_params(tiny_bert)

        full_params = build_functional_mlm_params(
            tiny_bert,
            bert_params,
            extra_params,
            torch.device("cpu"),
        )

        assert "bert.embeddings.word_embeddings.weight" in full_params
        assert "cls.predictions.bias" in full_params

    def test_run_functional_mlm_logits_matches_direct_forward(self, tiny_bert):
        batch = {
            "input_ids": torch.randint(0, BERT_BASE_CONFIG["vocab_size"], (2, 8)),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        bert_params = tiny_bert.bert.state_dict()
        extra_params = extract_non_bert_params(tiny_bert)

        direct = tiny_bert(**batch).logits
        functional = run_functional_mlm_logits(
            tiny_bert,
            bert_params,
            extra_params,
            batch,
            torch.device("cpu"),
        )

        assert torch.allclose(direct, functional)


def _write_projection_checkpoint(tmp_path: Path, d_model: int) -> Path:
    from quineformer.bias_absorption import BiasProjection

    path = tmp_path / "projection.pt"
    torch.save(BiasProjection(d_model).state_dict(), path)
    return path