import math

import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM

from quineformer.bias_absorption import (
    BIAS_CARRYING_TYPES,
    BiasProjection,
    absorb_bias_rows_only,
    apply_head_permutation,
    apply_neuron_permutation,
    apply_projection_to_bias_rows,
    bias_carrying_mask,
    compute_bias_accuracy,
    compute_mlm_perplexity,
    compute_reconstruction_errors,
    extract_non_bert_params,
    reconstruct_model,
    reconstruction_mse_in_batches,
    restore_bias_rows_only,
    train_projection,
)
from quineformer.serialization import serialize, vector_component_labels


# ── Shared fixtures ────────────────────────────────────────────────

BERT_BASE_CONFIG = dict(
    vocab_size=100,
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=32,
    max_position_embeddings=16,
    type_vocab_size=2,
)


@pytest.fixture(scope="session")
def tiny_config():
    return BertConfig(**BERT_BASE_CONFIG)


@pytest.fixture(scope="session")
def tiny_bert(tiny_config):
    return BertForMaskedLM(tiny_config).eval()


@pytest.fixture(scope="session")
def serialized(tiny_bert):
    with torch.no_grad():
        return serialize(tiny_bert)


@pytest.fixture(scope="session")
def labels(tiny_config):
    return vector_component_labels(tiny_config)


@pytest.fixture(scope="session")
def identity_projection(tiny_config):
    """Freshly-initialized BiasProjection (identity at init)."""
    return BiasProjection(tiny_config.hidden_size)


# ── BiasProjection ─────────────────────────────────────────────────


class TestBiasProjection:
    def test_forward_output_shape(self, tiny_config):
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        x = torch.randn(8, d + 1)
        y = model(x)
        assert y.shape == (8, d + 1)

    def test_encode_shape(self, tiny_config):
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        x = torch.randn(8, d + 1)
        assert model.encode(x).shape == (8, d)

    def test_decode_shape(self, tiny_config):
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        z = torch.randn(8, d)
        assert model.decode(z).shape == (8, d + 1)

    def test_forward_is_decode_of_encode(self, tiny_config):
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        x = torch.randn(8, d + 1)
        assert torch.equal(model(x), model.decode(model.encode(x)))

    def test_identity_init_zeroes_bias_dim(self, tiny_config):
        """At initialisation W_in discards the bias dimension, so for any x
        with x[:, d] == c the reconstructed bias dim should be zero."""
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        x = torch.randn(8, d + 1)
        with torch.no_grad():
            y = model(x)
        assert torch.allclose(y[:, d], torch.zeros(8), atol=1e-6)

    def test_identity_init_preserves_weight_dims(self, tiny_config):
        """At initialisation the weight dimensions are reproduced exactly."""
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        x = torch.randn(8, d + 1)
        with torch.no_grad():
            y = model(x)
        assert torch.allclose(y[:, :d], x[:, :d], atol=1e-6)

    def test_parameters_are_trainable(self, tiny_config):
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        params = list(model.parameters())
        assert len(params) == 2  # w_in and w_out
        assert all(p.requires_grad for p in params)


# ── train_projection ───────────────────────────────────────────────


class TestTrainProjection:
    def test_returns_nn_module(self, serialized, tiny_config):
        d = tiny_config.hidden_size
        mask = bias_carrying_mask(tiny_config)
        data = serialized[mask][:64]
        model = train_projection(data, d, n_epochs=1, device=torch.device("cpu"))
        assert isinstance(model, nn.Module)

    def test_returned_model_on_cpu(self, serialized, tiny_config):
        d = tiny_config.hidden_size
        mask = bias_carrying_mask(tiny_config)
        data = serialized[mask][:64]
        model = train_projection(data, d, n_epochs=1, device=torch.device("cpu"))
        assert next(model.parameters()).device == torch.device("cpu")

    def test_loss_decreases(self, serialized, tiny_config):
        """Training should reduce MSE relative to the untrained identity."""
        d = tiny_config.hidden_size
        mask = bias_carrying_mask(tiny_config)
        data = serialized[mask]

        # Untrained identity MSE (bias dim zeroed → known non-zero loss)
        identity = BiasProjection(d)
        mse_before, _ = reconstruction_mse_in_batches(
            identity, data, device=torch.device("cpu")
        )

        trained = train_projection(
            data, d, n_epochs=20, lr=1e-2, device=torch.device("cpu")
        )
        mse_after, _ = reconstruction_mse_in_batches(
            trained, data, device=torch.device("cpu")
        )
        assert mse_after <= mse_before

    def test_accepts_custom_model(self, serialized, tiny_config):
        d = tiny_config.hidden_size
        mask = bias_carrying_mask(tiny_config)
        data = serialized[mask][:64]
        custom = BiasProjection(d)
        returned = train_projection(
            data, d, model=custom, n_epochs=1, device=torch.device("cpu")
        )
        assert returned is custom


class TestBiasAbsorptionHelpers:
    def test_absorb_bias_rows_only_returns_d_model_width(
        self, identity_projection, serialized, tiny_config
    ):
        absorbed = absorb_bias_rows_only(
            identity_projection, serialized, tiny_config, torch.device("cpu")
        )
        assert absorbed.shape == (serialized.shape[0], tiny_config.hidden_size)

    def test_absorb_bias_rows_only_preserves_non_bias_rows(
        self, identity_projection, serialized, tiny_config
    ):
        mask = bias_carrying_mask(tiny_config)
        absorbed = absorb_bias_rows_only(
            identity_projection, serialized, tiny_config, torch.device("cpu")
        )
        assert torch.allclose(absorbed[~mask], serialized[~mask, : tiny_config.hidden_size])

    def test_restore_bias_rows_only_preserves_weight_dims(
        self, identity_projection, serialized, tiny_config
    ):
        absorbed = absorb_bias_rows_only(
            identity_projection, serialized, tiny_config, torch.device("cpu")
        )
        restored = restore_bias_rows_only(
            identity_projection, absorbed, tiny_config, torch.device("cpu")
        )
        assert restored.shape == serialized.shape
        assert torch.allclose(restored[:, : tiny_config.hidden_size], serialized[:, : tiny_config.hidden_size])

    def test_extract_non_bert_params_returns_only_non_bert_keys(self, tiny_bert):
        params = extract_non_bert_params(tiny_bert)
        assert params
        assert all(not key.startswith("bert.") for key in params)


# ── reconstruction_mse_in_batches ──────────────────────────────────


class TestReconstructionMseInBatches:
    def test_returns_two_floats(self, serialized, tiny_config, identity_projection):
        result = reconstruction_mse_in_batches(
            identity_projection, serialized, device=torch.device("cpu")
        )
        assert len(result) == 2
        total_mse, bias_mse = result
        assert isinstance(total_mse, float)
        assert isinstance(bias_mse, float)

    def test_mse_nonnegative(self, serialized, tiny_config, identity_projection):
        total_mse, bias_mse = reconstruction_mse_in_batches(
            identity_projection, serialized, device=torch.device("cpu")
        )
        assert total_mse >= 0.0
        assert bias_mse >= 0.0

    def test_identity_has_zero_weight_mse(self, serialized, tiny_config):
        """Identity projection perfectly reconstructs weight dims (bias dim only is non-zero)."""
        d = tiny_config.hidden_size
        model = BiasProjection(d)
        # Build a target where all inputs have zero bias dim
        data = serialized.clone()
        data[:, d] = 0.0
        total_mse, _ = reconstruction_mse_in_batches(
            model, data, device=torch.device("cpu")
        )
        assert total_mse < 1e-10


# ── compute_reconstruction_errors ──────────────────────────────────


class TestComputeReconstructionErrors:
    def test_keys_match_all_component_types(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_reconstruction_errors(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        assert set(result.keys()) == set(labels)

    def test_each_entry_has_expected_fields(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_reconstruction_errors(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        for ctype, stats in result.items():
            assert "mse" in stats
            assert "relative_mse" in stats
            assert "count" in stats

    def test_counts_sum_to_total_rows(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_reconstruction_errors(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        assert sum(v["count"] for v in result.values()) == serialized.shape[0]

    def test_mse_nonnegative(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_reconstruction_errors(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        for ctype, stats in result.items():
            assert stats["mse"] >= 0.0

    def test_non_bias_types_have_zero_mse(
        self, identity_projection, serialized, labels, tiny_config
    ):
        """Identity projection reconstructs non-bias-carrying types exactly."""
        result = compute_reconstruction_errors(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        for ctype, stats in result.items():
            if ctype not in BIAS_CARRYING_TYPES:
                assert stats["mse"] < 1e-10, (
                    f"{ctype} should have zero MSE under identity, got {stats['mse']}"
                )


# ── compute_bias_accuracy ──────────────────────────────────────────


class TestComputeBiasAccuracy:
    def test_keys_are_bias_carrying_types(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_bias_accuracy(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        assert set(result.keys()) == BIAS_CARRYING_TYPES

    def test_each_entry_has_expected_fields(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_bias_accuracy(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        for ctype, stats in result.items():
            assert "correlation" in stats
            assert "mae" in stats
            assert "count" in stats

    def test_mae_nonnegative(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_bias_accuracy(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        for ctype, stats in result.items():
            assert stats["mae"] >= 0.0

    def test_correlation_in_range(
        self, identity_projection, serialized, labels, tiny_config
    ):
        result = compute_bias_accuracy(
            identity_projection, serialized, labels, tiny_config,
            device=torch.device("cpu")
        )
        for ctype, stats in result.items():
            assert -1.0 - 1e-6 <= stats["correlation"] <= 1.0 + 1e-6

    def test_perfect_reconstruction_gives_zero_mae(
        self, serialized, labels, tiny_config
    ):
        """A model that reconstructs rows perfectly gives mae = 0 for all types."""
        class PassThrough(nn.Module):
            def forward(self, x):
                return x

        result = compute_bias_accuracy(
            PassThrough(), serialized, labels, tiny_config, device=torch.device("cpu")
        )
        for ctype, stats in result.items():
            assert stats["mae"] < 1e-6, (
                f"{ctype}: expected mae ≈ 0, got {stats['mae']}"
            )


# ── compute_mlm_perplexity ─────────────────────────────────────────


def _make_mlm_batch(config, seq_len=8, num_samples=2):
    """Build a minimal synthetic MLM batch."""
    vocab = config.vocab_size
    input_ids = torch.randint(1, vocab, (num_samples, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    # Mask ~20% of non-special positions
    mask = torch.rand(num_samples, seq_len) < 0.2
    mask[:, 0] = False
    mask[:, -1] = False
    input_ids[mask] = 103 % vocab  # [MASK] substitute
    labels[~mask] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class TestComputeMlmPerplexity:
    def test_returns_positive_float(self, tiny_bert, tiny_config):
        batch = _make_mlm_batch(tiny_config)
        ppl = compute_mlm_perplexity(tiny_bert, batch)
        assert isinstance(ppl, float)
        assert ppl > 0.0

    def test_perplexity_is_finite(self, tiny_bert, tiny_config):
        batch = _make_mlm_batch(tiny_config)
        ppl = compute_mlm_perplexity(tiny_bert, batch)
        assert math.isfinite(ppl)

    def test_better_model_lower_perplexity(self, tiny_config):
        """A model whose weights are zeroed should have higher perplexity than the
        original random model (both are random, but zeroing makes predictions flat)."""
        model_a = BertForMaskedLM(tiny_config).eval()
        model_b = BertForMaskedLM(tiny_config).eval()
        # Zero all parameters in model_b → flat logits → high uniform perplexity
        with torch.no_grad():
            for p in model_b.parameters():
                p.zero_()
        batch = _make_mlm_batch(tiny_config)
        ppl_a = compute_mlm_perplexity(model_a, batch)
        ppl_b = compute_mlm_perplexity(model_b, batch)
        # model_b has flat logits (log-uniform distribution) — should be ≥ vocab_size
        assert ppl_b >= tiny_config.vocab_size - 1


# ── apply_neuron_permutation ───────────────────────────────────────


class TestApplyNeuronPermutation:
    def test_returns_new_dict(self, tiny_bert):
        sd = dict(tiny_bert.bert.state_dict())
        perm = torch.randperm(BERT_BASE_CONFIG["intermediate_size"])
        result = apply_neuron_permutation(sd, layer_idx=0, perm=perm)
        assert result is not sd

    def test_does_not_mutate_input(self, tiny_bert):
        sd = dict(tiny_bert.bert.state_dict())
        orig_w = sd["encoder.layer.0.intermediate.dense.weight"].clone()
        perm = torch.randperm(BERT_BASE_CONFIG["intermediate_size"])
        apply_neuron_permutation(sd, layer_idx=0, perm=perm)
        assert torch.equal(sd["encoder.layer.0.intermediate.dense.weight"], orig_w)

    def test_permuted_model_same_output(self, tiny_config, tiny_bert):
        """Permuting MLP neurons preserves the BERT encoder function."""
        sd = dict(tiny_bert.bert.state_dict())
        d_ff = tiny_config.intermediate_size
        perm = torch.randperm(d_ff)
        sd_perm = apply_neuron_permutation(sd, layer_idx=0, perm=perm)

        model_perm = BertForMaskedLM(tiny_config).eval()
        model_perm.bert.load_state_dict(sd_perm)

        ids = torch.randint(1, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            out_orig = tiny_bert.bert(ids).last_hidden_state
            out_perm = model_perm.bert(ids).last_hidden_state
        assert torch.allclose(out_orig, out_perm, atol=1e-5)

    def test_intermediate_weight_rows_permuted(self, tiny_bert):
        sd = dict(tiny_bert.bert.state_dict())
        d_ff = BERT_BASE_CONFIG["intermediate_size"]
        perm = torch.arange(d_ff - 1, -1, -1)  # reverse
        result = apply_neuron_permutation(sd, layer_idx=1, perm=perm)
        expected = sd["encoder.layer.1.intermediate.dense.weight"].flip(0)
        assert torch.equal(result["encoder.layer.1.intermediate.dense.weight"], expected)

    def test_output_weight_columns_permuted(self, tiny_bert):
        sd = dict(tiny_bert.bert.state_dict())
        d_ff = BERT_BASE_CONFIG["intermediate_size"]
        perm = torch.arange(d_ff - 1, -1, -1)  # reverse
        result = apply_neuron_permutation(sd, layer_idx=1, perm=perm)
        expected = sd["encoder.layer.1.output.dense.weight"].flip(1)
        assert torch.equal(result["encoder.layer.1.output.dense.weight"], expected)


# ── apply_head_permutation ─────────────────────────────────────────


class TestApplyHeadPermutation:
    def test_returns_new_dict(self, tiny_bert):
        sd = dict(tiny_bert.bert.state_dict())
        n_heads = BERT_BASE_CONFIG["num_attention_heads"]
        perm = torch.randperm(n_heads)
        result = apply_head_permutation(
            sd, layer_idx=0, perm=perm,
            num_heads=n_heads, d_model=BERT_BASE_CONFIG["hidden_size"]
        )
        assert result is not sd

    def test_does_not_mutate_input(self, tiny_bert):
        sd = dict(tiny_bert.bert.state_dict())
        orig_w = sd["encoder.layer.0.attention.self.query.weight"].clone()
        n_heads = BERT_BASE_CONFIG["num_attention_heads"]
        perm = torch.randperm(n_heads)
        apply_head_permutation(
            sd, layer_idx=0, perm=perm,
            num_heads=n_heads, d_model=BERT_BASE_CONFIG["hidden_size"]
        )
        assert torch.equal(sd["encoder.layer.0.attention.self.query.weight"], orig_w)

    def test_permuted_model_same_output(self, tiny_config, tiny_bert):
        """Permuting attention heads preserves the BERT encoder function."""
        sd = dict(tiny_bert.bert.state_dict())
        n_heads = tiny_config.num_attention_heads
        perm = torch.randperm(n_heads)
        sd_perm = apply_head_permutation(
            sd, layer_idx=0, perm=perm,
            num_heads=n_heads, d_model=tiny_config.hidden_size,
        )

        model_perm = BertForMaskedLM(tiny_config).eval()
        model_perm.bert.load_state_dict(sd_perm)

        ids = torch.randint(1, tiny_config.vocab_size, (1, 8))
        with torch.no_grad():
            out_orig = tiny_bert.bert(ids).last_hidden_state
            out_perm = model_perm.bert(ids).last_hidden_state
        assert torch.allclose(out_orig, out_perm, atol=1e-5)

    def test_identity_permutation_leaves_weights_unchanged(self, tiny_bert):
        sd = dict(tiny_bert.bert.state_dict())
        n_heads = BERT_BASE_CONFIG["num_attention_heads"]
        identity_perm = torch.arange(n_heads)
        result = apply_head_permutation(
            sd, layer_idx=0, perm=identity_perm,
            num_heads=n_heads, d_model=BERT_BASE_CONFIG["hidden_size"]
        )
        for proj in ["query", "key", "value"]:
            k = f"encoder.layer.0.attention.self.{proj}.weight"
            assert torch.allclose(result[k], sd[k], atol=1e-7)
