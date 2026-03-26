from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from .bias_absorption import BiasProjection, load_multibert_model


def load_serialized_models(
    seeds: list[int],
    cache_dir: str | Path,
    config: BertConfig | None = None,
) -> tuple[dict[int, Tensor], BertConfig]:
    """Load cached serialized checkpoints for the requested seeds.

    Args:
        seeds: MultiBERT seed ids whose cached `.pt` tensors should be loaded.
        cache_dir: Directory containing `seed_{seed}.pt` files.

    Returns:
        A mapping from seed id to serialized tensor plus a BertConfig. When
        `config` is omitted, the config is inferred from the first loaded model.
    """
    cache_path = Path(cache_dir)
    serialized: dict[int, Tensor] = {}
    for seed in seeds:
        path = cache_path / f"seed_{seed}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing serialized cache: {path}")
        serialized[seed] = torch.load(path, map_location="cpu", weights_only=True)

    if config is None:
        config = load_multibert_model(seeds[0]).bert.config
    return serialized, config


def load_frozen_bias_projection(
    checkpoint_path: str | Path,
    d_model: int,
) -> BiasProjection:
    """Load a BiasProjection checkpoint and freeze its parameters."""
    projection = BiasProjection(d_model)
    projection.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    projection.eval()
    for parameter in projection.parameters():
        parameter.requires_grad_(False)
    return projection


def sample_masked_mlm_batch_from_token_ids(
    token_ids: list[int],
    tokenizer: BertTokenizer,
    num_samples: int,
    max_length: int,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor]:
    """Sample a masked-LM batch from a flat token stream.

    Each batch element draws its own contiguous span uniformly from the full
    token stream, wraps it with `[CLS]` and `[SEP]`, and masks 15% of
    non-special-token positions.
    """
    if seed is not None and generator is not None:
        raise ValueError("Provide either `seed` or `generator`, not both")

    stride = max_length - 2
    if stride <= 0:
        raise ValueError(f"max_length must be at least 3, got {max_length}")

    max_start = len(token_ids) - stride
    if max_start < 0:
        raise ValueError(
            f"Not enough token ids to build {num_samples} sequences of length {max_length}"
        )

    if generator is None and seed is not None:
        generator = torch.Generator().manual_seed(seed)

    starts = torch.randint(0, max_start + 1, (num_samples,), generator=generator)
    sequences = [
        [
            tokenizer.cls_token_id,
            *token_ids[start : start + stride],
            tokenizer.sep_token_id,
        ]
        for start in starts.tolist()
    ]

    input_ids = torch.tensor(sequences, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    mask_draw = torch.rand(input_ids.shape, generator=generator)
    mask_draw[:, 0] = 1.0
    mask_draw[:, -1] = 1.0
    masked_positions = mask_draw < 0.15

    masked_input = input_ids.clone()
    masked_input[masked_positions] = tokenizer.mask_token_id
    labels[~masked_positions] = -100

    return {
        "input_ids": masked_input,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def get_extended_attention_mask(
    shell_model: BertForMaskedLM,
    attention_mask: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """Expand a standard attention mask to the shape expected by BertLayer."""
    return shell_model.bert.get_extended_attention_mask(
        attention_mask,
        input_shape=attention_mask.shape,
        dtype=dtype,
    )


def build_functional_mlm_params(
    shell_model: BertForMaskedLM,
    bert_params: dict[str, Tensor],
    extra_params: dict[str, Tensor],
    device: torch.device,
) -> dict[str, Tensor]:
    """Assemble a complete parameter mapping for BertForMaskedLM functional calls."""
    full_params = {f"bert.{name}": value.to(device) for name, value in bert_params.items()}
    for name, value in extra_params.items():
        full_params[name] = value.to(device)
    for name, value in shell_model.state_dict().items():
        if name not in full_params:
            full_params[name] = value.to(device)
    return full_params


def run_functional_mlm_logits(
    shell_model: BertForMaskedLM,
    bert_params: dict[str, Tensor],
    extra_params: dict[str, Tensor],
    batch: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    """Run a functional MLM forward pass and return logits."""
    output = torch.func.functional_call(
        shell_model,
        build_functional_mlm_params(shell_model, bert_params, extra_params, device),
        tie_weights=False,
        kwargs={
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        },
    )
    return output.logits


def run_functional_mlm_loss(
    shell_model: BertForMaskedLM,
    bert_params: dict[str, Tensor],
    extra_params: dict[str, Tensor],
    batch: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    """Run a differentiable MLM loss with functional_call.

    `bert_params` should contain BertModel keys without the leading `bert.`.
    `extra_params` should contain the pretrained non-BERT parameters such as the
    untied MLM head.
    """
    output = torch.func.functional_call(
        shell_model,
        build_functional_mlm_params(shell_model, bert_params, extra_params, device),
        tie_weights=False,
        kwargs={
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        },
    )
    return output.loss