"""Canonicalization Experiment V2: minimal layer-local surrogate training.

This experiment trains the simplified Q/K canonicalization module with a
layer-local objective instead of running a full interpolated model forward on
every optimization step.

High-level training step:
  1. Compute model-level permutations P_i and P_j from endpoint embeddings.
  2. Pick one encoder layer l.
  3. Interpolate that layer's absorbed parameters in canonical space.
  4. Map the interpolated layer back into model i's basis and execute only that
     BertLayer on an interpolated canonical residual stream.
  5. Compare the predicted output residual stream to the interpolated canonical
     endpoint output residual stream.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from itertools import combinations
from pathlib import Path
from dotenv import load_dotenv
load_dotenv("../../.env")

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

try:
    import wandb
except ImportError:
    wandb = None

from quineformer.bias_absorption import (
    BiasProjection,
    absorb_bias_rows_only,
    bias_carrying_mask,
    extract_non_bert_params,
    load_multibert_model,
    restore_bias_rows_only,
)
from quineformer.canonicalization import CanonicalizationModule
from quineformer.canonicalization import sinkhorn
from quineformer.experiment_utils import (
    get_extended_attention_mask,
    load_frozen_bias_projection,
    load_serialized_models,
    run_functional_mlm_logits,
    run_functional_mlm_loss,
    sample_masked_mlm_batch_from_token_ids,
)
from quineformer.serialization import (
    deserialize,
    deserialize_encoder_layer,
    encoder_layer_row_bounds,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SERIALIZED_CACHE = REPO_ROOT / "data" / "multiberts" / "serialized"
BIAS_CKPT = REPO_ROOT / "experiments" / "bias_absorption" / "results_v1_min" / "projection_shared.pt"
RESULTS_DIR = Path(__file__).resolve().parent / "results_v2"
DEFAULT_ACTIVATION_DATASET_DIR = Path("/data/bill/datasets/quineformer/canonicalization_v2_activations")
WANDB_PROJECT = "quineformer-canonicalization"

TRAIN_SEEDS = list(range(20))
TEST_SEEDS = list(range(20, 25))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the v2 canonicalization experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lambda-sharp", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--tau-init", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validation-layers", type=int, nargs="*", default=[0, 5, 11])
    parser.add_argument("--eval-pairs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--activation-dataset-dir", type=Path, default=DEFAULT_ACTIVATION_DATASET_DIR)
    parser.add_argument("--steps-per-shard", type=int, default=32)
    parser.add_argument("--train-live-shards", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=1)
    parser.add_argument("--final-eval-batches", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--skip-perplexity-eval", action="store_true")
    return parser.parse_args()


def load_wikitext_token_ids(tokenizer: BertTokenizer, split: str) -> list[int]:
    """Load a WikiText split once and flatten it into a token stream."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("run_v2.py requires the `datasets` package to sample MLM batches.") from error

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(chunk for chunk in dataset["text"] if chunk.strip())
    return tokenizer.encode(text, add_special_tokens=False)


def sample_batch(
    token_ids: list[int],
    tokenizer: BertTokenizer,
    num_samples: int,
    max_length: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Sample a fresh masked-LM batch from the full token stream."""
    return sample_masked_mlm_batch_from_token_ids(
        token_ids=token_ids,
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
    )


def build_reference_batch(
    token_ids: list[int],
    tokenizer: BertTokenizer,
    num_samples: int,
    max_length: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build one fixed masked-LM batch reused across training and evaluation."""
    return sample_batch(token_ids, tokenizer, num_samples, max_length, seed)


def init_wandb(args: argparse.Namespace, config: BertConfig):
    """Initialize a W&B run when wandb is installed and credentials are available."""
    if wandb is None:
        print("W&B disabled: package not installed")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("W&B disabled: WANDB_API_KEY not set")
        return None

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME"),
        config={
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lambda_sharp": args.lambda_sharp,
            "sinkhorn_iters": args.sinkhorn_iters,
            "tau_init": args.tau_init,
            "alpha": args.alpha,
            "num_samples": args.num_samples,
            "max_length": args.max_length,
            "validation_layers": args.validation_layers,
            "eval_pairs": args.eval_pairs,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "train_seeds": TRAIN_SEEDS[:4] if args.smoke_test else TRAIN_SEEDS,
            "test_seeds": TEST_SEEDS[:2] if args.smoke_test else TEST_SEEDS,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
        },
    )
    print(f"W&B enabled: {run.project}/{run.name}")
    return run


def wandb_log(run, metrics: dict[str, float | int], step: int | None = None) -> None:
    """Log metrics to W&B when a run is active."""
    if run is not None:
        run.log(metrics, step=step)


def summarize_perplexity_results(results: list[dict[str, float | int]]) -> dict[str, float]:
    """Aggregate pairwise perplexity metrics for compact epoch logging."""
    if not results:
        return {
            "val_interp_ppl": float("nan"),
            "val_ensemble_ppl": float("nan"),
            "val_ppl_ratio": float("nan"),
        }

    denom = float(len(results))
    return {
        "val_interp_ppl": sum(float(result["interp_ppl"]) for result in results) / denom,
        "val_ensemble_ppl": sum(float(result["ensemble_ppl"]) for result in results) / denom,
        "val_ppl_ratio": sum(float(result["ratio"]) for result in results) / denom,
    }


def summarize_roundtrip_results(results: list[dict[str, float | int]]) -> dict[str, float]:
    """Aggregate round-trip perplexity metrics for compact epoch logging."""
    if not results:
        return {
            "val_roundtrip_ppl": float("nan"),
            "val_roundtrip_baseline_ppl": float("nan"),
            "val_roundtrip_ppl_ratio": float("nan"),
        }

    denom = float(len(results))
    return {
        "val_roundtrip_ppl": sum(float(result["roundtrip_ppl"]) for result in results) / denom,
        "val_roundtrip_baseline_ppl": sum(float(result["baseline_ppl"]) for result in results) / denom,
        "val_roundtrip_ppl_ratio": sum(float(result["ratio"]) for result in results) / denom,
    }


def load_activation_dataset_metadata(dataset_dir: Path) -> dict:
    """Load dataset metadata for the precomputed activation shards."""
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing activation dataset metadata: {metadata_path}. "
            "Run scripts/generate_canonicalization_v2_activation_dataset.py first."
        )
    with open(metadata_path, "r", encoding="ascii") as handle:
        return json.load(handle)


def list_activation_shards(dataset_dir: Path, split: str) -> list[Path]:
    """List shard files for one split of the activation dataset."""
    shard_paths = sorted((dataset_dir / split).glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No activation shards found in {dataset_dir / split}")
    return shard_paths


def load_activation_shard(shard_path: Path) -> dict[str, torch.Tensor | list[int]]:
    """Load one activation shard from disk."""
    return torch.load(shard_path, map_location="cpu", weights_only=True)


def shard_example_count(shard: dict[str, torch.Tensor | list[int]]) -> int:
    """Return the number of cached examples stored in one shard."""
    return int(shard["hidden_states"].shape[1])


def extract_activation_example(
    shard: dict[str, torch.Tensor | list[int]],
    example_idx: int,
    source: str,
) -> dict[str, object]:
    """Extract one cached example and its endpoint hidden states from a shard."""
    return {
        "hidden_states": shard["hidden_states"][:, example_idx],
        "source": source,
    }


def assemble_activation_minibatch(examples: list[dict[str, object]]) -> dict[str, object]:
    """Stack cached examples into one training minibatch."""
    return {
        "hidden_states": torch.stack([example["hidden_states"] for example in examples], dim=2),
        "source": ",".join(str(example["source"]) for example in examples),
    }


def sample_examples_from_shard(
    shard: dict[str, torch.Tensor | list[int]],
    batch_size: int,
    source_prefix: str,
) -> dict[str, object]:
    """Sample a minibatch of examples from a loaded shard."""
    examples = [
        extract_activation_example(
            shard,
            random.randrange(shard_example_count(shard)),
            f"{source_prefix}:{sample_idx}",
        )
        for sample_idx in range(batch_size)
    ]
    return assemble_activation_minibatch(examples)


def sample_examples_from_shards(
    shards: list[tuple[str, dict[str, torch.Tensor | list[int]]]],
    batch_size: int,
) -> dict[str, object]:
    """Sample one minibatch by mixing examples across multiple loaded shards."""
    if not shards:
        raise ValueError("Expected at least one loaded shard")

    examples = []
    for sample_idx in range(batch_size):
        shard_name, shard = random.choice(shards)
        examples.append(
            extract_activation_example(
                shard,
                random.randrange(shard_example_count(shard)),
                f"{shard_name}:{sample_idx}",
            )
        )

    return assemble_activation_minibatch(examples)


def load_training_shard_pool(
    shard_paths: list[Path],
    start_idx: int,
    pool_size: int,
) -> tuple[list[tuple[str, dict[str, torch.Tensor | list[int]]]], int]:
    """Load a small pool of training shards for mixed example sampling."""
    loaded_shards = []
    shard_idx = start_idx

    for _ in range(max(pool_size, 1)):
        if shard_idx >= len(shard_paths):
            random.shuffle(shard_paths)
            shard_idx = 0
        shard_path = shard_paths[shard_idx]
        loaded_shards.append((shard_path.stem, load_activation_shard(shard_path)))
        shard_idx += 1

    return loaded_shards, shard_idx


def release_training_shard_pool(
    loaded_shards: list[tuple[str, dict[str, torch.Tensor | list[int]]]],
) -> list[tuple[str, dict[str, torch.Tensor | list[int]]]]:
    """Release references held by a loaded shard pool before reloading."""
    for _, shard in loaded_shards:
        shard.clear()
    loaded_shards.clear()
    gc.collect()
    return []


def sample_activation_minibatches(
    shard_paths: list[Path],
    num_minibatches: int,
    batch_size: int,
) -> list[dict[str, object]]:
    """Randomly sample cached examples and assemble them into minibatches."""
    shard_cache: dict[Path, dict[str, torch.Tensor | list[int]]] = {}
    minibatches = []

    for _ in range(max(num_minibatches, 1)):
        shard_path = random.choice(shard_paths)
        shard = shard_cache.get(shard_path)
        if shard is None:
            shard = load_activation_shard(shard_path)
            shard_cache[shard_path] = shard
        minibatches.append(sample_examples_from_shard(shard, batch_size, shard_path.stem))

    return minibatches


def deterministic_activation_minibatches(
    shard_paths: list[Path],
    num_minibatches: int,
    batch_size: int,
) -> list[dict[str, object]]:
    """Load a deterministic prefix of cached examples and group them into minibatches."""
    gathered_examples = []
    required_examples = max(num_minibatches, 1) * batch_size

    for shard_path in shard_paths:
        shard = load_activation_shard(shard_path)
        for example_idx in range(shard_example_count(shard)):
            gathered_examples.append(
                extract_activation_example(shard, example_idx, f"{shard_path.stem}:{example_idx}")
            )
            if len(gathered_examples) >= required_examples:
                break
        if len(gathered_examples) >= required_examples:
            break

    if not gathered_examples:
        return []

    minibatches = []
    for start in range(0, len(gathered_examples), batch_size):
        chunk = gathered_examples[start : start + batch_size]
        if len(chunk) < batch_size:
            break
        minibatches.append(assemble_activation_minibatch(chunk))
        if len(minibatches) >= max(num_minibatches, 1):
            break

    return minibatches


def restore_layer_rows(
    projection: BiasProjection,
    layer_absorbed: torch.Tensor,
    config: BertConfig,
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    """Restore one absorbed encoder-layer slice back to serialized row form."""
    full_mask = bias_carrying_mask(config).to(device)
    start, end = encoder_layer_row_bounds(config, layer_idx)
    layer_mask = full_mask[start:end]
    restored = torch.cat(
        [
            layer_absorbed,
            torch.zeros(layer_absorbed.shape[0], 1, device=device, dtype=layer_absorbed.dtype),
        ],
        dim=1,
    )
    restored[layer_mask] = projection.decode(layer_absorbed[layer_mask])
    return restored


def build_endpoint_hidden_states(
    shell_model: BertForMaskedLM,
    endpoint_params: dict[int, dict[str, torch.Tensor]],
    seeds: list[int],
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, ...]]:
    """Compute residual-stream hidden states for endpoint models on one batch."""
    hidden_states: dict[int, tuple[torch.Tensor, ...]] = {}
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    for seed in seeds:
        params = {
            name: value.to(device)
            for name, value in endpoint_params[seed].items()
        }
        with torch.no_grad():
            output = torch.func.functional_call(
                shell_model.bert,
                params,
                kwargs={
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "output_hidden_states": True,
                },
            )
        hidden_states[seed] = tuple(hidden.detach() for hidden in output.hidden_states)

    return hidden_states


def build_interpolated_layer_params(
    projection: BiasProjection,
    absorbed_i: torch.Tensor,
    absorbed_j: torch.Tensor,
    P_i: torch.Tensor,
    P_j: torch.Tensor,
    layer_idx: int,
    alpha: float,
    config: BertConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Interpolate one encoder layer in canonical space and map it to model i's basis."""
    start, end = encoder_layer_row_bounds(config, layer_idx)
    P_i_inv = torch.linalg.pinv(P_i.float())

    canon_i = absorbed_i[start:end] @ P_i
    canon_j = absorbed_j[start:end] @ P_j
    interp_absorbed = ((1.0 - alpha) * canon_i + alpha * canon_j).float() @ P_i_inv
    layer_rows = restore_layer_rows(projection, interp_absorbed, config, layer_idx, device)
    return deserialize_encoder_layer(layer_rows, config), P_i_inv


def run_interpolated_layer(
    shell_layer: torch.nn.Module,
    layer_params: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Execute a BertLayer with functional_call and normalize its return type."""
    output = torch.func.functional_call(
        shell_layer,
        layer_params,
        args=(hidden_states, attention_mask),
    )
    if isinstance(output, tuple):
        return output[0]
    return output


def invert_soft_permutation(
    canon_module: CanonicalizationModule,
    embeddings: torch.Tensor,
    cond_threshold: float = 1e4,
    tau_decay: float = 0.9,
    max_attempts: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a soft permutation and sharpen it until a true inverse is usable."""
    E_t = embeddings.permute(0, 2, 1)
    Q = E_t @ canon_module.W_q
    K = E_t @ canon_module.W_k
    attention_logits = (Q @ K.permute(0, 2, 1)) / math.sqrt(canon_module.d_model)

    tau = float(canon_module.tau.detach().item())
    best_pair: tuple[torch.Tensor, torch.Tensor] | None = None
    best_cond = float("inf")

    for _ in range(max_attempts):
        P = sinkhorn(attention_logits / tau, n_iters=canon_module.sinkhorn_iters).squeeze(0)
        P_float = P.float()
        try:
            P_inv = torch.linalg.inv(P_float)
        except RuntimeError:
            tau *= tau_decay
            continue

        singular_values = torch.linalg.svdvals(P_float)
        cond = float((singular_values.max() / singular_values.min()).item())
        if cond < best_cond:
            best_cond = cond
            best_pair = (P, P_inv.to(device=P.device, dtype=P.dtype))
        if math.isfinite(cond) and cond <= cond_threshold:
            return P, P_inv.to(device=P.device, dtype=P.dtype)
        tau *= tau_decay

    if best_pair is None:
        raise RuntimeError("Failed to compute an invertible soft permutation matrix")
    return best_pair


def compute_layer_loss(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    absorbed: dict[int, torch.Tensor],
    hidden_states: torch.Tensor,
    seed_to_idx: dict[int, int],
    pair: tuple[int, int],
    layer_idx: int,
    alpha: float,
    extended_mask: torch.Tensor,
    shell_layer: torch.nn.Module,
    config: BertConfig,
    device: torch.device,
    lambda_sharp: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the layer-local surrogate loss for one model pair and one layer."""
    seed_i, seed_j = pair
    vocab_size = config.vocab_size

    absorbed_i = absorbed[seed_i].to(device)
    absorbed_j = absorbed[seed_j].to(device)

    _, P_i = canon_module(absorbed_i[:vocab_size].unsqueeze(0))
    _, P_j = canon_module(absorbed_j[:vocab_size].unsqueeze(0))
    P_i = P_i.squeeze(0)
    P_j = P_j.squeeze(0)

    layer_params, P_i_inv = build_interpolated_layer_params(
        projection,
        absorbed_i,
        absorbed_j,
        P_i,
        P_j,
        layer_idx,
        alpha,
        config,
        device,
    )

    seed_i_idx = seed_to_idx[seed_i]
    seed_j_idx = seed_to_idx[seed_j]
    h_i_in = hidden_states[seed_i_idx, layer_idx].to(device=device, dtype=torch.float32)
    h_j_in = hidden_states[seed_j_idx, layer_idx].to(device=device, dtype=torch.float32)
    h_i_out = hidden_states[seed_i_idx, layer_idx + 1].to(device=device, dtype=torch.float32)
    h_j_out = hidden_states[seed_j_idx, layer_idx + 1].to(device=device, dtype=torch.float32)

    input_can = (1.0 - alpha) * (h_i_in @ P_i) + alpha * (h_j_in @ P_j)
    target_can = (1.0 - alpha) * (h_i_out @ P_i) + alpha * (h_j_out @ P_j)

    exec_input = input_can.float() @ P_i_inv
    pred_exec = run_interpolated_layer(
        shell_layer,
        layer_params,
        exec_input,
        extended_mask,
    )
    pred_can = pred_exec @ P_i

    loss_pred = F.mse_loss(pred_can, target_can)
    loss_sharp = 0.5 * (
        canon_module.row_entropy(P_i.unsqueeze(0))
        + canon_module.row_entropy(P_j.unsqueeze(0))
    )
    total = loss_pred + lambda_sharp * loss_sharp
    metrics = {
        "loss_pred": loss_pred.item(),
        "loss_sharp": loss_sharp.item(),
        "loss_total": total.item(),
        "tau": canon_module.tau.item(),
        "layer": float(layer_idx),
    }
    return total, metrics


def validate(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    absorbed: dict[int, torch.Tensor],
    minibatches: list[dict[str, object]],
    seed_to_idx: dict[int, int],
    shell_model: BertForMaskedLM,
    extended_mask: torch.Tensor,
    config: BertConfig,
    device: torch.device,
    lambda_sharp: float,
    validation_layers: list[int],
    test_seeds: list[int],
) -> float:
    """Evaluate the mean layer-local validation loss over held-out pairs."""
    pairs = list(combinations(test_seeds, 2))
    losses = []

    canon_module.eval()
    with torch.no_grad():
        for minibatch in minibatches:
            hidden_states = minibatch["hidden_states"]
            for pair in pairs:
                for layer_idx in validation_layers:
                    loss, _ = compute_layer_loss(
                        canon_module,
                        projection,
                        absorbed,
                        hidden_states,
                        seed_to_idx,
                        pair,
                        layer_idx,
                        alpha=0.5,
                        extended_mask=extended_mask,
                        shell_layer=shell_model.bert.encoder.layer[layer_idx],
                        config=config,
                        device=device,
                        lambda_sharp=lambda_sharp,
                    )
                    losses.append(loss.item())
    canon_module.train()
    return sum(losses) / max(len(losses), 1)

def evaluate_perplexity(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    absorbed: dict[int, torch.Tensor],
    endpoint_params: dict[int, dict[str, torch.Tensor]],
    head_params: dict[int, dict[str, torch.Tensor]],
    config: BertConfig,
    batches: list[dict[str, torch.Tensor]],
    shell_model: BertForMaskedLM,
    device: torch.device,
    eval_pairs: int,
    test_seeds: list[int],
) -> list[dict[str, float | int]]:
    """Run a compact held-out perplexity check after training."""
    results = []
    pairs = list(combinations(test_seeds, 2))[:eval_pairs]

    for batch_idx, batch in enumerate(batches):
        for seed_i, seed_j in pairs:
            with torch.no_grad():
                absorbed_i = absorbed[seed_i].to(device)
                absorbed_j = absorbed[seed_j].to(device)
                P_i, P_i_inv = invert_soft_permutation(
                    canon_module,
                    absorbed_i[: config.vocab_size].unsqueeze(0),
                )
                P_j, _ = invert_soft_permutation(
                    canon_module,
                    absorbed_j[: config.vocab_size].unsqueeze(0),
                )

                canon_i = absorbed_i @ P_i
                canon_j = absorbed_j @ P_j
                interp_absorbed = (0.5 * canon_i + 0.5 * canon_j).float() @ P_i_inv
                interp_restored = restore_bias_rows_only(projection, interp_absorbed, config, device)
                interp_params = deserialize(interp_restored, config)

                logits_i = run_functional_mlm_logits(
                    shell_model,
                    endpoint_params[seed_i],
                    head_params[seed_i],
                    batch,
                    device,
                )
                logits_j = run_functional_mlm_logits(
                    shell_model,
                    endpoint_params[seed_j],
                    head_params[seed_j],
                    batch,
                    device,
                )
                ensemble_logits = 0.5 * (logits_i + logits_j)
                ensemble_loss = F.cross_entropy(
                    ensemble_logits.view(-1, ensemble_logits.shape[-1]),
                    batch["labels"].to(device).view(-1),
                    ignore_index=-100,
                )
                interp_loss = run_functional_mlm_loss(
                    shell_model,
                    interp_params,
                    head_params[seed_i],
                    batch,
                    device,
                )

            result = {
                "source": f"eval_batch:{batch_idx}",
                "seed_i": seed_i,
                "seed_j": seed_j,
                "interp_ppl": math.exp(float(interp_loss.item())),
                "ensemble_ppl": math.exp(float(ensemble_loss.item())),
                "ratio": math.exp(float(interp_loss.item() - ensemble_loss.item())),
            }
            results.append(result)

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def evaluate_roundtrip_perplexity(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    absorbed: dict[int, torch.Tensor],
    endpoint_params: dict[int, dict[str, torch.Tensor]],
    head_params: dict[int, dict[str, torch.Tensor]],
    config: BertConfig,
    batches: list[dict[str, torch.Tensor]],
    shell_model: BertForMaskedLM,
    device: torch.device,
    test_seeds: list[int],
) -> list[dict[str, float | int]]:
    """Measure perplexity drift after canonicalize-then-uncanonicalize round-trips."""
    results = []

    for batch_idx, batch in enumerate(batches):
        for seed in test_seeds:
            with torch.no_grad():
                absorbed_seed = absorbed[seed].to(device)
                P, P_inv = invert_soft_permutation(
                    canon_module,
                    absorbed_seed[: config.vocab_size].unsqueeze(0),
                )

                roundtrip_absorbed = (absorbed_seed @ P).float() @ P_inv
                roundtrip_restored = restore_bias_rows_only(projection, roundtrip_absorbed, config, device)
                roundtrip_params = deserialize(roundtrip_restored, config)

                baseline_logits = run_functional_mlm_logits(
                    shell_model,
                    endpoint_params[seed],
                    head_params[seed],
                    batch,
                    device,
                )
                baseline_loss = F.cross_entropy(
                    baseline_logits.view(-1, baseline_logits.shape[-1]),
                    batch["labels"].to(device).view(-1),
                    ignore_index=-100,
                )
                roundtrip_loss = run_functional_mlm_loss(
                    shell_model,
                    roundtrip_params,
                    head_params[seed],
                    batch,
                    device,
                )

            results.append(
                {
                    "source": f"eval_batch:{batch_idx}",
                    "seed": seed,
                    "roundtrip_ppl": math.exp(float(roundtrip_loss.item())),
                    "baseline_ppl": math.exp(float(baseline_loss.item())),
                    "ratio": math.exp(float(roundtrip_loss.item() - baseline_loss.item())),
                }
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def build_eval_batches(
    token_ids: list[int],
    tokenizer: BertTokenizer,
    num_batches: int,
    num_samples: int,
    max_length: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Build deterministic MLM batches for full-model perplexity evaluation."""
    return [
        sample_batch(
            token_ids,
            tokenizer,
            num_samples,
            max_length,
            seed=seed + batch_idx,
        )
        for batch_idx in range(max(num_batches, 1))
    ]


def load_eval_head_params(seeds: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    """Load held-out MLM heads once on CPU for functional evaluation."""
    heads = {}
    for seed in seeds:
        model = load_multibert_model(seed).eval()
        heads[seed] = extract_non_bert_params(model)
        del model
    return heads


def main() -> None:
    """Train the v2 canonicalization module and optionally run held-out evaluation."""
    args = parse_args()
    if args.smoke_test:
        args.epochs = 1
        args.steps_per_epoch = 2
        args.eval_pairs = 1
        args.num_samples = 2
        args.max_length = 32
        args.steps_per_shard = 2
        args.train_live_shards = 2
        args.validation_batches = 1
        args.final_eval_batches = 1
        args.validation_layers = [0]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_seeds = TRAIN_SEEDS[:4] if args.smoke_test else TRAIN_SEEDS
    test_seeds = TEST_SEEDS[:2] if args.smoke_test else TEST_SEEDS
    all_seeds = sorted(set(train_seeds + test_seeds))

    print("Loading serialized checkpoints...")
    serialized, config = load_serialized_models(all_seeds, SERIALIZED_CACHE)

    print("Loading frozen bias projection...")
    projection = load_frozen_bias_projection(BIAS_CKPT, config.hidden_size).to(device)

    print("Pre-encoding absorbed parameter matrices...")
    absorbed = {
        seed: absorb_bias_rows_only(projection, serialized[seed], config, device).cpu()
        for seed in all_seeds
    }

    print("Loading activation dataset metadata...")
    metadata = load_activation_dataset_metadata(args.activation_dataset_dir)
    train_shard_paths = list_activation_shards(args.activation_dataset_dir, "train")
    validation_shard_paths = list_activation_shards(args.activation_dataset_dir, "validation")
    dataset_max_length = int(metadata["max_length"])
    if args.max_length != dataset_max_length:
        raise ValueError(
            "Dataset sequence length does not match run configuration: "
            f"dataset has max_length={dataset_max_length}; got max_length={args.max_length}."
        )

    train_seed_to_idx = {
        int(seed): idx for idx, seed in enumerate(metadata["splits"]["train"]["seed_order"])
    }
    validation_seed_to_idx = {
        int(seed): idx for idx, seed in enumerate(metadata["splits"]["validation"]["seed_order"])
    }
    missing_train = [seed for seed in train_seeds if seed not in train_seed_to_idx]
    missing_validation = [seed for seed in test_seeds if seed not in validation_seed_to_idx]
    if missing_train or missing_validation:
        raise ValueError(
            f"Activation dataset is missing required seeds. train={missing_train}, validation={missing_validation}"
        )

    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
    print("Loading WikiText validation token stream for in-training perplexity...")
    validation_eval_token_ids = load_wikitext_token_ids(tokenizer, split="validation")
    print("Loading WikiText test token stream for final perplexity report...")
    test_eval_token_ids = load_wikitext_token_ids(tokenizer, split="test")
    print("Caching held-out evaluation params...")
    eval_endpoint_params = {seed: deserialize(serialized[seed], config) for seed in test_seeds}
    eval_head_params = load_eval_head_params(test_seeds)

    shell_model = load_multibert_model(train_seeds[0]).eval().to(device)
    wandb_run = init_wandb(args, config)
    extended_mask = get_extended_attention_mask(
        shell_model,
        torch.ones((args.num_samples, args.max_length), dtype=torch.long, device=device),
        dtype=torch.float32,
    )
    canon_module = CanonicalizationModule(
        vocab_size=config.vocab_size,
        d_model=config.hidden_size,
        sinkhorn_iters=args.sinkhorn_iters,
        tau_init=args.tau_init,
    ).to(device)
    optimizer = torch.optim.AdamW(
        canon_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_pairs = list(combinations(train_seeds, 2))
    best_val = float("inf")
    history = []
    random.shuffle(train_shard_paths)
    train_shard_idx = 0
    current_train_shards: list[tuple[str, dict[str, torch.Tensor | list[int]]]] = []
    current_train_shard_name = ""
    steps_on_current_shard = args.steps_per_shard

    epoch_bar = tqdm(range(args.epochs), desc="Canonicalizing heads", unit="epoch")

    for epoch in epoch_bar:
        canon_module.train()
        random.shuffle(train_pairs)
        epoch_metrics = []
        random.shuffle(train_shard_paths)
        train_shard_idx = 0
        current_train_shards = []
        current_train_shard_name = ""
        steps_on_current_shard = args.steps_per_shard
        step_bar = tqdm(
            range(args.steps_per_epoch),
            desc=f"Epoch {epoch + 1:02d} | aligning bases",
            unit="step",
            leave=False,
        )

        for step in step_bar:
            if not current_train_shards or steps_on_current_shard >= args.steps_per_shard:
                if current_train_shards:
                    current_train_shards = release_training_shard_pool(current_train_shards)
                current_train_shards, train_shard_idx = load_training_shard_pool(
                    train_shard_paths,
                    train_shard_idx,
                    args.train_live_shards,
                )
                current_train_shard_name = "+".join(name for name, _ in current_train_shards)
                steps_on_current_shard = 0

            pair = train_pairs[step % len(train_pairs)]
            layer_idx = random.randrange(config.num_hidden_layers)
            train_minibatch = sample_examples_from_shards(
                current_train_shards,
                args.num_samples,
            )
            hidden_states = train_minibatch["hidden_states"]
            loss, metrics = compute_layer_loss(
                canon_module,
                projection,
                absorbed,
                hidden_states,
                train_seed_to_idx,
                pair,
                layer_idx,
                alpha=args.alpha,
                extended_mask=extended_mask,
                shell_layer=shell_model.bert.encoder.layer[layer_idx],
                config=config,
                device=device,
                lambda_sharp=args.lambda_sharp,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_metrics.append(metrics)
            steps_on_current_shard += 1
            step_bar.set_postfix(
                shard=current_train_shard_name,
                examples=args.num_samples,
                pair=f"{pair[0]}-{pair[1]}",
                layer=layer_idx,
                loss=f"{metrics['loss_total']:.4f}",
                tau=f"{metrics['tau']:.3f}",
            )

        step_bar.close()

        if current_train_shards:
            current_train_shards = release_training_shard_pool(current_train_shards)

        validation_minibatches = sample_activation_minibatches(
            validation_shard_paths,
            args.validation_batches,
            args.num_samples,
        )

        val_loss = validate(
            canon_module,
            projection,
            absorbed,
            validation_minibatches,
            validation_seed_to_idx,
            shell_model,
            extended_mask,
            config,
            device,
            lambda_sharp=args.lambda_sharp,
            validation_layers=args.validation_layers,
            test_seeds=test_seeds,
        )

        train_loss = sum(metric["loss_total"] for metric in epoch_metrics) / len(epoch_metrics)
        train_pred = sum(metric["loss_pred"] for metric in epoch_metrics) / len(epoch_metrics)
        train_sharp = sum(metric["loss_sharp"] for metric in epoch_metrics) / len(epoch_metrics)
        summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_pred": train_pred,
            "train_loss_sharp": train_sharp,
            "val_loss": val_loss,
            "tau": canon_module.tau.item(),
        }

        if not args.skip_perplexity_eval:
            eval_batches = build_eval_batches(
                validation_eval_token_ids,
                tokenizer,
                num_batches=1,
                num_samples=args.num_samples,
                max_length=args.max_length,
                seed=args.seed + 100_000 + epoch,
            )
            val_perplexity_results = evaluate_perplexity(
                canon_module,
                projection,
                absorbed,
                eval_endpoint_params,
                eval_head_params,
                config,
                eval_batches,
                shell_model,
                device,
                eval_pairs=args.eval_pairs,
                test_seeds=test_seeds,
            )
            summary.update(summarize_perplexity_results(val_perplexity_results))
            roundtrip_results = evaluate_roundtrip_perplexity(
                canon_module,
                projection,
                absorbed,
                eval_endpoint_params,
                eval_head_params,
                config,
                eval_batches,
                shell_model,
                device,
                test_seeds=test_seeds,
            )
            summary.update(summarize_roundtrip_results(roundtrip_results))

        history.append(summary)
        epoch_bar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            tau=f"{canon_module.tau.item():.3f}",
            best=f"{best_val:.4f}",
        )
        print(
            f"epoch {epoch:02d}  train={train_loss:.5f}  pred={train_pred:.5f}  "
            f"sharp={train_sharp:.5f}  val={val_loss:.5f}  tau={canon_module.tau.item():.4f}"
        )
        wandb_log(
            wandb_run,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_loss_pred": train_pred,
                "train_loss_sharp": train_sharp,
                "val_loss": val_loss,
                "tau": canon_module.tau.item(),
                **({
                    "val_interp_ppl": summary["val_interp_ppl"],
                    "val_ensemble_ppl": summary["val_ensemble_ppl"],
                    "val_ppl_ratio": summary["val_ppl_ratio"],
                    "val_roundtrip_ppl": summary["val_roundtrip_ppl"],
                    "val_roundtrip_baseline_ppl": summary["val_roundtrip_baseline_ppl"],
                    "val_roundtrip_ppl_ratio": summary["val_roundtrip_ppl_ratio"],
                } if "val_interp_ppl" in summary else {}),
            },
            step=epoch,
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(canon_module.state_dict(), RESULTS_DIR / "canonicalization_module.pt")

    epoch_bar.close()

    with open(RESULTS_DIR / "training_log.json", "w", encoding="ascii") as handle:
        json.dump(history, handle, indent=2)

    if args.skip_perplexity_eval:
        if wandb_run is not None:
            wandb.finish()
        return

    eval_batches = build_eval_batches(
        test_eval_token_ids,
        tokenizer,
        num_batches=args.final_eval_batches,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed + 200_000,
    )
    best_path = RESULTS_DIR / "canonicalization_module.pt"
    canon_module.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    canon_module = canon_module.to(device).eval()
    perplexity_results = evaluate_perplexity(
        canon_module,
        projection,
        absorbed,
        eval_endpoint_params,
        eval_head_params,
        config,
        eval_batches,
        shell_model,
        device,
        eval_pairs=args.eval_pairs,
        test_seeds=test_seeds,
    )

    with open(RESULTS_DIR / "perplexity_results.json", "w", encoding="ascii") as handle:
        json.dump(perplexity_results, handle, indent=2)

    for result in perplexity_results:
        print(
            f"pair ({result['seed_i']}, {result['seed_j']})  "
            f"interp={result['interp_ppl']:.2f}  "
            f"ensemble={result['ensemble_ppl']:.2f}  "
            f"ratio={result['ratio']:.3f}"
        )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()