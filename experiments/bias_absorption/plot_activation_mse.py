"""Visualize how parameter errors propagate through BERT depth.

Compares original MultiBERT checkpoints against three baselines:
1. V1 reconstructed models
2. V2 reconstructed models
3. No-bias models (bias dimension zeroed before deserialization)

For each comparison the script reports:
- encoder-layer parameter MSE
- hidden-state MSE across depth
- post-attention block activation MSE per layer
- post-MLP block activation MSE per layer

This makes it easier to test whether small parameter-space errors compound into
larger activation drift deeper in the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertForMaskedLM, BertTokenizer

from quineformer.serialization import deserialize, vector_component_labels
from run_v1 import BIAS_CARRYING_TYPES, BiasProjection, TEST_SEEDS, get_reference_batch
from run_v2 import (
    DEVICE,
    LEGACY_RESULTS_DIR,
    RESULTS_DIR,
    SERIALIZED_CACHE,
    LazySerializedList,
    resolve_existing_path,
    V1_PROJECTION_PATH,
    load_multibert_model,
)

V2_PROJECTION_PATH = resolve_existing_path(
    RESULTS_DIR / "projection_v2.pt",
    LEGACY_RESULTS_DIR / "projection_v2.pt",
)
OUTPUT_PNG = RESULTS_DIR / "layerwise_activation_mse.png"
OUTPUT_JSON = RESULTS_DIR / "layerwise_activation_mse.json"
LEGACY_V1_PROJECTION_PATH = (
    Path(__file__).resolve().parent / "experiments" / "bias_absorption" / "results_v1" / "projection_shared.pt"
)

LABEL_STYLES = {
    "v1": {"color": "#c44e52", "name": "V1"},
    "v1_bias_only": {"color": "#dd8452", "name": "V1 Bias Only"},
    "v2": {"color": "#4c72b0", "name": "V2"},
    "no_bias": {"color": "#55a868", "name": "No Bias"},
}


def load_projection(checkpoint_path: Path, hidden_size: int) -> BiasProjection:
    projection = BiasProjection(hidden_size)
    projection.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    return projection.eval()


@torch.no_grad()
def reconstruct_bert(
    projection: BiasProjection,
    serialized: Tensor,
    config,
) -> BertForMaskedLM:
    projection = projection.to(DEVICE).eval()
    reconstructed = projection(serialized.to(DEVICE)).cpu()
    params = deserialize(reconstructed, config)
    model = BertForMaskedLM(config).eval()
    model.bert.load_state_dict(params)
    return model


@torch.no_grad()
def reconstruct_no_bias_bert(serialized: Tensor, config) -> BertForMaskedLM:
    no_bias = serialized.clone()
    no_bias[:, config.hidden_size] = 0.0
    params = deserialize(no_bias, config)
    model = BertForMaskedLM(config).eval()
    model.bert.load_state_dict(params)
    return model


def compute_layer_parameter_mse(
    original_serialized: Tensor,
    reconstructed_serialized: Tensor,
    config,
) -> list[float]:
    original_params = deserialize(original_serialized, config)
    reconstructed_params = deserialize(reconstructed_serialized, config)

    layer_mse: list[float] = []
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"encoder.layer.{layer_idx}."
        sq_error = 0.0
        n_values = 0
        for key, orig_tensor in original_params.items():
            if not key.startswith(prefix):
                continue
            recon_tensor = reconstructed_params[key]
            diff = orig_tensor - recon_tensor
            sq_error += diff.pow(2).sum().item()
            n_values += diff.numel()
        layer_mse.append(sq_error / max(n_values, 1))

    return layer_mse


@torch.no_grad()
def get_reconstructed_serialized(
    label: str,
    serialized: Tensor,
    config,
    projection: BiasProjection | None,
    bias_mask: Tensor | None,
) -> Tensor:
    if label == "no_bias":
        reconstructed = serialized.clone()
        reconstructed[:, config.hidden_size] = 0.0
        return reconstructed

    if label == "v1_bias_only":
        assert projection is not None
        assert bias_mask is not None
        reconstructed = serialized.clone()
        transformed = projection.to(DEVICE).eval()(serialized[bias_mask].to(DEVICE)).cpu()
        reconstructed[bias_mask] = transformed
        return reconstructed

    assert projection is not None
    return projection.to(DEVICE).eval()(serialized.to(DEVICE)).cpu()


@torch.no_grad()
def compute_hidden_state_mse(
    original_model: BertForMaskedLM,
    reconstructed_model: BertForMaskedLM,
    batch: dict[str, Tensor],
) -> tuple[list[float], list[float]]:
    original_model = original_model.to(DEVICE).eval()
    reconstructed_model = reconstructed_model.to(DEVICE).eval()

    bert_inputs = {
        "input_ids": batch["input_ids"].to(DEVICE),
        "attention_mask": batch["attention_mask"].to(DEVICE),
        "output_hidden_states": True,
        "return_dict": True,
    }

    original_outputs = original_model.bert(**bert_inputs)
    reconstructed_outputs = reconstructed_model.bert(**bert_inputs)

    mse_per_state: list[float] = []
    relative_mse_per_state: list[float] = []
    for orig_state, recon_state in zip(
        original_outputs.hidden_states,
        reconstructed_outputs.hidden_states,
        strict=True,
    ):
        diff = orig_state - recon_state
        mse = diff.pow(2).mean().item()
        rel = mse / (orig_state.pow(2).mean().item() + 1e-12)
        mse_per_state.append(mse)
        relative_mse_per_state.append(rel)

    return mse_per_state, relative_mse_per_state


@torch.no_grad()
def compute_block_output_mse(
    original_model: BertForMaskedLM,
    reconstructed_model: BertForMaskedLM,
    batch: dict[str, Tensor],
) -> tuple[list[float], list[float]]:
    original_model = original_model.to(DEVICE).eval()
    reconstructed_model = reconstructed_model.to(DEVICE).eval()

    def collect_outputs(model: BertForMaskedLM) -> tuple[list[Tensor], list[Tensor]]:
        attn_outputs: list[Tensor | None] = [None] * model.config.num_hidden_layers
        mlp_outputs: list[Tensor | None] = [None] * model.config.num_hidden_layers
        hooks: list = []

        def make_hook(store: list[Tensor | None], idx: int):
            def hook(_module: nn.Module, _inputs, output: Tensor):
                store[idx] = output.detach().cpu()
            return hook

        for layer_idx, layer in enumerate(model.bert.encoder.layer):
            hooks.append(
                layer.attention.output.LayerNorm.register_forward_hook(
                    make_hook(attn_outputs, layer_idx)
                )
            )
            hooks.append(
                layer.output.LayerNorm.register_forward_hook(
                    make_hook(mlp_outputs, layer_idx)
                )
            )

        model.bert(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            return_dict=True,
        )

        for hook in hooks:
            hook.remove()

        return attn_outputs, mlp_outputs

    orig_attn, orig_mlp = collect_outputs(original_model)
    recon_attn, recon_mlp = collect_outputs(reconstructed_model)

    attn_mse = [
        (orig - recon).pow(2).mean().item()
        for orig, recon in zip(orig_attn, recon_attn, strict=True)
    ]
    mlp_mse = [
        (orig - recon).pow(2).mean().item()
        for orig, recon in zip(orig_mlp, recon_mlp, strict=True)
    ]
    return attn_mse, mlp_mse


def summarize(values_by_seed: dict[int, list[float]]) -> tuple[list[float], list[float]]:
    values = torch.tensor([values_by_seed[seed] for seed in sorted(values_by_seed)])
    return values.mean(dim=0).tolist(), values.std(dim=0, unbiased=False).tolist()


def plot_curves(metrics: dict) -> None:
    param_layers = list(range(1, 13))
    hidden_layers = list(range(0, 13))
    hidden_labels = ["emb"] + [str(i) for i in range(1, 13)]

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), constrained_layout=True)

    for label in ["v1", "v1_bias_only", "v2", "no_bias"]:
        if label not in metrics:
            continue
        color = LABEL_STYLES[label]["color"]
        legend_name = LABEL_STYLES[label]["name"]

        param_mean = metrics[label]["parameter_mse_mean"]
        param_std = metrics[label]["parameter_mse_std"]
        hidden_mean = metrics[label]["hidden_mse_mean"]
        hidden_std = metrics[label]["hidden_mse_std"]
        attn_mean = metrics[label]["attn_block_mse_mean"]
        attn_std = metrics[label]["attn_block_mse_std"]
        mlp_mean = metrics[label]["mlp_block_mse_mean"]
        mlp_std = metrics[label]["mlp_block_mse_std"]

        axes[0].plot(param_layers, param_mean, marker="o", color=color, label=legend_name)
        axes[0].fill_between(
            param_layers,
            [m - s for m, s in zip(param_mean, param_std, strict=True)],
            [m + s for m, s in zip(param_mean, param_std, strict=True)],
            color=color,
            alpha=0.18,
        )

        axes[1].plot(hidden_layers, hidden_mean, marker="o", color=color, label=legend_name)
        axes[1].fill_between(
            hidden_layers,
            [m - s for m, s in zip(hidden_mean, hidden_std, strict=True)],
            [m + s for m, s in zip(hidden_mean, hidden_std, strict=True)],
            color=color,
            alpha=0.18,
        )

        axes[2].plot(param_layers, attn_mean, marker="o", color=color, label=legend_name)
        axes[2].fill_between(
            param_layers,
            [m - s for m, s in zip(attn_mean, attn_std, strict=True)],
            [m + s for m, s in zip(attn_mean, attn_std, strict=True)],
            color=color,
            alpha=0.18,
        )

        axes[3].plot(param_layers, mlp_mean, marker="o", color=color, label=legend_name)
        axes[3].fill_between(
            param_layers,
            [m - s for m, s in zip(mlp_mean, mlp_std, strict=True)],
            [m + s for m, s in zip(mlp_mean, mlp_std, strict=True)],
            color=color,
            alpha=0.18,
        )

    axes[0].set_title("Encoder Layer Parameter MSE")
    axes[0].set_xlabel("Encoder layer")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Hidden-State MSE Across Depth")
    axes[1].set_xlabel("Hidden state")
    axes[1].set_ylabel("MSE")
    axes[1].set_yscale("log")
    axes[1].set_xticks(hidden_layers)
    axes[1].set_xticklabels(hidden_labels)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Post-Attention Block Output MSE")
    axes[2].set_xlabel("Encoder layer")
    axes[2].set_ylabel("MSE")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].set_title("Post-MLP Block Output MSE")
    axes[3].set_xlabel("Encoder layer")
    axes[3].set_ylabel("MSE")
    axes[3].set_yscale("log")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.suptitle("Original vs Reconstructed BERT: Small Weight Errors vs Activation Drift")
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    serialized = LazySerializedList(SERIALIZED_CACHE, 25)
    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
    batch = get_reference_batch(tokenizer, num_samples=4, max_length=128)

    original_seed0 = load_multibert_model(TEST_SEEDS[0])
    config = original_seed0.config
    del original_seed0
    labels = vector_component_labels(config)
    bias_mask = torch.tensor([label in BIAS_CARRYING_TYPES for label in labels], dtype=torch.bool)

    projections: dict[str, BiasProjection | None] = {"no_bias": None}
    resolved_v1_path = resolve_existing_path(V1_PROJECTION_PATH, LEGACY_V1_PROJECTION_PATH)
    if resolved_v1_path.exists():
        v1_projection = load_projection(resolved_v1_path, config.hidden_size)
        projections["v1"] = v1_projection
        projections["v1_bias_only"] = v1_projection
    if V2_PROJECTION_PATH.exists():
        projections["v2"] = load_projection(V2_PROJECTION_PATH, config.hidden_size)

    if len(projections) == 1:
        raise FileNotFoundError("No projection checkpoints found for V1 or V2.")

    metrics: dict[str, dict] = {}

    for label, projection in projections.items():
        print(f"\n=== {LABEL_STYLES[label]['name']} ===")
        param_by_seed: dict[int, list[float]] = {}
        hidden_by_seed: dict[int, list[float]] = {}
        rel_hidden_by_seed: dict[int, list[float]] = {}
        attn_by_seed: dict[int, list[float]] = {}
        mlp_by_seed: dict[int, list[float]] = {}

        for seed in TEST_SEEDS:
            print(f"  seed {seed}")
            original_model = load_multibert_model(seed)
            if label == "no_bias":
                reconstructed_model = reconstruct_no_bias_bert(serialized[seed], config)
            elif label == "v1_bias_only":
                reconstructed_serialized = get_reconstructed_serialized(
                    label, serialized[seed], config, projection, bias_mask
                )
                params = deserialize(reconstructed_serialized, config)
                reconstructed_model = load_multibert_model(seed)
                reconstructed_model.bert.load_state_dict(params)
            else:
                reconstructed_model = reconstruct_bert(projection, serialized[seed], config)

            if label != "v1_bias_only":
                reconstructed_serialized = get_reconstructed_serialized(
                    label, serialized[seed], config, projection, bias_mask
                )

            param_by_seed[seed] = compute_layer_parameter_mse(
                serialized[seed], reconstructed_serialized, config
            )
            hidden_mse, hidden_rel = compute_hidden_state_mse(
                original_model,
                reconstructed_model,
                batch,
            )
            attn_mse, mlp_mse = compute_block_output_mse(
                original_model,
                reconstructed_model,
                batch,
            )
            hidden_by_seed[seed] = hidden_mse
            rel_hidden_by_seed[seed] = hidden_rel
            attn_by_seed[seed] = attn_mse
            mlp_by_seed[seed] = mlp_mse

            del original_model, reconstructed_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        param_mean, param_std = summarize(param_by_seed)
        hidden_mean, hidden_std = summarize(hidden_by_seed)
        rel_hidden_mean, rel_hidden_std = summarize(rel_hidden_by_seed)
        attn_mean, attn_std = summarize(attn_by_seed)
        mlp_mean, mlp_std = summarize(mlp_by_seed)

        metrics[label] = {
            "parameter_mse_by_seed": param_by_seed,
            "parameter_mse_mean": param_mean,
            "parameter_mse_std": param_std,
            "hidden_mse_by_seed": hidden_by_seed,
            "hidden_mse_mean": hidden_mean,
            "hidden_mse_std": hidden_std,
            "relative_hidden_mse_by_seed": rel_hidden_by_seed,
            "relative_hidden_mse_mean": rel_hidden_mean,
            "relative_hidden_mse_std": rel_hidden_std,
            "attn_block_mse_by_seed": attn_by_seed,
            "attn_block_mse_mean": attn_mean,
            "attn_block_mse_std": attn_std,
            "mlp_block_mse_by_seed": mlp_by_seed,
            "mlp_block_mse_mean": mlp_mean,
            "mlp_block_mse_std": mlp_std,
        }

        print(f"    param mse first/last layer: {param_mean[0]:.3e} -> {param_mean[-1]:.3e}")
        print(f"    hidden mse emb/final:       {hidden_mean[0]:.3e} -> {hidden_mean[-1]:.3e}")
        print(f"    attn block mse first/last:  {attn_mean[0]:.3e} -> {attn_mean[-1]:.3e}")
        print(f"    mlp block mse first/last:   {mlp_mean[0]:.3e} -> {mlp_mean[-1]:.3e}")

    plot_curves(metrics)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved plot: {OUTPUT_PNG}")
    print(f"Saved data: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()