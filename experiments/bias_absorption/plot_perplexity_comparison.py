"""Plot perplexity comparisons for original and reconstructed MultiBERTs.

Evaluates all baselines on the same MLM reference batch so the comparison is
directly apples-to-apples:
- Original model
- No-bias baseline
- V1 full reconstruction
- V1 bias-only reconstruction
- V2 full reconstruction (if checkpoint exists)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer

from quineformer.serialization import deserialize, vector_component_labels
from run_v1 import BIAS_CARRYING_TYPES, BiasProjection, TEST_SEEDS, compute_mlm_perplexity, get_reference_batch
from run_v2 import (
    DEVICE,
    LEGACY_RESULTS_DIR,
    LEGACY_V1_PROJECTION_PATH,
    RESULTS_DIR,
    SERIALIZED_CACHE,
    LazySerializedList,
    V1_PROJECTION_PATH,
    resolve_existing_path,
    load_multibert_model,
)

SCRIPT_DIR = Path(__file__).resolve().parent
V2_PROJECTION_PATH = resolve_existing_path(
    RESULTS_DIR / "projection_v2.pt",
    LEGACY_RESULTS_DIR / "projection_v2.pt",
)
OUTPUT_JSON = RESULTS_DIR / "perplexity_comparison.json"
OUTPUT_PNG = RESULTS_DIR / "perplexity_comparison.png"

SERIES_ORDER = ["original", "no_bias", "v1", "v1_bias_only", "v2"]
SERIES_STYLE = {
    "original": {"label": "Original", "color": "#222222"},
    "no_bias": {"label": "No Bias", "color": "#55a868"},
    "v1": {"label": "V1", "color": "#c44e52"},
    "v1_bias_only": {"label": "V1 Bias Only", "color": "#dd8452"},
    "v2": {"label": "V2", "color": "#4c72b0"},
}


def load_projection(checkpoint_path: Path, hidden_size: int) -> BiasProjection:
    projection = BiasProjection(hidden_size)
    projection.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    return projection.eval()


@torch.no_grad()
def reconstruct_model_from_serialized(serialized, config, seed: int):
    params = deserialize(serialized, config)
    model = load_multibert_model(seed)
    model.bert.load_state_dict(params)
    return model.eval()


@torch.no_grad()
def apply_projection(projection: BiasProjection, serialized: torch.Tensor) -> torch.Tensor:
    return projection.to(DEVICE).eval()(serialized.to(DEVICE)).cpu()


@torch.no_grad()
def make_no_bias_serialized(serialized: torch.Tensor, hidden_size: int) -> torch.Tensor:
    out = serialized.clone()
    out[:, hidden_size] = 0.0
    return out


@torch.no_grad()
def make_v1_bias_only_serialized(
    projection: BiasProjection,
    serialized: torch.Tensor,
    bias_mask: torch.Tensor,
) -> torch.Tensor:
    out = serialized.clone()
    out[bias_mask] = apply_projection(projection, serialized[bias_mask])
    return out


def plot_results(results: dict) -> None:
    seeds = [f"{seed}" for seed in TEST_SEEDS]
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    for key in SERIES_ORDER:
        if key not in results["means"]:
            continue
        style = SERIES_STYLE[key]
        values = [results["by_seed"][f"seed_{seed}"][key]["ppl"] for seed in TEST_SEEDS]
        ratios = [results["by_seed"][f"seed_{seed}"][key]["ratio"] for seed in TEST_SEEDS]

        axes[0].plot(seeds, values, marker="o", linewidth=2, color=style["color"], label=style["label"])
        axes[1].plot(seeds, ratios, marker="o", linewidth=2, color=style["color"], label=style["label"])

    axes[0].set_title("MLM Perplexity by Seed")
    axes[0].set_ylabel("Perplexity")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Relative Perplexity vs Original")
    axes[1].set_xlabel("Held-out seed")
    axes[1].set_ylabel("Perplexity ratio")
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    summary_lines = []
    for key in SERIES_ORDER:
        if key == "original" or key not in results["means"]:
            continue
        mean_ratio = results["means"][key]["ratio"]
        summary_lines.append(f"{SERIES_STYLE[key]['label']}: {mean_ratio:.4f}x")
    fig.suptitle("Perplexity Comparison on Shared Reference Batch\n" + " | ".join(summary_lines))
    fig.savefig(OUTPUT_PNG, dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    serialized_list = LazySerializedList(SERIALIZED_CACHE, 25)
    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
    ref_batch = get_reference_batch(tokenizer)
    config = load_multibert_model(TEST_SEEDS[0]).config

    labels = vector_component_labels(config)
    bias_mask = torch.tensor([label in BIAS_CARRYING_TYPES for label in labels], dtype=torch.bool)

    projections = {}
    resolved_v1_path = resolve_existing_path(V1_PROJECTION_PATH, LEGACY_V1_PROJECTION_PATH)
    if resolved_v1_path.exists():
        projections["v1"] = load_projection(resolved_v1_path, config.hidden_size)
    if V2_PROJECTION_PATH.exists():
        projections["v2"] = load_projection(V2_PROJECTION_PATH, config.hidden_size)

    results = {"by_seed": {}, "means": {}}

    for seed in TEST_SEEDS:
        print(f"seed {seed}")
        original_serialized = serialized_list[seed]
        original_model = load_multibert_model(seed).eval().to(DEVICE)
        original_ppl = compute_mlm_perplexity(original_model, ref_batch)

        seed_result = {
            "original": {"ppl": original_ppl, "ratio": 1.0},
        }

        no_bias_model = reconstruct_model_from_serialized(
            make_no_bias_serialized(original_serialized, config.hidden_size),
            config,
            seed,
        ).to(DEVICE)
        no_bias_ppl = compute_mlm_perplexity(no_bias_model, ref_batch)
        seed_result["no_bias"] = {"ppl": no_bias_ppl, "ratio": no_bias_ppl / original_ppl}
        del no_bias_model

        if "v1" in projections:
            v1_serialized = apply_projection(projections["v1"], original_serialized)
            v1_model = reconstruct_model_from_serialized(v1_serialized, config, seed).to(DEVICE)

            v1_ppl = compute_mlm_perplexity(v1_model, ref_batch)
            seed_result["v1"] = {"ppl": v1_ppl, "ratio": v1_ppl / original_ppl}
            del v1_model

            v1_bias_only_serialized = make_v1_bias_only_serialized(
                projections["v1"], original_serialized, bias_mask
            )
            v1_bias_only_model = reconstruct_model_from_serialized(
                v1_bias_only_serialized, config, seed
            ).to(DEVICE)
            v1_bias_only_ppl = compute_mlm_perplexity(v1_bias_only_model, ref_batch)
            seed_result["v1_bias_only"] = {
                "ppl": v1_bias_only_ppl,
                "ratio": v1_bias_only_ppl / original_ppl,
            }
            del v1_bias_only_model

        if "v2" in projections:
            v2_serialized = apply_projection(projections["v2"], original_serialized)
            v2_model = reconstruct_model_from_serialized(v2_serialized, config, seed).to(DEVICE)
            v2_ppl = compute_mlm_perplexity(v2_model, ref_batch)
            seed_result["v2"] = {"ppl": v2_ppl, "ratio": v2_ppl / original_ppl}
            del v2_model

        results["by_seed"][f"seed_{seed}"] = seed_result
        del original_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for key in SERIES_ORDER:
        if key == "original":
            results["means"][key] = {"ppl": None, "ratio": 1.0}
            continue
        ratios = []
        ppls = []
        for seed in TEST_SEEDS:
            seed_key = f"seed_{seed}"
            if key not in results["by_seed"][seed_key]:
                continue
            ratios.append(results["by_seed"][seed_key][key]["ratio"])
            ppls.append(results["by_seed"][seed_key][key]["ppl"])
        if ratios:
            results["means"][key] = {
                "ppl": sum(ppls) / len(ppls),
                "ratio": sum(ratios) / len(ratios),
            }

    plot_results(results)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    for key in SERIES_ORDER:
        if key in results["means"]:
            print(f"{SERIES_STYLE[key]['label']}: mean ratio={results['means'][key]['ratio']:.4f}")
    print(f"saved {OUTPUT_PNG}")
    print(f"saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()