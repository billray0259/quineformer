"""Minimal Bias Absorption Validation Experiment.

This pared-down variant of run_v1.py only:
1. loads and caches serialized MultiBERT checkpoints,
2. trains the shared linear bias-absorption projection, and
3. evaluates MLM perplexity on reconstructed validation models.
"""

import json
import time
from pathlib import Path

import torch
from transformers import BertTokenizer

from quineformer.bias_absorption import bias_carrying_mask, train_projection

from run_v1 import (
    TEST_SEEDS,
    TRAIN_SEEDS,
    evaluate_mlm_perplexity,
    get_reference_batch,
    load_and_serialize_all,
)


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_v1_min"


def make_serializable(obj):
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return obj


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("BIAS ABSORPTION VALIDATION EXPERIMENT v1 (MINIMAL)")
    print("=" * 70)

    print("\n▸ Loading and serializing MultiBERTs...")
    serialized, config = load_and_serialize_all()
    d_model = config.hidden_size
    mask = bias_carrying_mask(config)
    train_data = torch.cat([serialized[seed][mask] for seed in TRAIN_SEEDS], dim=0)
    print(
        f"  Training data: {train_data.shape[0]:,} bias-carrying vectors "
        f"from {len(TRAIN_SEEDS)} seeds"
    )

    print("\n▸ Preparing reference text batch for MLM evaluation...")
    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")
    ref_batch = get_reference_batch(tokenizer)
    print(f"  Reference batch: {ref_batch['input_ids'].shape}")

    print("\n▸ Training shared linear projection...")
    projection = train_projection(train_data, d_model, n_epochs=1, log_every=1, lr=3.25e-4, tag="shared")
    projection_path = RESULTS_DIR / "projection_shared.pt"
    torch.save(projection.state_dict(), projection_path)
    print(f"  Saved projection to {projection_path}")

    print("\n▸ Evaluating reconstructed validation models...")
    perplexity_results = evaluate_mlm_perplexity(
        projection,
        serialized,
        TEST_SEEDS,
        config,
        ref_batch,
        tag="learned",
    )

    mean_ratio = sum(result["ratio"] for result in perplexity_results.values()) / len(perplexity_results)
    results = {
        "train_seeds": TRAIN_SEEDS,
        "test_seeds": TEST_SEEDS,
        "mean_perplexity_ratio": mean_ratio,
        "perplexity": perplexity_results,
        "elapsed_seconds": time.time() - t0,
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as handle:
        json.dump(make_serializable(results), handle, indent=2)

    print("\n▸ Summary")
    print(f"  Mean reconstructed/original perplexity ratio: {mean_ratio:.4f}")
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()