"""Evaluate saved canonicalization checkpoints on held-out validation pairs.

This script reuses the validation-time evaluation logic from run_v1.py and
compares all saved checkpoint files in results_v1. It writes a JSON artifact
with per-checkpoint metrics and a compact Markdown table for quick inspection.
"""

import argparse
import importlib.util
import json
import sys
from itertools import combinations
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BertTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_v1"


def load_modules():
    """Load bias_absorption/run_v1.py and canonicalization/run_v1.py safely."""
    bias_dir = SCRIPT_DIR.parent / "bias_absorption"
    sys.path.insert(0, str(bias_dir))

    from run_v1 import TEST_SEEDS, get_reference_batch, load_and_serialize_all

    spec = importlib.util.spec_from_file_location(
        "canon_run_v1",
        SCRIPT_DIR / "run_v1.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["canon_run_v1"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module, TEST_SEEDS, get_reference_batch, load_and_serialize_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate canonicalization checkpoints on validation data.",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default="canon_epoch_*.pt",
        help="Glob for intermediate checkpoints inside results_v1.",
    )
    parser.add_argument(
        "--include-best",
        action="store_true",
        help="Also evaluate results_v1/canonicalization_module.pt.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Interpolation coefficient used for validation.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="Number of WikiText validation samples for the evaluation batch.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Sequence length for the evaluation batch.",
    )
    parser.add_argument(
        "--output-json",
        default="checkpoint_validation.json",
        help="Output JSON filename inside results_v1.",
    )
    parser.add_argument(
        "--output-md",
        default="checkpoint_validation.md",
        help="Output Markdown filename inside results_v1.",
    )
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    epoch = int(digits) if digits else 10**9
    return epoch, stem


def build_markdown(results: list[dict]) -> str:
    lines = [
        "# Checkpoint Validation",
        "",
        "| checkpoint | mean_ratio | mean_interp_ppl | mean_ensemble_ppl | mean_naive_ppl | best_pair | worst_pair |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in results:
        lines.append(
            "| {checkpoint} | {mean_ratio:.4f} | {mean_interp_ppl:.2f} | {mean_ensemble_ppl:.2f} | {mean_naive_ppl:.2f} | {best_pair} ({best_ratio:.4f}) | {worst_pair} ({worst_ratio:.4f}) |".format(
                checkpoint=row["checkpoint"],
                mean_ratio=row["mean_ratio"],
                mean_interp_ppl=row["mean_interp_ppl"],
                mean_ensemble_ppl=row["mean_ensemble_ppl"],
                mean_naive_ppl=row["mean_naive_ppl"],
                best_pair=row["best_pair"],
                best_ratio=row["best_ratio"],
                worst_pair=row["worst_pair"],
                worst_ratio=row["worst_ratio"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    canon_run, test_seeds, get_reference_batch, load_and_serialize_all = load_modules()
    config_module = canon_run

    checkpoint_paths = sorted(RESULTS_DIR.glob(args.checkpoint_glob), key=checkpoint_sort_key)
    if args.include_best:
        best_path = RESULTS_DIR / "canonicalization_module.pt"
        if best_path.exists() and best_path not in checkpoint_paths:
            checkpoint_paths.append(best_path)

    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched in {RESULTS_DIR}")

    print("Loading serialized models...")
    serialized, config = load_and_serialize_all()
    projection = config_module.load_frozen_projection().to(config_module.DEVICE)
    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")

    print("Preparing validation batch...")
    ref_batch = get_reference_batch(
        tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length,
    )

    print("Creating shell model...")
    shell_model = config_module.load_multibert_model(test_seeds[0]).eval().to(config_module.DEVICE)

    print("Pre-encoding held-out models...")
    absorbed = {}
    for seed in tqdm(test_seeds, desc="Encoding", unit="model"):
        with torch.no_grad():
            absorbed[seed] = config_module.absorb_bias_rows_only(
                projection,
                serialized[seed],
                config,
                config_module.DEVICE,
            ).cpu()

    val_pairs = list(combinations(test_seeds, 2))

    print("Computing naive interpolation baseline...")
    naive_results = {}
    for seed_i, seed_j in tqdm(val_pairs, desc="Naive", unit="pair"):
        pair_key = f"{seed_i}_{seed_j}"
        naive_results[pair_key] = config_module.evaluate_naive_interpolation(
            projection,
            serialized,
            seed_i,
            seed_j,
            args.alpha,
            config,
            ref_batch,
        )

    results = []
    for checkpoint_path in tqdm(checkpoint_paths, desc="Checkpoints", unit="ckpt"):
        canon_module = config_module.CanonicalizationModule(
            vocab_size=config.vocab_size,
            d_model=config.hidden_size,
            sinkhorn_iters=20,
            tau_init=0.5,
        )
        canon_module.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        canon_module = canon_module.to(config_module.DEVICE).eval()

        pair_results = {}
        ratios = []
        interp_ppls = []
        ensemble_ppls = []

        for seed_i, seed_j in tqdm(
            val_pairs,
            desc=f"  {checkpoint_path.name}",
            unit="pair",
            leave=False,
        ):
            pair_key = f"{seed_i}_{seed_j}"
            result = config_module.evaluate_pair_perplexity(
                canon_module,
                projection,
                absorbed,
                serialized,
                seed_i,
                seed_j,
                args.alpha,
                config,
                ref_batch,
                shell_model,
            )
            result["naive_ppl"] = naive_results[pair_key]
            pair_results[pair_key] = result
            ratios.append(result["ratio"])
            interp_ppls.append(result["interp_ppl"])
            ensemble_ppls.append(result["ensemble_ppl"])

        best_pair = min(pair_results.items(), key=lambda item: item[1]["ratio"])
        worst_pair = max(pair_results.items(), key=lambda item: item[1]["ratio"])

        summary = {
            "checkpoint": checkpoint_path.name,
            "checkpoint_path": str(checkpoint_path),
            "alpha": args.alpha,
            "num_samples": args.num_samples,
            "max_length": args.max_length,
            "mean_ratio": sum(ratios) / len(ratios),
            "mean_interp_ppl": sum(interp_ppls) / len(interp_ppls),
            "mean_ensemble_ppl": sum(ensemble_ppls) / len(ensemble_ppls),
            "mean_naive_ppl": sum(naive_results.values()) / len(naive_results),
            "best_pair": best_pair[0],
            "best_ratio": best_pair[1]["ratio"],
            "worst_pair": worst_pair[0],
            "worst_ratio": worst_pair[1]["ratio"],
            "pairs": pair_results,
        }
        results.append(summary)

        del canon_module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results.sort(key=lambda row: row["mean_ratio"])

    output_json = RESULTS_DIR / args.output_json
    output_md = RESULTS_DIR / args.output_md
    payload = {
        "alpha": args.alpha,
        "num_samples": args.num_samples,
        "max_length": args.max_length,
        "validation_pairs": [list(pair) for pair in val_pairs],
        "naive_results": naive_results,
        "checkpoints": results,
    }

    output_json.write_text(json.dumps(config_module.make_serializable(payload), indent=2))
    output_md.write_text(build_markdown(results))

    print(f"Saved JSON results to {output_json}")
    print(f"Saved Markdown summary to {output_md}")
    print("\nTop checkpoints by mean ratio:")
    for row in results[:5]:
        print(
            f"  {row['checkpoint']:<28} ratio={row['mean_ratio']:.4f} "
            f"interp={row['mean_interp_ppl']:.2f} ensemble={row['mean_ensemble_ppl']:.2f}"
        )


if __name__ == "__main__":
    main()