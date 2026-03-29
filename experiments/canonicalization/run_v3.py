"""Canonicalization Experiment V3: RotationSymmetry shared-reference baseline.

This experiment adapts the RotationSymmetry matching algorithm to MultiBERT as a
closed-form, no-training canonicalization baseline.

High-level procedure:
  1. Choose one train-side MultiBERT seed as a shared reference basis.
  2. Match every other seed into that reference basis using FFN permutations and
     attention-head orthogonal matching.
  3. Interpolate matched models directly in native BERT parameter space.
  4. Evaluate held-out MLM perplexity and interpolation curves against naive
     interpolation and endpoint ensembling baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertTokenizer

from quineformer.bias_absorption import extract_non_bert_params, get_multibert_snapshot, load_multibert_model
from quineformer.experiment_utils import (
    load_serialized_models,
    run_functional_mlm_logits,
    run_functional_mlm_loss,
    sample_masked_mlm_batch_from_token_ids,
)
from quineformer.rotation_symmetry import (
    canonicalize_model_to_reference,
    interpolate_state_dicts,
    model_state_distance,
    select_reference_seed,
    summarize_transform_metadata,
)
from quineformer.serialization import serialize


REPO_ROOT = Path(__file__).resolve().parents[2]
SERIALIZED_CACHE = REPO_ROOT / "data" / "multiberts" / "serialized"
RESULTS_DIR = Path(__file__).resolve().parent / "results_v3"
TRAIN_SEEDS = list(range(20))
TEST_SEEDS = list(range(20, 25))
MAX_EXP_INPUT = math.log(torch.finfo(torch.float64).max)


def safe_exp(value: float) -> float:
    if math.isnan(value):
        return float("nan")
    if value >= MAX_EXP_INPUT:
        return float("inf")
    return math.exp(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-seed", type=int, default=None)
    parser.add_argument("--reference-metric", choices=["embeddings", "full"], default="embeddings")
    parser.add_argument("--eval-pairs", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument("--curve-pairs", type=int, default=2)
    parser.add_argument("--curve-steps", type=int, default=11)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--use-ffn", action="store_true")
    parser.add_argument("--use-rescaling", action="store_true")
    parser.add_argument("--layer-indices", type=int, nargs="*", default=None)
    parser.add_argument("--cache-matched-serialized", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_wikitext_token_ids(tokenizer: BertTokenizer, split: str) -> list[int]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError("run_v3.py requires the `datasets` package to sample MLM batches.") from error

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(chunk for chunk in dataset["text"] if chunk.strip())
    return tokenizer.encode(text, add_special_tokens=False)


def build_eval_batches(
    token_ids: list[int],
    tokenizer: BertTokenizer,
    *,
    num_batches: int,
    num_samples: int,
    max_length: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    return [
        sample_masked_mlm_batch_from_token_ids(
            token_ids=token_ids,
            tokenizer=tokenizer,
            num_samples=num_samples,
            max_length=max_length,
            seed=seed + batch_idx,
        )
        for batch_idx in range(max(num_batches, 1))
    ]


def evaluate_pairwise_perplexities(
    shell_model: torch.nn.Module,
    *,
    endpoint_bert_params: dict[int, dict[str, torch.Tensor]],
    endpoint_head_params: dict[int, dict[str, torch.Tensor]],
    matched_bert_params: dict[int, dict[str, torch.Tensor]],
    matched_head_params: dict[int, dict[str, torch.Tensor]],
    batches: list[dict[str, torch.Tensor]],
    pairs: list[tuple[int, int]],
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    results = []
    for batch_idx, batch in enumerate(batches):
        for seed_i, seed_j in pairs:
            with torch.no_grad():
                logits_i = run_functional_mlm_logits(
                    shell_model,
                    endpoint_bert_params[seed_i],
                    endpoint_head_params[seed_i],
                    batch,
                    device,
                )
                logits_j = run_functional_mlm_logits(
                    shell_model,
                    endpoint_bert_params[seed_j],
                    endpoint_head_params[seed_j],
                    batch,
                    device,
                )
                ensemble_logits = 0.5 * (logits_i + logits_j)
                ensemble_loss = F.cross_entropy(
                    ensemble_logits.view(-1, ensemble_logits.shape[-1]),
                    batch["labels"].to(device).view(-1),
                    ignore_index=-100,
                )

                naive_interp_loss = run_functional_mlm_loss(
                    shell_model,
                    interpolate_state_dicts(endpoint_bert_params[seed_i], endpoint_bert_params[seed_j], 0.5),
                    interpolate_state_dicts(endpoint_head_params[seed_i], endpoint_head_params[seed_j], 0.5),
                    batch,
                    device,
                )
                matched_interp_loss = run_functional_mlm_loss(
                    shell_model,
                    interpolate_state_dicts(matched_bert_params[seed_i], matched_bert_params[seed_j], 0.5),
                    interpolate_state_dicts(matched_head_params[seed_i], matched_head_params[seed_j], 0.5),
                    batch,
                    device,
                )

            results.append(
                {
                    "source": f"eval_batch:{batch_idx}",
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "ensemble_ppl": safe_exp(float(ensemble_loss.item())),
                    "naive_interp_ppl": safe_exp(float(naive_interp_loss.item())),
                    "matched_interp_ppl": safe_exp(float(matched_interp_loss.item())),
                    "matched_over_naive": safe_exp(float(matched_interp_loss.item() - naive_interp_loss.item())),
                    "matched_over_ensemble": safe_exp(float(matched_interp_loss.item() - ensemble_loss.item())),
                }
            )
    return results


def evaluate_interpolation_curves(
    shell_model: torch.nn.Module,
    *,
    endpoint_bert_params: dict[int, dict[str, torch.Tensor]],
    endpoint_head_params: dict[int, dict[str, torch.Tensor]],
    matched_bert_params: dict[int, dict[str, torch.Tensor]],
    matched_head_params: dict[int, dict[str, torch.Tensor]],
    batch: dict[str, torch.Tensor],
    pairs: list[tuple[int, int]],
    curve_steps: int,
    device: torch.device,
) -> list[dict[str, float | int]]:
    alphas = [step / max(curve_steps - 1, 1) for step in range(curve_steps)]
    curves = []
    for seed_i, seed_j in pairs:
        for alpha in alphas:
            with torch.no_grad():
                naive_loss = run_functional_mlm_loss(
                    shell_model,
                    interpolate_state_dicts(endpoint_bert_params[seed_i], endpoint_bert_params[seed_j], alpha),
                    interpolate_state_dicts(endpoint_head_params[seed_i], endpoint_head_params[seed_j], alpha),
                    batch,
                    device,
                )
                matched_loss = run_functional_mlm_loss(
                    shell_model,
                    interpolate_state_dicts(matched_bert_params[seed_i], matched_bert_params[seed_j], alpha),
                    interpolate_state_dicts(matched_head_params[seed_i], matched_head_params[seed_j], alpha),
                    batch,
                    device,
                )
            curves.append(
                {
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "alpha": alpha,
                    "naive_interp_ppl": safe_exp(float(naive_loss.item())),
                    "matched_interp_ppl": safe_exp(float(matched_loss.item())),
                }
            )
    return curves


def main() -> None:
    args = parse_args()
    if not args.use_attention and not args.use_ffn:
        args.use_attention = True
        args.use_ffn = True
    if args.smoke_test:
        args.eval_pairs = 1
        args.eval_batches = 1
        args.curve_pairs = 1
        args.curve_steps = 3
        args.num_samples = 2
        args.max_length = 32

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seeds = TRAIN_SEEDS[:4] if args.smoke_test else TRAIN_SEEDS
    test_seeds = TEST_SEEDS[:2] if args.smoke_test else TEST_SEEDS
    all_seeds = sorted(set(train_seeds + test_seeds + ([args.reference_seed] if args.reference_seed is not None else [])))

    serialized, config = load_serialized_models(all_seeds, SERIALIZED_CACHE)
    if args.reference_seed is None:
        reference_seed = select_reference_seed(
            serialized,
            train_seeds,
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            metric=args.reference_metric,
        )
    else:
        reference_seed = args.reference_seed

    tokenizer = BertTokenizer.from_pretrained(get_multibert_snapshot(reference_seed), local_files_only=True)
    token_ids = load_wikitext_token_ids(tokenizer, split="validation")
    eval_batches = build_eval_batches(
        token_ids,
        tokenizer,
        num_batches=args.eval_batches,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.cache_matched_serialized:
        (RESULTS_DIR / "matched_serialized").mkdir(parents=True, exist_ok=True)

    reference_model = load_multibert_model(reference_seed).eval()
    shell_model = load_multibert_model(reference_seed).eval().to(device)

    endpoint_bert_params: dict[int, dict[str, torch.Tensor]] = {}
    endpoint_head_params: dict[int, dict[str, torch.Tensor]] = {}
    matched_bert_params: dict[int, dict[str, torch.Tensor]] = {}
    matched_head_params: dict[int, dict[str, torch.Tensor]] = {}
    distance_rows: list[dict[str, float | int]] = []

    reference_state = reference_model.bert.state_dict()
    for seed in all_seeds:
        model = load_multibert_model(seed).eval()
        endpoint_bert_params[seed] = {name: value.detach().clone() for name, value in model.bert.state_dict().items()}
        endpoint_head_params[seed] = extract_non_bert_params(model)

        if seed == reference_seed:
            matched_model = model
            transform_summary = {
                "matched_layers": 0,
                "mean_qk_orth_error": 0.0,
                "mean_vo_orth_error": 0.0,
                "mean_qk_scale": 1.0,
                "mean_vo_scale": 1.0,
                "mean_ffn_identity_deviation": 0.0,
            }
        else:
            matched_model, transform_metadata = canonicalize_model_to_reference(
                model,
                reference_model,
                use_attention=args.use_attention,
                use_ffn=args.use_ffn,
                use_rescaling=args.use_rescaling,
                layer_indices=args.layer_indices,
            )
            transform_summary = summarize_transform_metadata(transform_metadata)

        matched_bert_params[seed] = {name: value.detach().clone() for name, value in matched_model.bert.state_dict().items()}
        matched_head_params[seed] = extract_non_bert_params(matched_model)

        before_distance = model_state_distance(endpoint_bert_params[seed], reference_state)
        after_distance = model_state_distance(matched_bert_params[seed], reference_state)
        distance_rows.append(
            {
                "seed": seed,
                "reference_seed": reference_seed,
                "distance_before": before_distance,
                "distance_after": after_distance,
                **transform_summary,
            }
        )

        if args.cache_matched_serialized:
            torch.save(
                serialize(matched_model),
                RESULTS_DIR / "matched_serialized" / f"seed_{seed}.pt",
            )

    held_out_pairs = list(combinations(test_seeds, 2))[: args.eval_pairs]
    perplexity_rows = evaluate_pairwise_perplexities(
        shell_model,
        endpoint_bert_params=endpoint_bert_params,
        endpoint_head_params=endpoint_head_params,
        matched_bert_params=matched_bert_params,
        matched_head_params=matched_head_params,
        batches=eval_batches,
        pairs=held_out_pairs,
        device=device,
    )
    curve_rows = evaluate_interpolation_curves(
        shell_model,
        endpoint_bert_params=endpoint_bert_params,
        endpoint_head_params=endpoint_head_params,
        matched_bert_params=matched_bert_params,
        matched_head_params=matched_head_params,
        batch=eval_batches[0],
        pairs=held_out_pairs[: args.curve_pairs],
        curve_steps=args.curve_steps,
        device=device,
    )

    with open(RESULTS_DIR / "reference_choice.json", "w", encoding="ascii") as handle:
        json.dump(
            {
                "reference_seed": reference_seed,
                "reference_metric": args.reference_metric,
                "train_seeds": train_seeds,
                "test_seeds": test_seeds,
                "use_attention": args.use_attention,
                "use_ffn": args.use_ffn,
                "use_rescaling": args.use_rescaling,
                "layer_indices": args.layer_indices,
            },
            handle,
            indent=2,
        )
    with open(RESULTS_DIR / "distance_summary.json", "w", encoding="ascii") as handle:
        json.dump(distance_rows, handle, indent=2)
    with open(RESULTS_DIR / "perplexity_comparison.json", "w", encoding="ascii") as handle:
        json.dump(perplexity_rows, handle, indent=2)
    with open(RESULTS_DIR / "interpolation_curves.json", "w", encoding="ascii") as handle:
        json.dump(curve_rows, handle, indent=2)


if __name__ == "__main__":
    main()