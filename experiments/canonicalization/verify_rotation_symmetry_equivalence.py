"""Verify equivalence between this repo's RotationSymmetry adaptation and the upstream library.

This harness compares both implementations at three levels:
  1. Primitive tensor matching for FFN, QK, and VO blocks.
  2. Full matched-model state dicts under a fixed anchor seed.
  3. Functional MLM losses for matched endpoints and their midpoint interpolation.

The reference seed is fixed explicitly for the experiment. Anchor-selection logic is
intentionally out of scope because this repo adapts the upstream merger to a shared-
reference MultiBERT setting rather than reusing their full model-merging framework.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import math
import random
from pathlib import Path
from types import ModuleType

import torch
from transformers import BertTokenizer

from quineformer.bias_absorption import extract_non_bert_params, get_multibert_snapshot, load_multibert_model
from quineformer.experiment_utils import run_functional_mlm_loss, sample_masked_mlm_batch_from_token_ids
from quineformer.rotation_symmetry import (
    canonicalize_model_to_reference,
    get_layer_attention_tensors,
    get_layer_ffn_tensors,
    interpolate_state_dicts,
    match_attention_qk_tensors,
    match_attention_vo_tensors,
    match_ffn_tensors,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results_v3_equivalence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-repo", type=Path, default=Path("/tmp/RotationSymmetry"))
    parser.add_argument("--upstream-source", choices=["auto", "src", "vit"], default="auto")
    parser.add_argument("--anchor-seed", type=int, default=3)
    parser.add_argument("--local-seeds", type=int, nargs="*", default=[20, 21])
    parser.add_argument("--layer-indices", type=int, nargs="*", default=[0, 5, 11])
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--use-ffn", action="store_true")
    parser.add_argument("--use-rescaling", action="store_true")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_upstream_module(repo_path: Path, source: str, use_rescaling: bool) -> tuple[ModuleType, str]:
    src_path = repo_path / "src" / "model_merge" / "pi_utils" / "utils.py"
    vit_path = repo_path / "vit-fusion" / "otfusion" / "pi_utils.py"

    if source == "src":
        return load_module(src_path, "rotation_symmetry_upstream_src"), "src"
    if source == "vit":
        return load_module(vit_path, "rotation_symmetry_upstream_vit"), "vit"
    if use_rescaling:
        return load_module(vit_path, "rotation_symmetry_upstream_vit"), "vit"
    return load_module(src_path, "rotation_symmetry_upstream_src"), "src"


def load_wikitext_token_ids(tokenizer: BertTokenizer, split: str) -> list[int]:
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "verify_rotation_symmetry_equivalence.py requires the `datasets` package."
        ) from error

    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    text = "\n".join(chunk for chunk in dataset["text"] if chunk.strip())
    return tokenizer.encode(text, add_special_tokens=False)


def sample_batch(
    token_ids: list[int],
    tokenizer: BertTokenizer,
    *,
    num_samples: int,
    max_length: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    return sample_masked_mlm_batch_from_token_ids(
        token_ids=token_ids,
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=max_length,
        seed=seed,
    )


def _upstream_kwargs(module: ModuleType, function_name: str, config, use_rescaling: bool) -> dict[str, object]:
    fn = getattr(module, function_name)
    signature = inspect.signature(fn)
    kwargs: dict[str, object] = {
        "num_attention_heads": config.num_attention_heads,
        "attention_head_size": config.hidden_size // config.num_attention_heads,
    }
    if "use_scaling" in signature.parameters:
        kwargs["use_scaling"] = use_rescaling
    return kwargs


def call_upstream_attention_qk(
    module: ModuleType,
    local_param: dict[str, torch.Tensor],
    anchor_param: dict[str, torch.Tensor],
    *,
    config,
    use_rescaling: bool,
) -> dict[str, torch.Tensor]:
    return getattr(module, "Chunk_SVD_QK")(
        local_param=local_param,
        anchor_param=anchor_param,
        **_upstream_kwargs(module, "Chunk_SVD_QK", config, use_rescaling),
    )


def call_upstream_attention_vo(
    module: ModuleType,
    local_param: dict[str, torch.Tensor],
    anchor_param: dict[str, torch.Tensor],
    *,
    config,
    use_rescaling: bool,
) -> dict[str, torch.Tensor]:
    return getattr(module, "Chunk_SVD_VO")(
        local_param=local_param,
        anchor_param=anchor_param,
        **_upstream_kwargs(module, "Chunk_SVD_VO", config, use_rescaling),
    )


def call_upstream_ffn(
    module: ModuleType,
    local_param: dict[str, torch.Tensor],
    anchor_param: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return getattr(module, "Permute_IO")(local_param=local_param, anchor_param=anchor_param)


def validate_upstream_module(
    module: ModuleType,
    *,
    config,
    layer,
    anchor_layer,
    use_rescaling: bool,
) -> None:
    local_attention = get_layer_attention_tensors(layer)
    anchor_attention = get_layer_attention_tensors(anchor_layer)
    call_upstream_attention_qk(
        module,
        local_attention,
        anchor_attention,
        config=config,
        use_rescaling=use_rescaling,
    )
    call_upstream_attention_vo(
        module,
        local_attention,
        anchor_attention,
        config=config,
        use_rescaling=use_rescaling,
    )
    call_upstream_ffn(
        module,
        get_layer_ffn_tensors(layer),
        get_layer_ffn_tensors(anchor_layer),
    )


def compare_tensor_dicts(
    ours: dict[str, torch.Tensor],
    upstream: dict[str, torch.Tensor],
    *,
    expected_keys: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in expected_keys:
        if key not in ours or key not in upstream:
            continue
        diff = (ours[key].detach().float() - upstream[key].detach().float()).abs()
        metrics[f"{key}.max_abs_diff"] = float(diff.max().item())
        metrics[f"{key}.mean_abs_diff"] = float(diff.mean().item())
    return metrics


def apply_upstream_matching(
    model,
    anchor_model,
    upstream_module: ModuleType,
    *,
    use_attention: bool,
    use_ffn: bool,
    use_rescaling: bool,
    layer_indices: list[int],
):
    matched_model = copy.deepcopy(model).eval()
    config = matched_model.config

    for layer_idx in layer_indices:
        local_layer = matched_model.bert.encoder.layer[layer_idx]
        anchor_layer = anchor_model.bert.encoder.layer[layer_idx]

        if use_attention:
            local_attention = get_layer_attention_tensors(local_layer)
            anchor_attention = get_layer_attention_tensors(anchor_layer)
            matched_qk = call_upstream_attention_qk(
                upstream_module,
                local_attention,
                anchor_attention,
                config=config,
                use_rescaling=use_rescaling,
            )
            matched_vo = call_upstream_attention_vo(
                upstream_module,
                local_attention,
                anchor_attention,
                config=config,
                use_rescaling=use_rescaling,
            )
            local_layer.attention.self.query.weight.data.copy_(matched_qk["W_Q"])
            local_layer.attention.self.query.bias.data.copy_(matched_qk["B_Q"])
            local_layer.attention.self.key.weight.data.copy_(matched_qk["W_K"])
            local_layer.attention.self.key.bias.data.copy_(matched_qk["B_K"])
            local_layer.attention.self.value.weight.data.copy_(matched_vo["W_V"])
            local_layer.attention.self.value.bias.data.copy_(matched_vo["B_V"])
            local_layer.attention.output.dense.weight.data.copy_(matched_vo["W_O"])
            if "B_O" in matched_vo:
                local_layer.attention.output.dense.bias.data.copy_(matched_vo["B_O"])

        if use_ffn:
            local_ffn = get_layer_ffn_tensors(local_layer)
            anchor_ffn = get_layer_ffn_tensors(anchor_layer)
            matched_ffn = call_upstream_ffn(upstream_module, local_ffn, anchor_ffn)
            local_layer.intermediate.dense.weight.data.copy_(matched_ffn["W_I"])
            local_layer.intermediate.dense.bias.data.copy_(matched_ffn["B_I"])
            local_layer.output.dense.weight.data.copy_(matched_ffn["W_O_FFN"])
            if "B_O_FFN" in matched_ffn:
                local_layer.output.dense.bias.data.copy_(matched_ffn["B_O_FFN"])

    return matched_model


def aggregate_state_dict_diff(
    ours_state: dict[str, torch.Tensor],
    upstream_state: dict[str, torch.Tensor],
) -> dict[str, float]:
    max_abs = 0.0
    mean_abs_sum = 0.0
    total_values = 0
    compared_keys = 0
    differing_keys = 0

    for key in ours_state.keys() & upstream_state.keys():
        ours_value = ours_state[key]
        upstream_value = upstream_state[key]
        if not (torch.is_tensor(ours_value) and torch.is_tensor(upstream_value)):
            continue
        if not (torch.is_floating_point(ours_value) and torch.is_floating_point(upstream_value)):
            continue
        compared_keys += 1
        diff = (ours_value.detach().float() - upstream_value.detach().float()).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        mean_abs_sum += float(diff.sum().item())
        total_values += diff.numel()
        if float(diff.max().item()) > 0.0:
            differing_keys += 1

    return {
        "state_dict_max_abs_diff": max_abs,
        "state_dict_mean_abs_diff": mean_abs_sum / max(total_values, 1),
        "state_dict_compared_keys": compared_keys,
        "state_dict_differing_keys": differing_keys,
    }


def main() -> None:
    args = parse_args()
    if not args.use_attention and not args.use_ffn:
        args.use_attention = True
        args.use_ffn = True
    if args.smoke_test:
        args.local_seeds = args.local_seeds[:2]
        args.layer_indices = args.layer_indices[:1]
        args.num_samples = 2
        args.max_length = 32

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    requested_upstream_source = args.upstream_source
    upstream_module, upstream_source = load_upstream_module(
        args.upstream_repo,
        args.upstream_source,
        args.use_rescaling,
    )

    anchor_model = load_multibert_model(args.anchor_seed).eval()
    shell_model = load_multibert_model(args.anchor_seed).eval().to(device)
    config = anchor_model.config
    probe_model = load_multibert_model(args.local_seeds[0]).eval()

    if requested_upstream_source == "auto":
        try:
            validate_upstream_module(
                upstream_module,
                config=config,
                layer=probe_model.bert.encoder.layer[args.layer_indices[0]],
                anchor_layer=anchor_model.bert.encoder.layer[args.layer_indices[0]],
                use_rescaling=args.use_rescaling,
            )
        except Exception:
            upstream_module, upstream_source = load_upstream_module(
                args.upstream_repo,
                "vit",
                args.use_rescaling,
            )

    tokenizer = BertTokenizer.from_pretrained(get_multibert_snapshot(args.anchor_seed), local_files_only=True)
    token_ids = load_wikitext_token_ids(tokenizer, split="validation")
    batch = sample_batch(
        token_ids,
        tokenizer,
        num_samples=args.num_samples,
        max_length=args.max_length,
        seed=args.seed,
    )

    primitive_rows: list[dict[str, float | int | str]] = []
    model_rows: list[dict[str, float | int | str]] = []

    our_matched_models = {}
    upstream_matched_models = {}
    our_heads = {}
    upstream_heads = {}

    for local_seed in args.local_seeds:
        local_model = load_multibert_model(local_seed).eval()
        our_model, _ = canonicalize_model_to_reference(
            local_model,
            anchor_model,
            use_attention=args.use_attention,
            use_ffn=args.use_ffn,
            use_rescaling=args.use_rescaling,
            layer_indices=args.layer_indices,
        )
        upstream_model = apply_upstream_matching(
            local_model,
            anchor_model,
            upstream_module,
            use_attention=args.use_attention,
            use_ffn=args.use_ffn,
            use_rescaling=args.use_rescaling,
            layer_indices=args.layer_indices,
        )

        our_matched_models[local_seed] = our_model
        upstream_matched_models[local_seed] = upstream_model
        our_heads[local_seed] = extract_non_bert_params(our_model)
        upstream_heads[local_seed] = extract_non_bert_params(upstream_model)

        for layer_idx in args.layer_indices:
            local_layer = local_model.bert.encoder.layer[layer_idx]
            anchor_layer = anchor_model.bert.encoder.layer[layer_idx]

            if args.use_attention:
                local_attention = get_layer_attention_tensors(local_layer)
                anchor_attention = get_layer_attention_tensors(anchor_layer)
                our_qk, _ = match_attention_qk_tensors(
                    local_attention,
                    anchor_attention,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_size=config.hidden_size // config.num_attention_heads,
                    use_rescaling=args.use_rescaling,
                )
                upstream_qk = call_upstream_attention_qk(
                    upstream_module,
                    local_attention,
                    anchor_attention,
                    config=config,
                    use_rescaling=args.use_rescaling,
                )
                primitive_rows.append(
                    {
                        "seed": local_seed,
                        "layer_idx": layer_idx,
                        "block": "attention_qk",
                        **compare_tensor_dicts(our_qk, upstream_qk, expected_keys=["W_Q", "B_Q", "W_K", "B_K"]),
                    }
                )

                our_vo, _ = match_attention_vo_tensors(
                    local_attention,
                    anchor_attention,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_size=config.hidden_size // config.num_attention_heads,
                    use_rescaling=args.use_rescaling,
                )
                upstream_vo = call_upstream_attention_vo(
                    upstream_module,
                    local_attention,
                    anchor_attention,
                    config=config,
                    use_rescaling=args.use_rescaling,
                )
                primitive_rows.append(
                    {
                        "seed": local_seed,
                        "layer_idx": layer_idx,
                        "block": "attention_vo",
                        **compare_tensor_dicts(our_vo, upstream_vo, expected_keys=["W_V", "B_V", "W_O", "B_O"]),
                    }
                )

            if args.use_ffn:
                local_ffn = get_layer_ffn_tensors(local_layer)
                anchor_ffn = get_layer_ffn_tensors(anchor_layer)
                our_ffn, _ = match_ffn_tensors(local_ffn, anchor_ffn)
                upstream_ffn = call_upstream_ffn(upstream_module, local_ffn, anchor_ffn)
                primitive_rows.append(
                    {
                        "seed": local_seed,
                        "layer_idx": layer_idx,
                        "block": "ffn",
                        **compare_tensor_dicts(our_ffn, upstream_ffn, expected_keys=["W_I", "B_I", "W_O_FFN", "B_O_FFN"]),
                    }
                )

        state_metrics = aggregate_state_dict_diff(
            our_model.bert.state_dict(),
            upstream_model.bert.state_dict(),
        )

        our_loss = run_functional_mlm_loss(
            shell_model,
            {name: value.detach().clone() for name, value in our_model.bert.state_dict().items()},
            our_heads[local_seed],
            batch,
            device,
        )
        upstream_loss = run_functional_mlm_loss(
            shell_model,
            {name: value.detach().clone() for name, value in upstream_model.bert.state_dict().items()},
            upstream_heads[local_seed],
            batch,
            device,
        )
        model_rows.append(
            {
                "seed": local_seed,
                **state_metrics,
                "matched_endpoint_loss_abs_diff": float(abs(our_loss.item() - upstream_loss.item())),
            }
        )

    interpolation_rows: list[dict[str, float | int]] = []
    if len(args.local_seeds) >= 2:
        seed_i, seed_j = args.local_seeds[:2]
        our_interp_loss = run_functional_mlm_loss(
            shell_model,
            interpolate_state_dicts(
                {name: value.detach().clone() for name, value in our_matched_models[seed_i].bert.state_dict().items()},
                {name: value.detach().clone() for name, value in our_matched_models[seed_j].bert.state_dict().items()},
                0.5,
            ),
            interpolate_state_dicts(our_heads[seed_i], our_heads[seed_j], 0.5),
            batch,
            device,
        )
        upstream_interp_loss = run_functional_mlm_loss(
            shell_model,
            interpolate_state_dicts(
                {name: value.detach().clone() for name, value in upstream_matched_models[seed_i].bert.state_dict().items()},
                {name: value.detach().clone() for name, value in upstream_matched_models[seed_j].bert.state_dict().items()},
                0.5,
            ),
            interpolate_state_dicts(upstream_heads[seed_i], upstream_heads[seed_j], 0.5),
            batch,
            device,
        )
        interpolation_rows.append(
            {
                "seed_i": seed_i,
                "seed_j": seed_j,
                "midpoint_loss_abs_diff": float(abs(our_interp_loss.item() - upstream_interp_loss.item())),
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "upstream_repo": str(args.upstream_repo),
        "requested_upstream_source": requested_upstream_source,
        "upstream_source": upstream_source,
        "anchor_seed": args.anchor_seed,
        "local_seeds": args.local_seeds,
        "layer_indices": args.layer_indices,
        "use_attention": args.use_attention,
        "use_ffn": args.use_ffn,
        "use_rescaling": args.use_rescaling,
        "primitive_rows": primitive_rows,
        "model_rows": model_rows,
        "interpolation_rows": interpolation_rows,
    }
    with open(RESULTS_DIR / "equivalence_report.json", "w", encoding="ascii") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()