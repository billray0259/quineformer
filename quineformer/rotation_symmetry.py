from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from transformers import BertForMaskedLM


def _resolve_head_layout(
    tensor_rows: int,
    num_attention_heads: int | None,
    attention_head_size: int | None,
) -> tuple[int, int]:
    if num_attention_heads is None and attention_head_size is None:
        raise ValueError("Provide num_attention_heads or attention_head_size")
    if attention_head_size is None:
        assert num_attention_heads is not None
        if tensor_rows % num_attention_heads != 0:
            raise ValueError(
                f"tensor_rows={tensor_rows} is not divisible by num_attention_heads={num_attention_heads}"
            )
        attention_head_size = tensor_rows // num_attention_heads
    if num_attention_heads is None:
        if tensor_rows % attention_head_size != 0:
            raise ValueError(
                f"tensor_rows={tensor_rows} is not divisible by attention_head_size={attention_head_size}"
            )
        num_attention_heads = tensor_rows // attention_head_size
    if num_attention_heads * attention_head_size != tensor_rows:
        raise ValueError(
            "Inconsistent attention layout: "
            f"num_attention_heads={num_attention_heads}, "
            f"attention_head_size={attention_head_size}, tensor_rows={tensor_rows}"
        )
    return num_attention_heads, attention_head_size


def _orthogonal_from_summary(summary: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(summary)
    return u @ vh


def _best_qk_rescaling(
    source_q: torch.Tensor,
    target_q: torch.Tensor,
    source_q_bias: torch.Tensor,
    target_q_bias: torch.Tensor,
    source_k: torch.Tensor,
    target_k: torch.Tensor,
    source_k_bias: torch.Tensor,
    target_k_bias: torch.Tensor,
) -> float:
    coeffs = np.array(
        [
            float(torch.sum(source_q * source_q).item() + torch.sum(source_q_bias * source_q_bias).item()),
            float(-torch.sum(source_q * target_q).item() - torch.sum(source_q_bias * target_q_bias).item()),
            0.0,
            float(torch.sum(source_k * target_k).item() + torch.sum(source_k_bias * target_k_bias).item()),
            float(-torch.sum(source_k * source_k).item() - torch.sum(source_k_bias * source_k_bias).item()),
        ],
        dtype=np.float64,
    )
    roots = np.roots(coeffs)
    candidates = [float(np.real(root)) for root in roots if np.isreal(root) and abs(np.real(root)) > 1e-8]
    if not candidates:
        return 1.0

    def objective(scale: float) -> float:
        return float(
            torch.sum((scale * source_q - target_q) ** 2).item()
            + torch.sum((scale * source_q_bias - target_q_bias) ** 2).item()
            + torch.sum((source_k / scale - target_k) ** 2).item()
            + torch.sum((source_k_bias / scale - target_k_bias) ** 2).item()
        )

    return min(candidates, key=objective)


def _best_vo_rescaling(
    source_v: torch.Tensor,
    target_v: torch.Tensor,
    source_v_bias: torch.Tensor,
    target_v_bias: torch.Tensor,
    source_o: torch.Tensor,
    target_o: torch.Tensor,
) -> float:
    coeffs = np.array(
        [
            float(torch.sum(source_v * source_v).item() + torch.sum(source_v_bias * source_v_bias).item()),
            float(-torch.sum(source_v * target_v).item() - torch.sum(source_v_bias * target_v_bias).item()),
            0.0,
            float(torch.sum(source_o * target_o).item()),
            float(-torch.sum(source_o * source_o).item()),
        ],
        dtype=np.float64,
    )
    roots = np.roots(coeffs)
    candidates = [float(np.real(root)) for root in roots if np.isreal(root) and abs(np.real(root)) > 1e-8]
    if not candidates:
        return 1.0

    def objective(scale: float) -> float:
        return float(
            torch.sum((scale * source_v - target_v) ** 2).item()
            + torch.sum((scale * source_v_bias - target_v_bias) ** 2).item()
            + torch.sum((source_o / scale - target_o) ** 2).item()
        )

    return min(candidates, key=objective)


def match_ffn_tensors(
    local_param: dict[str, torch.Tensor],
    anchor_param: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Match one FFN module to an anchor by solving the Hungarian assignment."""
    w_i_local = local_param["W_I"]
    w_o_local = local_param["W_O_FFN"]
    b_i_local = local_param["B_I"]
    b_o_local = local_param["B_O_FFN"]

    w_i_anchor = anchor_param["W_I"]
    w_o_anchor = anchor_param["W_O_FFN"]
    b_i_anchor = anchor_param["B_I"]

    cost = (
        w_i_local @ w_i_anchor.transpose(0, 1)
        + torch.outer(b_i_local, b_i_anchor)
        + w_o_local.transpose(0, 1) @ w_o_anchor
    )
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy(), maximize=True)
    permutation = torch.zeros_like(cost)
    permutation[row_ind, col_ind] = 1.0

    matched = {
        "W_I": permutation.transpose(0, 1) @ w_i_local,
        "B_I": b_i_local @ permutation,
        "W_O_FFN": w_o_local @ permutation,
        "B_O_FFN": b_o_local.detach().clone(),
    }
    metadata = {
        "permutation": permutation.detach().clone(),
        "assignment": torch.as_tensor(col_ind, dtype=torch.long),
    }
    return matched, metadata


def match_attention_qk_tensors(
    local_param: dict[str, torch.Tensor],
    anchor_param: dict[str, torch.Tensor],
    *,
    num_attention_heads: int | None = None,
    attention_head_size: int | None = None,
    use_rescaling: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Match Q/K attention parameters to an anchor with per-head orthogonal maps."""
    w_q_local = local_param["W_Q"]
    w_k_local = local_param["W_K"]
    b_q_local = local_param["B_Q"]
    b_k_local = local_param["B_K"]

    w_q_anchor = anchor_param["W_Q"]
    w_k_anchor = anchor_param["W_K"]
    b_q_anchor = anchor_param["B_Q"]
    b_k_anchor = anchor_param["B_K"]

    num_attention_heads, attention_head_size = _resolve_head_layout(
        w_q_local.shape[0],
        num_attention_heads,
        attention_head_size,
    )

    matched = {
        "W_Q": torch.zeros_like(w_q_local),
        "W_K": torch.zeros_like(w_k_local),
        "B_Q": torch.zeros_like(b_q_local),
        "B_K": torch.zeros_like(b_k_local),
    }
    rotations: list[torch.Tensor] = []
    scales: list[float] = []

    for head_idx in range(num_attention_heads):
        start = head_idx * attention_head_size
        end = (head_idx + 1) * attention_head_size

        q_local = w_q_local[start:end]
        k_local = w_k_local[start:end]
        q_anchor = w_q_anchor[start:end]
        k_anchor = w_k_anchor[start:end]
        bq_local = b_q_local[start:end]
        bk_local = b_k_local[start:end]
        bq_anchor = b_q_anchor[start:end]
        bk_anchor = b_k_anchor[start:end]

        summary = (
            q_local @ q_anchor.transpose(0, 1)
            + k_local @ k_anchor.transpose(0, 1)
            + torch.outer(bq_local, bq_anchor)
            + torch.outer(bk_local, bk_anchor)
        )
        rotation = _orthogonal_from_summary(summary)
        q_matched = rotation.transpose(0, 1) @ q_local
        k_matched = rotation.transpose(0, 1) @ k_local
        bq_matched = bq_local @ rotation
        bk_matched = bk_local @ rotation
        scale = 1.0
        if use_rescaling:
            scale = _best_qk_rescaling(
                q_matched,
                q_anchor,
                bq_matched,
                bq_anchor,
                k_matched,
                k_anchor,
                bk_matched,
                bk_anchor,
            )
            q_matched = q_matched * scale
            bq_matched = bq_matched * scale
            k_matched = k_matched / scale
            bk_matched = bk_matched / scale

        matched["W_Q"][start:end] = q_matched
        matched["W_K"][start:end] = k_matched
        matched["B_Q"][start:end] = bq_matched
        matched["B_K"][start:end] = bk_matched
        rotations.append(rotation.detach().clone())
        scales.append(float(scale))

    return matched, {"rotations": rotations, "scales": scales}


def match_attention_vo_tensors(
    local_param: dict[str, torch.Tensor],
    anchor_param: dict[str, torch.Tensor],
    *,
    num_attention_heads: int | None = None,
    attention_head_size: int | None = None,
    use_rescaling: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Match V/O attention parameters to an anchor with per-head orthogonal maps."""
    w_v_local = local_param["W_V"]
    w_o_local = local_param["W_O"]
    b_v_local = local_param["B_V"]
    b_o_local = local_param["B_O"]

    w_v_anchor = anchor_param["W_V"]
    w_o_anchor = anchor_param["W_O"]
    b_v_anchor = anchor_param["B_V"]

    num_attention_heads, attention_head_size = _resolve_head_layout(
        w_v_local.shape[0],
        num_attention_heads,
        attention_head_size,
    )

    matched = {
        "W_V": torch.zeros_like(w_v_local),
        "W_O": torch.zeros_like(w_o_local),
        "B_V": torch.zeros_like(b_v_local),
        "B_O": b_o_local.detach().clone(),
    }
    rotations: list[torch.Tensor] = []
    scales: list[float] = []

    for head_idx in range(num_attention_heads):
        start = head_idx * attention_head_size
        end = (head_idx + 1) * attention_head_size

        v_local = w_v_local[start:end]
        v_anchor = w_v_anchor[start:end]
        o_local = w_o_local[:, start:end]
        o_anchor = w_o_anchor[:, start:end]
        bv_local = b_v_local[start:end]
        bv_anchor = b_v_anchor[start:end]

        summary = (
            v_local @ v_anchor.transpose(0, 1)
            + o_local.transpose(0, 1) @ o_anchor
            + torch.outer(bv_local, bv_anchor)
        )
        rotation = _orthogonal_from_summary(summary)
        v_matched = rotation.transpose(0, 1) @ v_local
        o_matched = o_local @ rotation
        bv_matched = bv_local @ rotation
        scale = 1.0
        if use_rescaling:
            scale = _best_vo_rescaling(v_matched, v_anchor, bv_matched, bv_anchor, o_matched, o_anchor)
            v_matched = v_matched * scale
            bv_matched = bv_matched * scale
            o_matched = o_matched / scale

        matched["W_V"][start:end] = v_matched
        matched["W_O"][:, start:end] = o_matched
        matched["B_V"][start:end] = bv_matched
        rotations.append(rotation.detach().clone())
        scales.append(float(scale))

    return matched, {"rotations": rotations, "scales": scales}


def get_layer_attention_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    attention = layer.attention.self
    output = layer.attention.output.dense
    return {
        "W_Q": attention.query.weight.detach().clone(),
        "B_Q": attention.query.bias.detach().clone(),
        "W_K": attention.key.weight.detach().clone(),
        "B_K": attention.key.bias.detach().clone(),
        "W_V": attention.value.weight.detach().clone(),
        "B_V": attention.value.bias.detach().clone(),
        "W_O": output.weight.detach().clone(),
        "B_O": output.bias.detach().clone(),
    }


def get_layer_ffn_tensors(layer: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        "W_I": layer.intermediate.dense.weight.detach().clone(),
        "B_I": layer.intermediate.dense.bias.detach().clone(),
        "W_O_FFN": layer.output.dense.weight.detach().clone(),
        "B_O_FFN": layer.output.dense.bias.detach().clone(),
    }


def apply_layer_attention_tensors(layer: torch.nn.Module, matched: dict[str, torch.Tensor]) -> None:
    layer.attention.self.query.weight.data.copy_(matched["W_Q"])
    layer.attention.self.query.bias.data.copy_(matched["B_Q"])
    layer.attention.self.key.weight.data.copy_(matched["W_K"])
    layer.attention.self.key.bias.data.copy_(matched["B_K"])
    layer.attention.self.value.weight.data.copy_(matched["W_V"])
    layer.attention.self.value.bias.data.copy_(matched["B_V"])
    layer.attention.output.dense.weight.data.copy_(matched["W_O"])
    layer.attention.output.dense.bias.data.copy_(matched["B_O"])


def apply_layer_ffn_tensors(layer: torch.nn.Module, matched: dict[str, torch.Tensor]) -> None:
    layer.intermediate.dense.weight.data.copy_(matched["W_I"])
    layer.intermediate.dense.bias.data.copy_(matched["B_I"])
    layer.output.dense.weight.data.copy_(matched["W_O_FFN"])
    layer.output.dense.bias.data.copy_(matched["B_O_FFN"])


def model_state_distance(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> float:
    """Compute the Frobenius distance between matching floating-point tensors."""
    distance_sq = 0.0
    for key in state_a.keys() & state_b.keys():
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if not (torch.is_tensor(tensor_a) and torch.is_tensor(tensor_b)):
            continue
        if not (torch.is_floating_point(tensor_a) and torch.is_floating_point(tensor_b)):
            continue
        diff = tensor_a.detach().float() - tensor_b.detach().float()
        distance_sq += float(torch.sum(diff * diff).item())
    return math.sqrt(distance_sq)


def interpolate_state_dicts(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Linearly interpolate matching floating-point tensors and copy the rest from state_a."""
    result: dict[str, torch.Tensor] = {}
    for key, value_a in state_a.items():
        value_a = value_a.detach().clone()
        value_b = state_b.get(key)
        if value_b is None:
            result[key] = value_a
            continue
        if not (torch.is_tensor(value_a) and torch.is_tensor(value_b)):
            result[key] = value_a
            continue
        if torch.is_floating_point(value_a) and torch.is_floating_point(value_b):
            result[key] = (1.0 - alpha) * value_a + alpha * value_b.detach().to(value_a.dtype)
        else:
            result[key] = value_a
    return result


def summarize_transform_metadata(metadata: dict[str, Any]) -> dict[str, float | int]:
    """Collapse raw transform metadata into scalar diagnostics for logging."""
    layer_summaries = metadata.get("layers", [])
    qk_orth_errors: list[float] = []
    vo_orth_errors: list[float] = []
    qk_scales: list[float] = []
    vo_scales: list[float] = []
    ffn_identity: list[float] = []

    for layer_info in layer_summaries:
        attention = layer_info.get("attention")
        if attention is not None:
            for rotation in attention["qk"].get("rotations", []):
                ident = torch.eye(rotation.shape[0], dtype=rotation.dtype)
                qk_orth_errors.append(float(torch.norm(rotation.transpose(0, 1) @ rotation - ident).item()))
            for rotation in attention["vo"].get("rotations", []):
                ident = torch.eye(rotation.shape[0], dtype=rotation.dtype)
                vo_orth_errors.append(float(torch.norm(rotation.transpose(0, 1) @ rotation - ident).item()))
            qk_scales.extend(float(scale) for scale in attention["qk"].get("scales", []))
            vo_scales.extend(float(scale) for scale in attention["vo"].get("scales", []))
        ffn = layer_info.get("ffn")
        if ffn is not None:
            permutation = ffn["permutation"]
            ident = torch.eye(permutation.shape[0], dtype=permutation.dtype)
            ffn_identity.append(float(torch.norm(permutation - ident).item()))

    def _mean(values: list[float]) -> float:
        if not values:
            return float("nan")
        return sum(values) / len(values)

    return {
        "matched_layers": len(layer_summaries),
        "mean_qk_orth_error": _mean(qk_orth_errors),
        "mean_vo_orth_error": _mean(vo_orth_errors),
        "mean_qk_scale": _mean(qk_scales),
        "mean_vo_scale": _mean(vo_scales),
        "mean_ffn_identity_deviation": _mean(ffn_identity),
    }


def canonicalize_model_to_reference(
    model: BertForMaskedLM,
    reference_model: BertForMaskedLM,
    *,
    use_attention: bool = True,
    use_ffn: bool = True,
    use_rescaling: bool = False,
    layer_indices: list[int] | None = None,
) -> tuple[BertForMaskedLM, dict[str, Any]]:
    """Match a BERT model into the basis of a fixed reference model."""
    matched_model = copy.deepcopy(model).eval()
    config = matched_model.config
    num_heads = config.num_attention_heads
    head_size = config.hidden_size // num_heads

    if layer_indices is None:
        layer_indices = list(range(config.num_hidden_layers))

    metadata: dict[str, Any] = {
        "layers": [],
        "use_attention": use_attention,
        "use_ffn": use_ffn,
        "use_rescaling": use_rescaling,
        "layer_indices": list(layer_indices),
    }

    for layer_idx in layer_indices:
        local_layer = matched_model.bert.encoder.layer[layer_idx]
        anchor_layer = reference_model.bert.encoder.layer[layer_idx]
        layer_metadata: dict[str, Any] = {"layer_idx": layer_idx}

        if use_attention:
            local_attention = get_layer_attention_tensors(local_layer)
            anchor_attention = get_layer_attention_tensors(anchor_layer)
            matched_qk, qk_meta = match_attention_qk_tensors(
                local_attention,
                anchor_attention,
                num_attention_heads=num_heads,
                attention_head_size=head_size,
                use_rescaling=use_rescaling,
            )
            matched_vo, vo_meta = match_attention_vo_tensors(
                local_attention,
                anchor_attention,
                num_attention_heads=num_heads,
                attention_head_size=head_size,
                use_rescaling=use_rescaling,
            )
            matched_attention = {
                **matched_qk,
                **matched_vo,
            }
            apply_layer_attention_tensors(local_layer, matched_attention)
            layer_metadata["attention"] = {"qk": qk_meta, "vo": vo_meta}

        if use_ffn:
            matched_ffn, ffn_meta = match_ffn_tensors(
                get_layer_ffn_tensors(local_layer),
                get_layer_ffn_tensors(anchor_layer),
            )
            apply_layer_ffn_tensors(local_layer, matched_ffn)
            layer_metadata["ffn"] = ffn_meta

        metadata["layers"].append(layer_metadata)

    return matched_model, metadata


def select_reference_seed(
    serialized_by_seed: dict[int, torch.Tensor],
    seeds: list[int],
    *,
    vocab_size: int,
    hidden_size: int,
    metric: str = "embeddings",
) -> int:
    """Choose the seed closest to the average projected model, matching upstream anchor selection."""
    if not seeds:
        raise ValueError("Expected at least one seed")

    def project(serialized: torch.Tensor) -> torch.Tensor:
        if metric == "full":
            return serialized[:, :hidden_size].reshape(-1).float()
        if metric == "embeddings":
            return serialized[:vocab_size, :hidden_size].reshape(-1).float()
        raise ValueError(f"Unsupported metric: {metric}")

    projected = {seed: project(serialized_by_seed[seed]) for seed in seeds}
    average = torch.stack([projected[seed] for seed in seeds], dim=0).mean(dim=0)
    return min(seeds, key=lambda seed: float(torch.norm(projected[seed] - average).item()))