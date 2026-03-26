from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from torch import Tensor
from transformers import BertConfig, BertForMaskedLM

from .serialization import deserialize, vector_component_labels


BIAS_CARRYING_TYPES = {"Q", "K", "V", "mlp_up"}


def bias_carrying_mask(config) -> Tensor:
    labels = vector_component_labels(config)
    return torch.tensor([label in BIAS_CARRYING_TYPES for label in labels], dtype=torch.bool)


@lru_cache(maxsize=None)
def get_multibert_snapshot(seed: int) -> str:
    return snapshot_download(
        f"google/multiberts-seed_{seed}",
        local_files_only=True,
    )


def load_multibert_model(seed: int) -> BertForMaskedLM:
    snapshot_path = get_multibert_snapshot(seed)
    config = BertConfig.from_pretrained(snapshot_path)
    config.tie_word_embeddings = False
    # The canonicalization experiment requests `output_attentions=True` and
    # captures internal activations via hooks. Force eager attention so those
    # features are available under recent Transformers defaults.
    config._attn_implementation = "eager"
    model = BertForMaskedLM(config)

    state_dict = torch.load(
        Path(snapshot_path) / "pytorch_model.bin",
        map_location="cpu",
        weights_only=True,
    )
    model_keys = set(model.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    model.load_state_dict(filtered_state, strict=False)
    return model


def apply_projection_to_bias_rows(
    projection: nn.Module,
    serialized: Tensor,
    config,
    device: torch.device,
) -> Tensor:
    """Apply the projection only to rows whose last dimension carries bias."""
    mask = bias_carrying_mask(config)
    projection = projection.to(device).eval()

    with torch.no_grad():
        reconstructed = serialized.clone()
        reconstructed[mask] = projection(serialized[mask].to(device)).cpu()

    return reconstructed


def apply_projection_to_bias_rows_with_grad(
    projection: nn.Module,
    serialized: Tensor,
    config,
    device: torch.device,
) -> Tensor:
    """Differentiable variant used during training.

    Non-bias rows are passed through unchanged while bias-carrying rows go
    through the trainable projection.
    """
    mask = bias_carrying_mask(config).to(device)
    serialized_device = serialized.to(device)
    reconstructed = serialized_device.clone()
    reconstructed[mask] = projection(serialized_device[mask])
    return reconstructed


def absorb_bias_rows_only(
    projection: nn.Module,
    serialized: Tensor,
    config,
    device: torch.device,
) -> Tensor:
    """Map serialized vectors from (d_model+1) to d_model.

    Non-bias rows simply drop the final zero-padded slot. Bias-carrying rows are
    encoded through the shared bias projection so the bias scalar is absorbed
    into the d_model representation.
    """
    mask = bias_carrying_mask(config).to(device)
    serialized_device = serialized.to(device)
    absorbed = serialized_device[:, : config.hidden_size].clone()
    absorbed[mask] = projection.encode(serialized_device[mask])
    return absorbed


def restore_bias_rows_only(
    projection: nn.Module,
    absorbed: Tensor,
    config,
    device: torch.device,
) -> Tensor:
    """Map absorbed vectors from d_model back to (d_model+1).

    Non-bias rows recover a zero-padded final slot. Bias-carrying rows are
    decoded through the shared bias projection to recover weight + bias form.
    """
    mask = bias_carrying_mask(config).to(device)
    absorbed_device = absorbed.to(device)
    restored = torch.cat(
        [
            absorbed_device,
            torch.zeros(
                absorbed_device.shape[0],
                1,
                device=device,
                dtype=absorbed_device.dtype,
            ),
        ],
        dim=1,
    )
    restored[mask] = projection.decode(absorbed_device[mask])
    return restored


def extract_non_bert_params(model: BertForMaskedLM) -> dict[str, Tensor]:
    """Return pretrained parameters outside the BertModel body.

    This is primarily the untied MLM head (`cls.*`). These parameters must be
    supplied explicitly when using torch.func.functional_call with deserialized
    `bert.*` parameters, otherwise the shell model's fallback head is used.
    """
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("bert.")
    }


def assemble_reconstructed_model(params: dict[str, Tensor], seed: int) -> BertForMaskedLM:
    model = load_multibert_model(seed).eval()
    model.bert.load_state_dict(params)
    return model


def reconstruct_model(
    projection: nn.Module,
    serialized: Tensor,
    config,
    seed: int,
    device: torch.device,
) -> BertForMaskedLM:
    reconstructed = apply_projection_to_bias_rows(projection, serialized, config, device)
    params = deserialize(reconstructed, config)
    return assemble_reconstructed_model(params, seed)


def zero_bias_dimension(serialized: Tensor, hidden_size: int) -> Tensor:
    out = serialized.clone()
    out[:, hidden_size] = 0.0
    return out


# ── BiasProjection model ──────────────────────────────────────────────────


class BiasProjection(nn.Module):
    """Linear autoencoder: (d_model+1) → d_model → (d_model+1).

    W_in compresses bias-absorbed vectors; W_out reconstructs them.
    Initialized as identity (discards bias at step 0).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.w_in = nn.Linear(d_model + 1, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model + 1, bias=False)
        self._init_identity()

    def _init_identity(self):
        with torch.no_grad():
            self.w_in.weight.zero_()
            self.w_in.weight[:, :self.d_model] = torch.eye(self.d_model)
            self.w_out.weight.zero_()
            self.w_out.weight[:self.d_model, :] = torch.eye(self.d_model)

    def encode(self, x: Tensor) -> Tensor:
        """(*, d_model+1) → (*, d_model)"""
        return self.w_in(x)

    def decode(self, z: Tensor) -> Tensor:
        """(*, d_model) → (*, d_model+1)"""
        return self.w_out(z)

    def forward(self, x: Tensor) -> Tensor:
        """Round-trip: (*, d_model+1) → (*, d_model+1)"""
        return self.decode(self.encode(x))


# ── Training ──────────────────────────────────────────────────────────────


def train_projection(
    train_data: Tensor,
    d_model: int,
    model: nn.Module | None = None,
    n_epochs: int = 5,
    log_every: int = 1,
    lr: float = 2e-4,
    convergence_tol: float = 1e-8,
    batch_size: int = 8192,
    tag: str = "shared",
    bias_scale: float = 1.0,
    device: torch.device | None = None,
) -> nn.Module:
    """Train a bias projection module on the given (N, d+1) data.

    Args:
        train_data: (N, d_model+1) tensor of all training vectors.
        d_model: hidden size.
        model: module to train (default: BiasProjection).
        device: target device. Defaults to CUDA if available, else CPU.
        tag: label for logging.

    Returns:
        Trained module (on CPU).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = BiasProjection(d_model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n = train_data.shape[0]
    prev_loss = float("inf")

    print(f"\n{'='*60}")
    print(f"Training {tag}  |  {n:,} vectors  |  {sum(p.numel() for p in model.parameters()):,} params")
    print(f"{'='*60}")

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = train_data[idx].to(device)
            if bias_scale != 1.0:
                batch = batch.clone()
                batch[:, d_model] *= bias_scale

            x_hat = model(batch)
            loss = (batch - x_hat).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            print(f"  [{tag}] epoch {epoch:4d}  loss {avg_loss:.12f}")

        rel_change = abs(prev_loss - avg_loss) / (prev_loss + 1e-12)
        if rel_change < convergence_tol and epoch > 5:
            print(f"  [{tag}] converged at epoch {epoch}  loss {avg_loss:.12f}")
            break
        prev_loss = avg_loss

    return model.cpu()


# ── Evaluation utilities ──────────────────────────────────────────────────


def reconstruction_mse_in_batches(
    model: nn.Module,
    inputs: Tensor,
    target: Tensor | None = None,
    batch_size: int = 8192,
    bias_scale: float = 1.0,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Compute total and bias-dimension MSE without materializing all outputs on GPU.

    Returns:
        (total_mse, bias_dim_mse) — mean over all elements / mean over rows.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target is None:
        target = inputs

    model = model.to(device).eval()
    total_sq_error = 0.0
    total_bias_sq_error = 0.0
    total_values = 0
    total_rows = 0

    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch_in = inputs[start:start + batch_size].to(device)
            batch_target = target[start:start + batch_size]
            if bias_scale != 1.0:
                batch_in = batch_in.clone()
                batch_in[:, -1] *= bias_scale
                batch_target = batch_target.clone()
                batch_target[:, -1] *= bias_scale
            batch_recon = model(batch_in).cpu()

            diff = batch_target - batch_recon
            total_sq_error += diff.pow(2).sum().item()
            total_bias_sq_error += diff[:, -1].pow(2).sum().item()
            total_values += diff.numel()
            total_rows += diff.shape[0]

    model = model.cpu()
    return total_sq_error / total_values, total_bias_sq_error / total_rows


def compute_reconstruction_errors(
    model: nn.Module,
    data: Tensor,
    labels: list[str],
    config,
    device: torch.device | None = None,
) -> dict:
    """Per-component-type reconstruction error.

    Returns dict mapping component type → {mse, relative_mse, count}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reconstructed = apply_projection_to_bias_rows(model, data, config, device)
    results = {}
    unique_types = sorted(set(labels))

    with torch.no_grad():
        for ctype in unique_types:
            mask = [i for i, l in enumerate(labels) if l == ctype]
            subset = data[mask]
            x_hat = reconstructed[mask]
            mse = (subset - x_hat).pow(2).mean().item()
            norm_sq = subset.pow(2).mean().item()
            results[ctype] = {
                "mse": mse,
                "relative_mse": mse / (norm_sq + 1e-12),
                "count": len(mask),
            }

    return results


def compute_bias_accuracy(
    model: nn.Module,
    data: Tensor,
    labels: list[str],
    config,
    device: torch.device | None = None,
) -> dict:
    """Bias reconstruction accuracy for bias-carrying vectors.

    Returns dict mapping component type → {correlation, mae, count}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reconstructed = apply_projection_to_bias_rows(model, data, config, device)
    results = {}

    with torch.no_grad():
        for ctype in sorted(BIAS_CARRYING_TYPES):
            mask = [i for i, l in enumerate(labels) if l == ctype]
            if not mask:
                continue
            subset = data[mask]
            x_hat = reconstructed[mask]

            orig_bias = subset[:, -1]
            recon_bias = x_hat[:, -1]

            mae = (orig_bias - recon_bias).abs().mean().item()

            o = orig_bias - orig_bias.mean()
            r = recon_bias - recon_bias.mean()
            corr_num = (o * r).sum()
            corr_den = (o.pow(2).sum() * r.pow(2).sum()).sqrt()
            corr = (corr_num / (corr_den + 1e-12)).item()

            results[ctype] = {"correlation": corr, "mae": mae, "count": len(mask)}

    return results


@torch.no_grad()
def compute_mlm_perplexity(model: BertForMaskedLM, batch: dict) -> float:
    """Compute masked language modeling perplexity on the given batch.

    Args:
        model: BertForMaskedLM — already placed on its target device.
        batch: dict with keys input_ids, attention_mask, labels
               (labels use -100 to mark non-masked positions).

    Returns:
        Perplexity (float).
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    total_loss = 0.0
    total_tokens = 0
    for i in range(input_ids.shape[0]):
        out = model(
            input_ids=input_ids[i:i+1],
            attention_mask=attention_mask[i:i+1],
            labels=labels[i:i+1],
        )
        n_masked = (labels[i] != -100).sum().item()
        if n_masked > 0:
            total_loss += out.loss.item() * n_masked
            total_tokens += n_masked

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


# ── Symmetry utilities ────────────────────────────────────────────────────


def apply_neuron_permutation(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    perm: Tensor,
) -> dict[str, Tensor]:
    """Apply MLP neuron permutation to a BERT state dict.

    Permutes rows of intermediate.dense (weight + bias) and columns of
    output.dense.weight together, preserving the function of the layer.
    """
    out = {k: v.clone() for k, v in state_dict.items()}
    pre = f"encoder.layer.{layer_idx}"

    w_key = f"{pre}.intermediate.dense.weight"
    b_key = f"{pre}.intermediate.dense.bias"
    out[w_key] = out[w_key][perm]
    out[b_key] = out[b_key][perm]

    w2_key = f"{pre}.output.dense.weight"
    out[w2_key] = out[w2_key][:, perm]

    return out


def apply_head_permutation(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    perm: Tensor,
    num_heads: int,
    d_model: int,
) -> dict[str, Tensor]:
    """Apply attention head permutation to a BERT state dict.

    Moves all Q, K, V, O vectors for each head together, preserving
    the function of the layer.
    """
    out = {k: v.clone() for k, v in state_dict.items()}
    pre = f"encoder.layer.{layer_idx}"
    d_head = d_model // num_heads

    for proj in ["query", "key", "value"]:
        w_key = f"{pre}.attention.self.{proj}.weight"
        b_key = f"{pre}.attention.self.{proj}.bias"
        w = out[w_key].view(num_heads, d_head, d_model)
        b = out[b_key].view(num_heads, d_head)
        out[w_key] = w[perm].reshape(d_model, d_model)
        out[b_key] = b[perm].reshape(d_model)

    o_key = f"{pre}.attention.output.dense.weight"
    w_o = out[o_key].view(d_model, num_heads, d_head)
    out[o_key] = w_o[:, perm, :].reshape(d_model, d_model)

    return out