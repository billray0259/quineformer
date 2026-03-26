"""Canonicalization Experiment V1: Interpolation-Aligned Coordinate Canonicalization.

Trains a CanonicalizationModule that maps MultiBERT parameters into a shared
canonical coordinate system where linear interpolation of canonical parameters
produces functionally valid intermediate models.

Pipeline per model:
    serialized (N, 769)
    → W_in (frozen bias absorption) → absorbed (N, 768)
    → extract word embeddings → CanonicalizationModule → P
    → canonical = absorbed @ P

Interpolation:
    canon_interp = (1 - α) * canonical_i + α * canonical_j
    interp = canon_interp @ P_i^{-1}
    → W_out (frozen) → (N, 769) → deserialize → BERT state dict

Loss: MSE between activations of the canon-interpolated model and
activation-space interpolation of the two endpoint models, averaged
across all 12 layers. Plus entropy regularization on P.

Evaluation: perplexity ratio of α=0.5 interpolated models vs. a
logit-ensemble of the same pair.

See experiments/canonicalization/experiment_v1.md for full specification.
"""

import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from dotenv import load_dotenv

try:
    import wandb
except ImportError:
    wandb = None

from quineformer.bias_absorption import (
    BiasProjection,
    bias_carrying_mask,
    compute_mlm_perplexity,
    load_multibert_model,
)
from quineformer.canonicalization import CanonicalizationModule
from quineformer.serialization import deserialize

# ── Reuse data loading from bias_absorption experiment ───────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bias_absorption"))
from run_v1 import (
    TRAIN_SEEDS,
    TEST_SEEDS,
    get_reference_batch,
    load_and_serialize_all,
)

# ── Configuration ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_v1"
BIAS_ABSORPTION_DIR = SCRIPT_DIR.parent / "bias_absorption"

# Training hyperparameters
N_EPOCHS = 50
SAVE_EVERY = 5  # save intermediate results every this many epochs
LR = 1e-4
WEIGHT_DECAY = 0.01
GRAD_ACCUM_STEPS = 4
LAMBDA_SHARP = 0.1
LOG_TAU_MIN = math.log(0.05)
LOG_TAU_MAX = math.log(2.0)

# Reference batch for activation extraction
REF_SEQ_LEN = 128
TEXT_POOL_SIZE = 2048  # number of pre-tokenized sequences for training
TRAIN_BATCH_SIZE = 32   # sequences sampled per training step (3 BERT passes/step)
WANDB_PROJECT = "quineformer-canonicalization"
USE_AMP = False
TRAIN_ALPHA = 0.5
RECONSTRUCTION_SANITY_CHECK = False
LAMBDA_MLM = 1.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def load_frozen_projection() -> BiasProjection:
    """Load the pre-trained bias absorption projection and freeze it."""
    projection = BiasProjection(d_model=768)
    path = BIAS_ABSORPTION_DIR / "results_v1_min" / "projection_shared.pt"
    projection.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    projection.eval()
    for p in projection.parameters():
        p.requires_grad_(False)
    return projection


def absorb_bias_rows_only(
    projection: BiasProjection,
    serialized: torch.Tensor,
    config: BertConfig,
    device: torch.device,
) -> torch.Tensor:
    """Map (N, 769) -> (N, 768) using the learned projection only on bias rows."""
    mask = bias_carrying_mask(config).to(device)
    serialized = serialized.to(device)
    absorbed = serialized[:, :config.hidden_size].clone()
    absorbed[mask] = projection.encode(serialized[mask])
    return absorbed


def restore_bias_rows_only(
    projection: BiasProjection,
    absorbed: torch.Tensor,
    config: BertConfig,
    device: torch.device,
) -> torch.Tensor:
    """Map (N, 768) -> (N, 769) using the learned decoder only on bias rows."""
    mask = bias_carrying_mask(config).to(device)
    absorbed = absorbed.to(device)
    restored = torch.cat(
        [
            absorbed,
            torch.zeros(absorbed.shape[0], 1, device=device, dtype=absorbed.dtype),
        ],
        dim=1,
    )
    restored[mask] = projection.decode(absorbed[mask])
    return restored


class StreamingTextPool:
    """Continuously refresh a pool of token sequences from the training corpus."""

    def __init__(
        self,
        tokenizer,
        pool_size: int = TEXT_POOL_SIZE,
        seq_len: int = REF_SEQ_LEN,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.pool_size = pool_size
        self.seq_len = seq_len
        self.inner_len = seq_len - 2
        self.ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        self.row_order: list[int] = []
        self.row_ptr = 0
        self.buf: list[int] = []
        self._reshuffle_rows()
        self.pool = torch.stack([self._next_sequence() for _ in range(pool_size)])

    def _reshuffle_rows(self):
        self.row_order = torch.randperm(len(self.ds)).tolist()
        self.row_ptr = 0

    def _next_sequence(self) -> torch.Tensor:
        while len(self.buf) < self.inner_len:
            if self.row_ptr >= len(self.row_order):
                self._reshuffle_rows()
            text = self.ds[self.row_order[self.row_ptr]]["text"].strip()
            self.row_ptr += 1
            if not text:
                continue
            self.buf.extend(self.tokenizer.encode(text, add_special_tokens=False))

        ids = [self.tokenizer.cls_token_id] + self.buf[:self.inner_len] + [self.tokenizer.sep_token_id]
        self.buf = self.buf[self.inner_len:]
        return torch.tensor(ids, dtype=torch.long)

    def sample_masked_batch(self, batch_size: int, mask_token_id: int) -> dict:
        idx = torch.randint(self.pool_size, (batch_size,))
        input_ids = self.pool[idx].clone()
        for pool_idx in idx.tolist():
            self.pool[pool_idx] = self._next_sequence()

        labels = input_ids.clone()
        mask_prob = torch.rand_like(input_ids, dtype=torch.float)
        mask_prob[:, 0] = 1.0
        mask_prob[:, -1] = 1.0
        mask_positions = mask_prob < 0.15
        input_ids[mask_positions] = mask_token_id
        labels[~mask_positions] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }


def sample_masked_batch(
    pool: StreamingTextPool,
    batch_size: int,
    mask_token_id: int,
) -> dict:
    """Sample a fresh mini-batch with randomly sampled mask positions."""
    return pool.sample_masked_batch(batch_size, mask_token_id)


def get_residual_stream_states(
    shell_model: BertForMaskedLM,
    params: dict[str, torch.Tensor],
    ref_batch: dict,
    device: torch.device,
    return_loss: bool = False,
    extra_params: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor]:
    """Run a differentiable forward pass and return only residual stream states."""
    full_params = {}
    for k, v in params.items():
        full_params[f"bert.{k}"] = v.to(device)
    if extra_params is not None:
        for k, v in extra_params.items():
            full_params[k] = v.to(device)
    for k, v in shell_model.state_dict().items():
        if k not in full_params:
            full_params[k] = v.to(device)

    kwargs = dict(
        input_ids=ref_batch["input_ids"].to(device),
        attention_mask=ref_batch["attention_mask"].to(device),
        output_hidden_states=True,
    )
    if return_loss and "labels" in ref_batch:
        kwargs["labels"] = ref_batch["labels"].to(device)

    out = torch.func.functional_call(shell_model, full_params, kwargs=kwargs)
    activations = list(out.hidden_states)
    if return_loss:
        return activations, out.loss
    return activations


def extract_non_bert_params(model: BertForMaskedLM) -> dict[str, torch.Tensor]:
    """Return pretrained params that live outside the BertModel body.

    These are primarily the untied MLM head weights (`cls.*`). They must be
    overridden during `functional_call` when computing perplexity; otherwise the
    shell model's fallback head is used.
    """
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("bert.")
    }


def get_all_hidden_states(
    shell_model: BertForMaskedLM,
    params: dict[str, torch.Tensor],
    ref_batch: dict,
    device: torch.device,
    return_loss: bool = False,
    extra_params: dict[str, torch.Tensor] | None = None,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor]:
    """Run a differentiable forward pass and return all intermediate activations.

    Uses functional_call to swap in the given parameters without mutating
    the shell model. Returns a flat list of 73 tensors:
      - 13 residual stream states  (embedding output + 12 layer outputs)
      - 12 attention weight matrices  (batch, n_heads, seq, seq)
      - 12 × Q projection outputs     (batch, seq, d_model)
      - 12 × K projection outputs     (batch, seq, d_model)
      - 12 × V projection outputs     (batch, seq, d_model)
      - 12 × MLP intermediate outputs (batch, seq, d_ff)
    The hook-captured tensors are ordered Q_0,K_0,V_0,MLP_0, Q_1,… so the
    ordering is identical across calls and MSE pairing is correct.

    If return_loss=True, also returns the scalar MLM cross-entropy loss
    (requires ref_batch to contain "labels"). Cost is negligible — just the
    CLS head projection + cross-entropy on top of the last hidden state.
    """
    full_params = {}
    for k, v in params.items():
        full_params[f"bert.{k}"] = v.to(device)
    if extra_params is not None:
        for k, v in extra_params.items():
            full_params[k] = v.to(device)
    for k, v in shell_model.state_dict().items():
        if k not in full_params:
            full_params[k] = v.to(device)

    # Register hooks to capture Q/K/V projections and MLP intermediate acts.
    captured: list[torch.Tensor] = []
    hooks = []
    for layer in shell_model.bert.encoder.layer:
        hooks.append(layer.attention.self.query.register_forward_hook(
            lambda m, inp, out: captured.append(out)
        ))
        hooks.append(layer.attention.self.key.register_forward_hook(
            lambda m, inp, out: captured.append(out)
        ))
        hooks.append(layer.attention.self.value.register_forward_hook(
            lambda m, inp, out: captured.append(out)
        ))
        hooks.append(layer.intermediate.register_forward_hook(
            lambda m, inp, out: captured.append(out)
        ))

    input_ids = ref_batch["input_ids"].to(device)
    attention_mask = ref_batch["attention_mask"].to(device)
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        output_attentions=True,
    )
    if return_loss and "labels" in ref_batch:
        kwargs["labels"] = ref_batch["labels"].to(device)

    try:
        out = torch.func.functional_call(shell_model, full_params, kwargs=kwargs)
    finally:
        for h in hooks:
            h.remove()

    activations = list(out.hidden_states)   # 13 residual stream states
    activations.extend(out.attentions)      # 12 attention weight matrices
    activations.extend(captured)            # 48: Q,K,V,MLP per layer

    if return_loss:
        return activations, out.loss        # out.loss is None if no labels
    return activations  # 73 tensors total


def canonicalize_model(
    serialized: torch.Tensor,
    projection: BiasProjection,
    canon_module: CanonicalizationModule,
    config: BertConfig,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply bias absorption + canonicalization to one serialized model.

    Returns:
        canonical: (N, 768) canonicalized parameter vectors.
        P: (768, 768) soft permutation matrix (no batch dim).
    """
    s = serialized.to(device)
    absorbed = absorb_bias_rows_only(projection, s, config, device)
    E = absorbed[:vocab_size].unsqueeze(0)               # (1, V, 768)
    _, P = canon_module(E)                               # (1, 768, 768)
    P = P.squeeze(0)                                     # (768, 768)
    canonical = absorbed @ P                             # (N, 768)
    return canonical, P


def invert_soft_permutation(P: torch.Tensor) -> torch.Tensor:
    """Return a stable inverse for a soft Sinkhorn matrix.

    The Sinkhorn output is only approximately permutation-like, so `P.T` is not
    the correct inverse unless P is exactly orthogonal. Use a pseudoinverse in
    float32 for numerical stability and to preserve gradient flow.
    """
    return torch.linalg.pinv(P.float())


def invert_soft_permutation_with_metrics(
    P: torch.Tensor,
    acted_matrix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Choose the most reliable inverse available and report diagnostics.

    Preference order:
    1. Exact inverse, if it exists and gives finite low residuals.
    2. Pseudoinverse fallback.

    Metrics are computed in float64 on detached tensors so they reflect
    numerical quality without perturbing the training graph.
    """
    P32 = P.float()
    P64 = P32.detach().double()
    eye = torch.eye(P64.shape[0], device=P64.device, dtype=P64.dtype)

    method = "pinv"
    inv_used = torch.linalg.pinv(P32)

    try:
        candidate = torch.linalg.inv(P32)
        candidate64 = candidate.detach().double()
        left = P64 @ candidate64 - eye
        right = candidate64 @ P64 - eye
        left_rel = left.norm(p="fro").item() / math.sqrt(P64.shape[0])
        right_rel = right.norm(p="fro").item() / math.sqrt(P64.shape[0])
        if math.isfinite(left_rel) and math.isfinite(right_rel) and left_rel < 1e-3 and right_rel < 1e-3:
            inv_used = candidate
            method = "inv"
    except RuntimeError:
        pass

    inv64 = inv_used.detach().double()
    left = P64 @ inv64 - eye
    right = inv64 @ P64 - eye
    metrics = {
        "method_code": 0.0 if method == "inv" else 1.0,
        "left_residual_fro": left.norm(p="fro").item(),
        "right_residual_fro": right.norm(p="fro").item(),
        "left_residual_rel": left.norm(p="fro").item() / math.sqrt(P64.shape[0]),
        "right_residual_rel": right.norm(p="fro").item() / math.sqrt(P64.shape[0]),
        "cond": torch.linalg.cond(P64).item(),
    }

    if acted_matrix is not None:
        acted64 = acted_matrix.detach().double()
        recon = acted64 @ P64 @ inv64
        denom = acted64.norm(p="fro").item()
        metrics["param_reconstruction_rel"] = (
            (recon - acted64).norm(p="fro").item() / max(denom, 1e-12)
        )

    return inv_used, metrics


def interpolate_and_reconstruct(
    canon_i: torch.Tensor,
    canon_j: torch.Tensor,
    P_i: torch.Tensor,
    alpha: float,
    projection: BiasProjection,
    config: BertConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Interpolate in canonical space and reconstruct a BERT state dict.

    Returns:
        params: dict mapping BertModel state_dict keys to tensors.
    """
    P_i_inv, _ = invert_soft_permutation_with_metrics(P_i, acted_matrix=canon_i)
    canon_interp = (1 - alpha) * canon_i + alpha * canon_j  # (N, 768)
    interp_768 = canon_interp.float() @ P_i_inv              # (N, 768)
    interp_769 = restore_bias_rows_only(projection, interp_768, config, device)
    params = deserialize(interp_769, config)
    return params


def compute_activation_mse(
    A_pred: list[torch.Tensor],
    A_target: list[torch.Tensor],
) -> tuple[torch.Tensor, list[float]]:
    """Compute per-layer and mean activation MSE.

    Args:
        A_pred: list of 13 activation tensors from interpolated model.
        A_target: list of 13 target activation tensors.

    Returns:
        mean_mse: scalar tensor (differentiable).
        per_layer: list of 13 floats (detached, for logging).
    """
    per_layer = []
    total = torch.tensor(0.0, device=A_pred[0].device)
    for a_pred, a_tgt in zip(A_pred, A_target):
        mse = F.mse_loss(a_pred, a_tgt)
        total = total + mse
        per_layer.append(mse.item())
    mean_mse = total / len(A_pred)
    return mean_mse, per_layer


def clamp_log_tau(canon_module: CanonicalizationModule):
    """Clamp log_tau to prevent degenerate temperature."""
    with torch.no_grad():
        canon_module.log_tau.clamp_(LOG_TAU_MIN, LOG_TAU_MAX)


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return obj


def init_wandb(config: BertConfig):
    """Initialize a W&B run if wandb is installed and WANDB_API_KEY is set."""
    if wandb is None:
        print("\n▸ W&B disabled: package not installed")
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("\n▸ W&B disabled: WANDB_API_KEY not set")
        return None

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
        entity=os.environ.get("WANDB_ENTITY"),
        name=os.environ.get("WANDB_RUN_NAME"),
        config={
            "n_epochs": N_EPOCHS,
            "save_every": SAVE_EVERY,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_accum_steps": GRAD_ACCUM_STEPS,
            "lambda_sharp": LAMBDA_SHARP,
            "lambda_mlm": LAMBDA_MLM,
            "log_tau_min": LOG_TAU_MIN,
            "log_tau_max": LOG_TAU_MAX,
            "ref_seq_len": REF_SEQ_LEN,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "text_pool_size": TEXT_POOL_SIZE,
            "train_alpha": TRAIN_ALPHA,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "train_seeds": TRAIN_SEEDS,
            "test_seeds": TEST_SEEDS,
        },
    )
    print(f"\n▸ W&B enabled: {run.project}/{run.name}")
    return run


def wandb_log(run, metrics: dict, step: int | None = None):
    if run is not None:
        run.log(metrics, step=step)


# ── Training ─────────────────────────────────────────────────────────────────


def train(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    serialized: list[torch.Tensor],
    config: BertConfig,
    tokenizer,
    shell_model: BertForMaskedLM,
    wandb_run=None,
):
    """Train the canonicalization module."""
    canon_module = canon_module.to(DEVICE)
    projection = projection.to(DEVICE)
    shell_model = shell_model.to(DEVICE).eval()

    vocab_size = config.vocab_size
    train_pairs = list(combinations(TRAIN_SEEDS, 2))

    optimizer = torch.optim.AdamW(
        canon_module.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    # Cosine schedule with warmup
    total_steps = N_EPOCHS * len(train_pairs) // GRAD_ACCUM_STEPS
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")
    training_log = []

    # Pre-tokenize a large text pool; masking is sampled fresh each step
    print("\n▸ Building streaming text sampler...")
    text_pool = StreamingTextPool(tokenizer)
    print(f"  Pool: ({text_pool.pool_size}, {text_pool.seq_len}) sequences with continual refresh")

    # Pre-encode all models through frozen W_in (saves repeated computation)
    print("\n▸ Pre-encoding all models through frozen W_in...")
    absorbed = {}
    for seed in tqdm(TRAIN_SEEDS + TEST_SEEDS, desc="  Encoding", unit="model"):
        with torch.no_grad():
            absorbed[seed] = absorb_bias_rows_only(
                projection, serialized[seed], config, DEVICE,
            ).cpu()
    print(f"  Encoded {len(absorbed)} models → (N, 768)")

    # Pre-deserialize endpoint model params (avoids repeated deserialize calls)
    print("\n▸ Pre-deserializing endpoint params for training models...")
    endpoint_params = {}
    for seed in tqdm(TRAIN_SEEDS, desc="  Deserializing", unit="model"):
        endpoint_params[seed] = deserialize(serialized[seed], config)
    print(f"  Deserialized {len(endpoint_params)} models")

    print("\n▸ Loading pretrained MLM heads for training models...")
    endpoint_head_params = {}
    for seed in tqdm(TRAIN_SEEDS, desc="  MLM heads", unit="model"):
        model = load_multibert_model(seed)
        endpoint_head_params[seed] = extract_non_bert_params(model)
        del model
    print(f"  Loaded {len(endpoint_head_params)} pretrained MLM heads")

    global_step = 0
    wandb_step = 0
    best_val_loss = float("inf")

    print(f"\n{'='*70}")
    print(f"TRAINING CANONICALIZATION MODULE")
    print(f"  {len(train_pairs)} pairs × {N_EPOCHS} epochs = {len(train_pairs) * N_EPOCHS} steps")
    print(f"  Grad accumulation: {GRAD_ACCUM_STEPS} → {total_steps} optimizer steps")
    print(f"  Warmup: {warmup_steps} steps")
    print("  Mode: interpolation training (alpha=0.5, residual-MSE + MLM loss)")
    print(f"{'='*70}")

    epoch_bar = tqdm(range(N_EPOCHS), desc="Epochs", unit="epoch", position=0)

    for epoch in epoch_bar:
        epoch_t0 = time.time()
        canon_module.train()

        # Shuffle pairs each epoch
        perm = torch.randperm(len(train_pairs))
        epoch_loss_act = 0.0
        epoch_loss_mlm = 0.0
        epoch_loss_sharp = 0.0
        epoch_loss_total = 0.0
        epoch_ppl_interp = 0.0
        epoch_ppl_endpoint = 0.0
        n_steps = 0

        optimizer.zero_grad()

        step_bar = tqdm(
            range(len(train_pairs)),
            desc=f"  Epoch {epoch:3d}",
            unit="pair",
            position=1,
            leave=False,
        )

        for step_idx in step_bar:
            pair_idx = perm[step_idx].item()
            seed_i, seed_j = train_pairs[pair_idx]
            alpha = TRAIN_ALPHA if TRAIN_ALPHA is not None else torch.rand(1).item()

            # Fresh mini-batch for this step (new text + new mask positions)
            batch = sample_masked_batch(
                text_pool, TRAIN_BATCH_SIZE, tokenizer.mask_token_id
            )

            # Canonicalize both models
            a_i = absorbed[seed_i].to(DEVICE)
            a_j = absorbed[seed_j].to(DEVICE)

            # Endpoint activations + perplexity on the fresh batch (no grad needed)
            with torch.no_grad():
                A_i, loss_i = get_residual_stream_states(
                    shell_model,
                    endpoint_params[seed_i],
                    batch,
                    DEVICE,
                    return_loss=True,
                    extra_params=endpoint_head_params[seed_i],
                )
                A_j, loss_j = get_residual_stream_states(
                    shell_model,
                    endpoint_params[seed_j],
                    batch,
                    DEVICE,
                    return_loss=True,
                    extra_params=endpoint_head_params[seed_j],
                )
                ppl_endpoint = math.exp(0.5 * (loss_i.item() + loss_j.item()))

            with autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda", dtype=torch.float16):
                E_i = a_i[:vocab_size].unsqueeze(0)
                E_j = a_j[:vocab_size].unsqueeze(0)
                _, P_i = canon_module(E_i)
                _, P_j = canon_module(E_j)

                P_i_mat = P_i.squeeze(0)
                P_j_mat = P_j.squeeze(0)

                canon_i = a_i @ P_i_mat
                canon_j = a_j @ P_j_mat

                # Interpolate in canonical space
                canon_interp = (1 - alpha) * canon_i + alpha * canon_j
                P_i_inv, inverse_metrics = invert_soft_permutation_with_metrics(
                    P_i_mat,
                    acted_matrix=a_i,
                )
                interp_768 = canon_interp.float() @ P_i_inv

                # Recover bias dimension
                interp_769 = restore_bias_rows_only(projection, interp_768, config, DEVICE)
                params_interp = deserialize(interp_769, config)

                # Get interpolated model's residual states + MLM loss (with grad)
                A_interp, loss_interp = get_residual_stream_states(
                    shell_model,
                    params_interp,
                    batch,
                    DEVICE,
                    return_loss=True,
                    extra_params=endpoint_head_params[seed_i],
                )
                ppl_interp = math.exp(min(loss_interp.item(), 20.0))  # cap at e^20 to avoid inf

                # Target: residual-stream interpolation on the same batch
                A_target = [
                    (1 - alpha) * ai.to(DEVICE) + alpha * aj.to(DEVICE)
                    for ai, aj in zip(A_i, A_j)
                ]

                # Residual-stream activation MSE loss
                loss_act, _ = compute_activation_mse(A_interp, A_target)
                loss_mlm = loss_interp

                # Sharpness regularization
                loss_sharp = 0.5 * (
                    canon_module.row_entropy(P_i)
                    + canon_module.row_entropy(P_j)
                )
                loss = loss_act + LAMBDA_MLM * loss_mlm + LAMBDA_SHARP * loss_sharp
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            epoch_loss_act += loss_act.item()
            epoch_loss_mlm += loss_mlm.item()
            epoch_loss_sharp += loss_sharp.item()
            epoch_loss_total += loss.item() * GRAD_ACCUM_STEPS
            epoch_ppl_interp += ppl_interp
            epoch_ppl_endpoint += ppl_endpoint
            n_steps += 1

            wandb_log(
                wandb_run,
                {
                    "train/step_loss_act": loss_act.item(),
                    "train/step_loss_mlm": loss_mlm.item(),
                    "train/step_loss_sharp": loss_sharp.item(),
                    "train/step_loss_total": loss.item() * GRAD_ACCUM_STEPS,
                    "train/step_ppl_interp": ppl_interp,
                    "train/step_ppl_endpoint": ppl_endpoint,
                    "train/step_ppl_ratio": ppl_interp / max(ppl_endpoint, 1e-12),
                    "train/tau": canon_module.tau.item(),
                    "train/alpha": alpha,
                    "train/epoch": epoch,
                    "inverse/method_code": inverse_metrics["method_code"],
                    "inverse/left_residual_fro": inverse_metrics["left_residual_fro"],
                    "inverse/right_residual_fro": inverse_metrics["right_residual_fro"],
                    "inverse/left_residual_rel": inverse_metrics["left_residual_rel"],
                    "inverse/right_residual_rel": inverse_metrics["right_residual_rel"],
                    "inverse/cond": inverse_metrics["cond"],
                    "inverse/param_reconstruction_rel": inverse_metrics.get("param_reconstruction_rel", 0.0),
                },
                step=wandb_step,
            )
            wandb_step += 1

            # Optimizer step every GRAD_ACCUM_STEPS
            if (step_idx + 1) % GRAD_ACCUM_STEPS == 0 or step_idx == len(train_pairs) - 1:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                clamp_log_tau(canon_module)
                global_step += 1

            step_bar.set_postfix(
                act=f"{loss_act.item():.4f}",
                mlm=f"{loss_mlm.item():.3f}",
                ppl=f"{ppl_interp:.1f}/{ppl_endpoint:.1f}",
                τ=f"{canon_module.tau.item():.3f}",
            )

            # Free memory
            del A_interp, A_target, A_i, A_j
            del interp_769, params_interp
            del canon_i, canon_j, canon_interp, interp_768
            del P_i, P_j
            torch.cuda.empty_cache()

        avg_act = epoch_loss_act / n_steps
        avg_mlm = epoch_loss_mlm / n_steps
        avg_sharp = epoch_loss_sharp / n_steps
        avg_total = epoch_loss_total / n_steps
        avg_ppl_interp = epoch_ppl_interp / n_steps
        avg_ppl_endpoint = epoch_ppl_endpoint / n_steps
        avg_ppl_ratio = avg_ppl_interp / avg_ppl_endpoint
        tau = canon_module.tau.item()
        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_t0

        epoch_log = {
            "epoch": epoch,
            "loss_act": avg_act,
            "loss_mlm": avg_mlm,
            "loss_sharp": avg_sharp,
            "loss_total": avg_total,
            "ppl_interp": avg_ppl_interp,
            "ppl_endpoint": avg_ppl_endpoint,
            "ppl_ratio": avg_ppl_ratio,
            "tau": tau,
            "lr": lr_now,
            "elapsed_s": elapsed,
        }
        training_log.append(epoch_log)

        wandb_log(
            wandb_run,
            {
                "epoch/loss_act": avg_act,
                "epoch/loss_mlm": avg_mlm,
                "epoch/loss_sharp": avg_sharp,
                "epoch/loss_total": avg_total,
                "epoch/ppl_interp": avg_ppl_interp,
                "epoch/ppl_endpoint": avg_ppl_endpoint,
                "epoch/ppl_ratio": avg_ppl_ratio,
                "epoch/tau": tau,
                "epoch/lr": lr_now,
                "epoch/elapsed_s": elapsed,
            },
            step=wandb_step,
        )

        epoch_bar.set_postfix(
            act=f"{avg_act:.4f}",
            mlm=f"{avg_mlm:.3f}",
            ppl=f"{avg_ppl_interp:.1f}/{avg_ppl_endpoint:.1f}",
            ratio=f"{avg_ppl_ratio:.3f}",
            τ=f"{tau:.3f}",
        )

        # Save best checkpoint
        if avg_total < best_val_loss:
            best_val_loss = avg_total
            torch.save(
                canon_module.state_dict(),
                RESULTS_DIR / "canonicalization_module.pt",
            )

        # Periodic intermediate saves
        if (epoch + 1) % SAVE_EVERY == 0:
            ckpt_path = RESULTS_DIR / f"canon_epoch_{epoch + 1:04d}.pt"
            torch.save(canon_module.state_dict(), ckpt_path)
            log_path = RESULTS_DIR / "training_log.json"
            with open(log_path, "w") as f:
                json.dump(make_serializable(training_log), f, indent=2)
            tqdm.write(
                f"  [epoch {epoch + 1}] checkpoint → {ckpt_path.name}  "
                f"log → {log_path.name}"
            )

    epoch_bar.close()
    return canon_module, training_log


# ── Evaluation ───────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_pair_perplexity(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    absorbed: dict[int, torch.Tensor],
    serialized: list[torch.Tensor],
    seed_i: int,
    seed_j: int,
    alpha: float,
    config: BertConfig,
    ref_batch: dict,
    shell_model: BertForMaskedLM,
) -> dict:
    """Evaluate a single pair: canon-interpolation PPL and ensemble PPL."""
    vocab_size = config.vocab_size
    canon_module.eval()

    a_i = absorbed[seed_i].to(DEVICE)
    a_j = absorbed[seed_j].to(DEVICE)

    # Canonicalize
    E_i = a_i[:vocab_size].unsqueeze(0)
    E_j = a_j[:vocab_size].unsqueeze(0)
    _, P_i = canon_module(E_i)
    _, P_j = canon_module(E_j)

    P_i_mat = P_i.squeeze(0)
    P_j_mat = P_j.squeeze(0)
    canon_i = a_i @ P_i_mat
    canon_j = a_j @ P_j_mat

    # Interpolate in canonical space
    canon_interp = (1 - alpha) * canon_i + alpha * canon_j
    P_i_inv, _ = invert_soft_permutation_with_metrics(P_i_mat, acted_matrix=a_i)
    interp_768 = canon_interp.float() @ P_i_inv
    interp_769 = restore_bias_rows_only(projection, interp_768, config, DEVICE)
    params_interp = deserialize(interp_769, config)

    # Build interpolated model
    interp_model = load_multibert_model(seed_i).eval().to(DEVICE)
    interp_model.bert.load_state_dict(
        {k: v.to(DEVICE) for k, v in params_interp.items()}
    )
    interp_ppl = compute_mlm_perplexity(interp_model, ref_batch)
    del interp_model
    torch.cuda.empty_cache()

    # Ensemble baseline: average logits
    model_i = load_multibert_model(seed_i).eval().to(DEVICE)
    model_j = load_multibert_model(seed_j).eval().to(DEVICE)

    input_ids = ref_batch["input_ids"].to(DEVICE)
    attention_mask = ref_batch["attention_mask"].to(DEVICE)
    labels = ref_batch["labels"].to(DEVICE)

    total_loss = 0.0
    total_tokens = 0
    for b in range(input_ids.shape[0]):
        logits_i = model_i(
            input_ids=input_ids[b:b+1],
            attention_mask=attention_mask[b:b+1],
        ).logits
        logits_j = model_j(
            input_ids=input_ids[b:b+1],
            attention_mask=attention_mask[b:b+1],
        ).logits
        ensemble_logits = (1 - alpha) * logits_i + alpha * logits_j

        # Compute cross-entropy loss on masked positions
        lab = labels[b:b+1]
        loss = F.cross_entropy(
            ensemble_logits.view(-1, ensemble_logits.size(-1)),
            lab.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_masked = (lab != -100).sum().item()
        if n_masked > 0:
            total_loss += loss.item()
            total_tokens += n_masked

    ensemble_ppl = math.exp(total_loss / max(total_tokens, 1))

    del model_i, model_j
    torch.cuda.empty_cache()

    return {
        "interp_ppl": interp_ppl,
        "ensemble_ppl": ensemble_ppl,
        "ratio": interp_ppl / ensemble_ppl,
    }


@torch.no_grad()
def evaluate_naive_interpolation(
    projection: BiasProjection,
    serialized: list[torch.Tensor],
    seed_i: int,
    seed_j: int,
    alpha: float,
    config: BertConfig,
    ref_batch: dict,
) -> float:
    """Naive interpolation baseline (no canonicalization)."""
    
    s_i = serialized[seed_i].to(DEVICE)
    s_j = serialized[seed_j].to(DEVICE)

    # Interpolate raw serialized (after bias absorption encode/decode)
    a_i = absorb_bias_rows_only(projection, s_i, config, DEVICE)
    a_j = absorb_bias_rows_only(projection, s_j, config, DEVICE)
    interp_768 = (1 - alpha) * a_i + alpha * a_j
    interp_769 = restore_bias_rows_only(projection, interp_768, config, DEVICE)
    params = deserialize(interp_769, config)

    model = load_multibert_model(seed_i).eval().to(DEVICE)
    model.bert.load_state_dict({k: v.to(DEVICE) for k, v in params.items()})
    ppl = compute_mlm_perplexity(model, ref_batch)
    del model
    torch.cuda.empty_cache()
    return ppl


@torch.no_grad()
def evaluate_interpolation_curve(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    absorbed: dict[int, torch.Tensor],
    serialized: list[torch.Tensor],
    seed_i: int,
    seed_j: int,
    config: BertConfig,
    ref_batch: dict,
    shell_model: BertForMaskedLM,
    n_points: int = 11,
) -> list[dict]:
    """Evaluate perplexity at evenly-spaced α values."""
    alphas = [k / (n_points - 1) for k in range(n_points)]
    curve = []
    for alpha in alphas:
        result = evaluate_pair_perplexity(
            canon_module, projection, absorbed, serialized,
            seed_i, seed_j, alpha, config, ref_batch, shell_model,
        )
        result["alpha"] = alpha
        curve.append(result)
        print(f"    α={alpha:.1f}  interp_ppl={result['interp_ppl']:.2f}  "
              f"ensemble_ppl={result['ensemble_ppl']:.2f}  "
              f"ratio={result['ratio']:.4f}")
    return curve


@torch.no_grad()
def compute_secondary_metrics(
    canon_module: CanonicalizationModule,
    projection: BiasProjection,
    absorbed: dict[int, torch.Tensor],
    serialized: list[torch.Tensor],
    config: BertConfig,
    ref_batch: dict,
    shell_model: BertForMaskedLM,
    seeds: list[int],
) -> dict:
    """Compute permutation sharpness and P stability."""
    canon_module.eval()
    vocab_size = config.vocab_size

    Ps = {}
    entropies = {}
    for seed in seeds:
        a = absorbed[seed].to(DEVICE)
        E = a[:vocab_size].unsqueeze(0)
        _, P = canon_module(E)
        P = P.squeeze(0)
        Ps[seed] = P.cpu()
        entropies[seed] = canon_module.row_entropy(P.unsqueeze(0)).item()

    # P stability: pairwise Frobenius distance
    pairs = list(combinations(seeds, 2))
    p_dists = {}
    for si, sj in pairs:
        dist = (Ps[si] - Ps[sj]).norm(p="fro").item()
        p_dists[f"{si}_{sj}"] = dist

    # Per-layer activation MSE on a sample validation pair
    sample_pair = pairs[0] if pairs else None
    per_layer_mse = None
    if sample_pair:
        si, sj = sample_pair
        alpha = 0.5
        a_i = absorbed[si].to(DEVICE)
        a_j = absorbed[sj].to(DEVICE)

        E_i = a_i[:vocab_size].unsqueeze(0)
        E_j = a_j[:vocab_size].unsqueeze(0)
        _, P_i = canon_module(E_i)
        _, P_j = canon_module(E_j)

        P_i_mat = P_i.squeeze(0)
        P_j_mat = P_j.squeeze(0)
        canon_i = a_i @ P_i_mat
        canon_j = a_j @ P_j_mat

        canon_interp = 0.5 * canon_i + 0.5 * canon_j
        P_i_inv, _ = invert_soft_permutation_with_metrics(P_i_mat, acted_matrix=a_i)
        interp_768 = canon_interp.float() @ P_i_inv
        interp_769 = restore_bias_rows_only(projection, interp_768, config, DEVICE)
        params_interp = deserialize(interp_769, config)

        A_interp = get_residual_stream_states(shell_model, params_interp, ref_batch, DEVICE)

        # Endpoint activations
        params_i = deserialize(serialized[si], config)
        params_j = deserialize(serialized[sj], config)
        A_i = get_residual_stream_states(shell_model, params_i, ref_batch, DEVICE)
        A_j = get_residual_stream_states(shell_model, params_j, ref_batch, DEVICE)
        A_target = [0.5 * ai + 0.5 * aj for ai, aj in zip(A_i, A_j)]

        _, per_layer_mse = compute_activation_mse(A_interp, A_target)

    return {
        "entropies": entropies,
        "p_frobenius_distances": p_dists,
        "per_layer_activation_mse": per_layer_mse,
        "sample_pair": list(sample_pair) if sample_pair else None,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("CANONICALIZATION EXPERIMENT V1")
    print("=" * 70)

    # Load data
    print("\n▸ Loading serialized MultiBERTs...")
    serialized, config = load_and_serialize_all()
    vocab_size = config.vocab_size
    d_model = config.hidden_size

    # Load frozen bias absorption
    print("\n▸ Loading frozen bias absorption projection...")
    projection = load_frozen_projection()
    projection = projection.to(DEVICE)

    wandb_run = init_wandb(config)

    # Tokenizer (used by training pool builder and evaluation)
    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")

    # Evaluation reference batch (larger, for final perplexity evaluation)
    print("\n▸ Preparing evaluation reference batch...")
    eval_ref_batch = get_reference_batch(tokenizer, num_samples=16, max_length=512)
    print(f"  Eval reference batch: {eval_ref_batch['input_ids'].shape}")

    # Shell model for functional_call
    print("\n▸ Creating shell BERT model...")
    shell_model = load_multibert_model(TRAIN_SEEDS[0]).eval().to(DEVICE)

    # Initialize canonicalization module
    print("\n▸ Initializing CanonicalizationModule...")
    canon_module = CanonicalizationModule(
        vocab_size=vocab_size,
        d_model=d_model,
        sinkhorn_iters=20,
        tau_init=0.5,
    )
    n_params = sum(p.numel() for p in canon_module.parameters())
    print(f"  Trainable parameters: {n_params:,}")

    # ── Phase 1: Training ────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("PHASE 1: TRAINING")
    print("=" * 70)

    canon_module, training_log = train(
        canon_module, projection, serialized, config, tokenizer, shell_model, wandb_run,
    )

    # Save training log
    log_path = RESULTS_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(make_serializable(training_log), f, indent=2)
    print(f"\n  Training log saved to {log_path}")

    # ── Phase 2: Evaluation ──────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("PHASE 2: EVALUATION")
    print("=" * 70)

    # Load best checkpoint
    best_path = RESULTS_DIR / "canonicalization_module.pt"
    if best_path.exists():
        canon_module.load_state_dict(
            torch.load(best_path, map_location="cpu", weights_only=True)
        )
    canon_module = canon_module.to(DEVICE).eval()

    # Pre-encode all models through frozen W_in
    absorbed = {}
    for seed in TRAIN_SEEDS + TEST_SEEDS:
        with torch.no_grad():
            absorbed[seed] = absorb_bias_rows_only(
                projection, serialized[seed], config, DEVICE,
            ).cpu()

    val_pairs = list(combinations(TEST_SEEDS, 2))

    # Primary metric: perplexity ratio at α=0.5
    print("\n▸ Evaluating validation pairs (α=0.5)...")
    ppl_results = {}
    for seed_i, seed_j in val_pairs:
        print(f"  Pair ({seed_i}, {seed_j}):")
        result = evaluate_pair_perplexity(
            canon_module, projection, absorbed, serialized,
            seed_i, seed_j, 0.5, config, eval_ref_batch, shell_model,
        )
        ppl_results[f"{seed_i}_{seed_j}"] = result
        print(f"    interp_ppl={result['interp_ppl']:.2f}  "
              f"ensemble_ppl={result['ensemble_ppl']:.2f}  "
              f"ratio={result['ratio']:.4f}")

    mean_ratio = sum(r["ratio"] for r in ppl_results.values()) / len(ppl_results)
    print(f"\n  Mean perplexity ratio: {mean_ratio:.4f}")
    wandb_log(
        wandb_run,
        {
            "eval/mean_perplexity_ratio": mean_ratio,
            "eval/mean_ensemble_ppl": sum(r["ensemble_ppl"] for r in ppl_results.values()) / len(ppl_results),
            "eval/mean_interp_ppl": sum(r["interp_ppl"] for r in ppl_results.values()) / len(ppl_results),
        },
    )

    # Naive interpolation baseline
    print("\n▸ Evaluating naive interpolation baseline...")
    naive_results = {}
    for seed_i, seed_j in val_pairs:
        ppl = evaluate_naive_interpolation(
            projection, serialized, seed_i, seed_j, 0.5, config, eval_ref_batch,
        )
        naive_results[f"{seed_i}_{seed_j}"] = ppl
        print(f"  Pair ({seed_i}, {seed_j}): naive_ppl={ppl:.2f}")

    mean_naive = sum(naive_results.values()) / len(naive_results)
    print(f"  Mean naive PPL: {mean_naive:.2f}")
    wandb_log(wandb_run, {"eval/mean_naive_ppl": mean_naive})

    # Secondary metrics
    print("\n▸ Computing secondary metrics...")
    secondary = compute_secondary_metrics(
        canon_module, projection, absorbed, serialized,
        config, eval_ref_batch, shell_model, TEST_SEEDS,
    )

    # Interpolation curves (on 2 representative pairs)
    print("\n▸ Computing interpolation curves...")
    curve_pairs = val_pairs[:2]
    interpolation_curves = {}
    for seed_i, seed_j in curve_pairs:
        print(f"  Pair ({seed_i}, {seed_j}):")
        curve = evaluate_interpolation_curve(
            canon_module, projection, absorbed, serialized,
            seed_i, seed_j, config, eval_ref_batch, shell_model,
        )
        interpolation_curves[f"{seed_i}_{seed_j}"] = curve

    # ── Save results ─────────────────────────────────────────────────────

    results = {
        "train_seeds": TRAIN_SEEDS,
        "test_seeds": TEST_SEEDS,
        "mean_perplexity_ratio": mean_ratio,
        "perplexity_results": ppl_results,
        "naive_interpolation": naive_results,
        "mean_naive_ppl": mean_naive,
        "secondary_metrics": secondary,
        "elapsed_seconds": time.time() - t0,
    }

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Results saved to {results_path}")

    curves_path = RESULTS_DIR / "interpolation_curves.json"
    with open(curves_path, "w") as f:
        json.dump(make_serializable(interpolation_curves), f, indent=2)
    print(f"  Interpolation curves saved to {curves_path}")

    if wandb_run is not None:
        wandb_run.summary["mean_perplexity_ratio"] = mean_ratio
        wandb_run.summary["mean_naive_ppl"] = mean_naive
        wandb_run.summary["final_tau"] = canon_module.tau.item()
        wandb_run.finish()

    # ── Summary ──────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Mean perplexity ratio (canon-interp / ensemble): {mean_ratio:.4f}")
    print(f"  Mean naive interpolation PPL:                     {mean_naive:.2f}")
    print(f"  Mean ensemble PPL:                                "
          f"{sum(r['ensemble_ppl'] for r in ppl_results.values()) / len(ppl_results):.2f}")
    print(f"  Mean canon-interp PPL:                            "
          f"{sum(r['interp_ppl'] for r in ppl_results.values()) / len(ppl_results):.2f}")
    print(f"  τ (final):                                        "
          f"{canon_module.tau.item():.4f}")
    print(f"  Total elapsed:                                    "
          f"{time.time() - t0:.1f}s")

    # Success criteria check
    if mean_ratio < 1.2:
        print("\n  ✓ EXCELLENT: ratio < 1.2")
    elif mean_ratio < 1.5:
        print("\n  ✓ GREAT: ratio < 1.5")
    elif mean_ratio < 2.0:
        print("\n  ✓ GOOD: ratio < 2.0")
    elif mean_ratio < float("inf"):
        print("\n  ✓ MINIMUM: finite perplexity")
    else:
        print("\n  ✗ FAILED: non-finite perplexity")


if __name__ == "__main__":
    main()
