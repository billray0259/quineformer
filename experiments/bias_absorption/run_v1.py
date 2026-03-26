# %%
"""Bias Absorption Validation Experiment v1.

Validates that the learned W_in/W_out linear projection compresses
(d_model+1)-dim bias-absorbed vectors to d_model with negligible
information loss, and that per-vector bias absorption preserves neuron
permutation symmetry while chunked-bias serialization breaks it.

See experiments/bias_absorption/Experiment_v1.md for full specification.
"""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from quineformer.bias_absorption import (
    BIAS_CARRYING_TYPES,
    BiasProjection,
    apply_head_permutation,
    apply_neuron_permutation,
    apply_projection_to_bias_rows,
    assemble_reconstructed_model,
    bias_carrying_mask,
    compute_bias_accuracy,
    compute_mlm_perplexity,
    compute_reconstruction_errors,
    load_multibert_model,
    reconstruct_model,
    reconstruction_mse_in_batches,
    train_projection,
    zero_bias_dimension,
)
from quineformer.serialization import serialize, deserialize, vector_component_labels

# ── Configuration ────────────────────────────────────────────────────────────

SERIALIZED_CACHE = "data/multiberts/serialized"
NUM_SEEDS = 25
TRAIN_SEEDS = list(range(20))
TEST_SEEDS = list(range(20, 25))

# Training hyperparameters
LR = 2e-4
BATCH_SIZE = 8192
MAX_EPOCHS = 5
CONVERGENCE_TOL = 1e-8  # stop when relative loss change < this
LOG_EVERY = 1

# MLM evaluation
MLM_MAX_LENGTH = 512
MLM_NUM_SAMPLES = 16  # number of 512-token sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_v1"


# ── Phase 1: Data pipeline ──────────────────────────────────────────────────

# %%
def load_and_serialize_all():
    """Load all 25 MultiBERTs and serialize to (N, d+1) matrices.

    Caches serialized tensors to disk to avoid re-loading on subsequent runs.
    Returns list of 25 tensors and the BertConfig.
    """
    cache = Path(SERIALIZED_CACHE)
    cache.mkdir(parents=True, exist_ok=True)

    serialized = []
    config = None

    for seed in range(NUM_SEEDS):
        cache_file = cache / f"seed_{seed}.pt"
        if cache_file.exists():
            data = torch.load(cache_file, weights_only=True)
            print(f"  seed {seed:2d}: loaded from cache  shape={tuple(data.shape)}")
        else:
            model_id = f"google/multiberts-seed_{seed}"
            m = load_multibert_model(seed)
            with torch.no_grad():
                data = serialize(m).clone()
            torch.save(data, cache_file)
            if config is None:
                config = m.bert.config
            del m
            print(f"  seed {seed:2d}: serialized & cached  shape={tuple(data.shape)}")

        serialized.append(data)

    if config is None:
        # Load config from one model if all were cached
        m = load_multibert_model(0)
        config = m.bert.config
        del m

    return serialized, config


def get_reference_batch(tokenizer, num_samples=MLM_NUM_SAMPLES,
                        max_length=MLM_MAX_LENGTH):
    """Prepare a reference text batch for MLM perplexity evaluation.

    Uses wikitext-103-raw-v1. Returns dict with input_ids, attention_mask,
    and labels (with 15% random masking).
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    # Concatenate text, filter empties
    text = "\n".join(t for t in ds["text"] if t.strip())

    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        padding=False,
    )

    # Tokenize enough text to get num_samples sequences
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    sequences = []
    for i in range(0, len(all_ids) - max_length + 1, max_length):
        if len(sequences) >= num_samples:
            break
        ids = [tokenizer.cls_token_id] + all_ids[i:i + max_length - 2] + [tokenizer.sep_token_id]
        sequences.append(ids)

    input_ids = torch.tensor(sequences, dtype=torch.long)  # (num_samples, max_length)
    attention_mask = torch.ones_like(input_ids)

    # Create MLM labels: mask 15% of tokens (excluding [CLS] and [SEP])
    labels = input_ids.clone()
    mask_prob = torch.rand_like(input_ids, dtype=torch.float)
    # Don't mask special tokens (first and last positions)
    mask_prob[:, 0] = 1.0
    mask_prob[:, -1] = 1.0
    mask_positions = mask_prob < 0.15
    # Replace masked positions with [MASK] token in input
    masked_input = input_ids.clone()
    masked_input[mask_positions] = tokenizer.mask_token_id
    # Set non-masked positions to -100 in labels (ignored by loss)
    labels[~mask_positions] = -100

    return {
        "input_ids": masked_input,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ── Phase 2: Ablation models ────────────────────────────────────────────────

# %%
class NonlinearBiasProjection(nn.Module):
    """Ablation 3: shallow MLP with GELU activation."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model, bias=True),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model + 1, bias=True),
        )

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        return self.decode(self.encode(x))


# ── Phase 3: Evaluation metrics ─────────────────────────────────────────────

# %%
def evaluate_mlm_perplexity(projection: nn.Module, serialized_list: list[Tensor],
                            test_seeds: list[int], config: BertConfig,
                            ref_batch: dict, tag: str = "") -> dict:
    """Metric 3/4: MLM perplexity for reconstructed vs original models."""
    results = {}

    for i, seed in enumerate(test_seeds):
        data = serialized_list[seed]

        # Original model perplexity
        orig_model = load_multibert_model(seed).eval().to(DEVICE)
        orig_ppl = compute_mlm_perplexity(orig_model, ref_batch)
        del orig_model
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        # Reconstructed model perplexity
        recon_model = reconstruct_model(projection, data, config, seed, DEVICE).to(DEVICE)
        recon_ppl = compute_mlm_perplexity(recon_model, ref_batch)
        del recon_model
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        ratio = recon_ppl / orig_ppl
        results[f"seed_{seed}"] = {
            "original_ppl": orig_ppl,
            "reconstructed_ppl": recon_ppl,
            "ratio": ratio,
        }
        print(f"  [{tag}] seed {seed}: orig={orig_ppl:.3f}  recon={recon_ppl:.3f}  ratio={ratio:.4f}")

    return results


def evaluate_no_bias_baseline(serialized_list: list[Tensor], test_seeds: list[int],
                              config: BertConfig, ref_batch: dict) -> dict:
    """Metric 4: MLM perplexity with all biases zeroed (identity W_in baseline)."""
    d = config.hidden_size
    results = {}

    for seed in test_seeds:
        data = zero_bias_dimension(serialized_list[seed], d)

        params = deserialize(data, config)
        model = assemble_reconstructed_model(params, seed).to(DEVICE)
        ppl = compute_mlm_perplexity(model, ref_batch)
        del model
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        # Also get original
        orig_model = load_multibert_model(seed).eval().to(DEVICE)
        orig_ppl = compute_mlm_perplexity(orig_model, ref_batch)
        del orig_model
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        results[f"seed_{seed}"] = {
            "original_ppl": orig_ppl,
            "no_bias_ppl": ppl,
            "ratio": ppl / orig_ppl,
        }
        print(f"  [no-bias] seed {seed}: orig={orig_ppl:.3f}  no_bias={ppl:.3f}  ratio={ppl/orig_ppl:.4f}")

    return results


# ── Metric 5: Symmetry preservation ─────────────────────────────────────────

# %%
def test_symmetry_neuron(serialized_data: Tensor, config: BertConfig,
                         projection: nn.Module, layer_idx: int = 5) -> dict:
    """Metric 5a: verify MLP neuron permutation produces identical vector sets."""
    d = config.hidden_size
    d_ff = config.intermediate_size
    n_layers = config.num_hidden_layers

    # Deserialize original to state dict, apply neuron permutation
    params_orig = deserialize(serialized_data, config)
    perm = torch.randperm(d_ff)
    params_perm = apply_neuron_permutation(params_orig, layer_idx, perm)

    # Load permuted model, serialize
    model_perm = BertForMaskedLM(config)
    model_perm.bert.load_state_dict(params_perm)
    with torch.no_grad():
        data_perm = serialize(model_perm).clone()
    del model_perm

    # Extract MLP up vectors for the target layer
    labels = vector_component_labels(config)
    vocab = config.vocab_size
    max_pos = config.max_position_embeddings
    n_type = config.type_vocab_size
    global_count = vocab + max_pos + n_type + 2
    per_layer = d * 4 + d_ff * 2 + 6
    layer_start = global_count + layer_idx * per_layer
    # MLP up vectors start after Q(d) + K(d) + V(d) + O(d) + b_O(1) + LN(2) = 4d+3
    mlp_up_start = layer_start + 4 * d + 3
    mlp_up_end = mlp_up_start + d_ff

    orig_mlp = serialized_data[mlp_up_start:mlp_up_end]
    perm_mlp = data_perm[mlp_up_start:mlp_up_end]

    # Check (a): vectors are identical sets (same rows, possibly reordered)
    # Sort both by content and compare
    orig_sorted, _ = orig_mlp.sort(dim=0)
    perm_sorted, _ = perm_mlp.sort(dim=0)
    sets_identical = torch.allclose(orig_sorted, perm_sorted, atol=1e-6)

    # Check the permuted vectors are exactly the original vectors reordered
    reordered = orig_mlp[perm]
    reorder_exact = torch.allclose(reordered, perm_mlp, atol=1e-6)

    # Check (b): round-trip reconstructed models produce identical outputs
    projection = projection.eval()
    with torch.no_grad():
        recon_orig = apply_projection_to_bias_rows(projection, serialized_data, config, DEVICE)
        recon_perm = apply_projection_to_bias_rows(projection, data_perm, config, DEVICE)

    params_ro = deserialize(recon_orig, config)
    params_rp = deserialize(recon_perm, config)

    m_ro = BertForMaskedLM(config).eval()
    m_rp = BertForMaskedLM(config).eval()
    m_ro.bert.load_state_dict(params_ro)
    m_rp.bert.load_state_dict(params_rp)

    ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out_ro = m_ro(ids).logits
        out_rp = m_rp(ids).logits
    outputs_match = torch.allclose(out_ro, out_rp, atol=1e-4)

    return {
        "sets_identical": sets_identical,
        "reorder_exact": reorder_exact,
        "outputs_match": outputs_match,
        "max_set_diff": (orig_sorted - perm_sorted).abs().max().item(),
        "max_output_diff": (out_ro - out_rp).abs().max().item(),
    }


def test_symmetry_head(serialized_data: Tensor, config: BertConfig,
                       projection: nn.Module, layer_idx: int = 5) -> dict:
    """Metric 5c: verify attention head permutation produces identical vector sets."""
    d = config.hidden_size
    d_ff = config.intermediate_size
    n_heads = config.num_attention_heads
    d_head = d // n_heads

    params_orig = deserialize(serialized_data, config)
    perm = torch.randperm(n_heads)
    params_perm = apply_head_permutation(params_orig, layer_idx, perm, n_heads, d)

    model_perm = BertForMaskedLM(config)
    model_perm.bert.load_state_dict(params_perm)
    with torch.no_grad():
        data_perm = serialize(model_perm).clone()
    del model_perm

    # Extract Q vectors for the target layer
    vocab = config.vocab_size
    max_pos = config.max_position_embeddings
    n_type = config.type_vocab_size
    global_count = vocab + max_pos + n_type + 2
    per_layer = d * 4 + d_ff * 2 + 6
    layer_start = global_count + layer_idx * per_layer
    q_start = layer_start
    q_end = q_start + d

    orig_q = serialized_data[q_start:q_end]
    perm_q = data_perm[q_start:q_end]

    # Head permutation reorders blocks of d_head vectors
    orig_sorted, _ = orig_q.sort(dim=0)
    perm_sorted, _ = perm_q.sort(dim=0)
    sets_identical = torch.allclose(orig_sorted, perm_sorted, atol=1e-6)

    # Round-trip output comparison
    projection = projection.eval()
    with torch.no_grad():
        recon_orig = apply_projection_to_bias_rows(projection, serialized_data, config, DEVICE)
        recon_perm = apply_projection_to_bias_rows(projection, data_perm, config, DEVICE)

    params_ro = deserialize(recon_orig, config)
    params_rp = deserialize(recon_perm, config)

    m_ro = BertForMaskedLM(config).eval()
    m_rp = BertForMaskedLM(config).eval()
    m_ro.bert.load_state_dict(params_ro)
    m_rp.bert.load_state_dict(params_rp)

    ids = torch.randint(0, config.vocab_size, (1, 16))
    with torch.no_grad():
        out_ro = m_ro(ids).logits
        out_rp = m_rp(ids).logits
    outputs_match = torch.allclose(out_ro, out_rp, atol=1e-4)

    return {
        "sets_identical": sets_identical,
        "outputs_match": outputs_match,
        "max_set_diff": (orig_sorted - perm_sorted).abs().max().item(),
        "max_output_diff": (out_ro - out_rp).abs().max().item(),
    }


def test_chunked_bias_symmetry_break(config: BertConfig,
                                     serialized_data: Tensor,
                                     layer_idx: int = 5) -> dict:
    """Baseline 1 symmetry test: chunked bias vectors change content under permutation.

    Chunks the d_ff-dim b₁ into d_ff/d_model vectors of d_model dimensions each.
    Applies MLP neuron permutation and shows chunk contents change.
    """
    d = config.hidden_size
    d_ff = config.intermediate_size

    params = deserialize(serialized_data, config)
    pre = f"encoder.layer.{layer_idx}"
    b1 = params[f"{pre}.intermediate.dense.bias"]  # (d_ff,)

    # Chunk b₁ into d_ff/d_model vectors of d_model
    n_chunks = d_ff // d
    chunks_orig = b1.reshape(n_chunks, d)  # (4, 768) for BERT-base

    # Apply neuron permutation
    perm = torch.randperm(d_ff)
    b1_perm = b1[perm]
    chunks_perm = b1_perm.reshape(n_chunks, d)

    # Check if chunk contents changed (they should!)
    chunks_match = torch.allclose(chunks_orig, chunks_perm, atol=1e-6)
    # Sort and compare — even as sets, chunks are different because permutation
    # shuffles entries across chunk boundaries
    orig_sorted, _ = chunks_orig.sort(dim=0)
    perm_sorted, _ = chunks_perm.sort(dim=0)
    sets_match = torch.allclose(orig_sorted, perm_sorted, atol=1e-6)

    return {
        "chunks_identical": chunks_match,
        "chunks_sets_identical": sets_match,
        "symmetry_broken": not chunks_match,
        "max_chunk_diff": (chunks_orig - chunks_perm).abs().max().item(),
    }


# ── Phase 4: Baselines ──────────────────────────────────────────────────────

# %%
def compute_pca_from_covariance(train_data: Tensor, d_model: int,
                                chunk_size: int = 50_000
                                ) -> tuple[Tensor, Tensor, Tensor]:
    """Compute PCA via chunked covariance eigendecomposition.

    Memory-efficient: never materializes a full centered copy of train_data.
    Works on the (d+1)×(d+1) covariance matrix instead of SVD on (N, d+1).

    Returns (mean, eigenvectors V of shape (d+1, d+1), eigenvalues descending).
    """
    dim = train_data.shape[1]  # d_model + 1
    n = train_data.shape[0]
    mean = train_data.mean(dim=0)  # (dim,)

    # Accumulate X_centered^T @ X_centered in chunks
    cov = torch.zeros(dim, dim, dtype=train_data.dtype)
    for start in range(0, n, chunk_size):
        chunk = train_data[start:start + chunk_size] - mean
        cov.addmm_(chunk.T, chunk)
        del chunk
    cov /= n

    # Eigendecompose the small (769×769) covariance matrix
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    # eigh returns ascending order; flip to descending
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    return mean, eigenvectors, eigenvalues


def pca_baseline(test_data: Tensor, d_model: int,
                 pca_mean: Tensor, pca_vecs: Tensor,
                 pca_eigenvalues: Tensor) -> tuple[Tensor, dict]:
    """Baseline 2: PCA projection (top d_model components).

    Uses pre-computed PCA (from compute_pca_from_covariance).
    Returns reconstructed test data and error metrics.
    """
    V = pca_vecs[:, :d_model]  # (d+1, d_model)

    test_centered = test_data - pca_mean
    encoded = test_centered @ V      # (N, d_model)
    decoded = encoded @ V.T + pca_mean  # (N, d+1)

    mse = (test_data - decoded).pow(2).mean().item()
    total_var = pca_eigenvalues.sum().item()
    top_k_var = pca_eigenvalues[:d_model].sum().item()
    var_explained = top_k_var / (total_var + 1e-12)
    var_discarded = 1.0 - var_explained

    S = pca_eigenvalues.clamp(min=0).sqrt()  # singular-value scale

    return decoded, {
        "mse": mse,
        "variance_explained": var_explained,
        "variance_discarded": var_discarded,
        "singular_values": S[:d_model + 5].tolist(),
    }


def random_projection_baseline(train_data: Tensor, test_data: Tensor,
                                d_model: int) -> tuple[Tensor, dict]:
    """Baseline 3: random orthogonal projection.

    Projects (d_model+1) → d_model via random orthogonal rows.
    Reconstructs via pseudoinverse.
    """
    d_plus_1 = d_model + 1
    # Generate random orthogonal matrix via QR decomposition
    A = torch.randn(d_plus_1, d_plus_1)
    Q, _ = torch.linalg.qr(A)
    # Take first d_model rows as projection
    P = Q[:d_model, :]  # (d_model, d+1)

    # Project and reconstruct via pseudoinverse (P^T for orthogonal P)
    encoded = test_data @ P.T  # (N, d_model)
    decoded = encoded @ P      # (N, d+1)

    mse = (test_data - decoded).pow(2).mean().item()

    return decoded, {"mse": mse}


# ── Phase 5: Ablations ──────────────────────────────────────────────────────

# %%
def ablation_per_component(train_data: Tensor, labels: list[str],
                           d_model: int) -> dict[str, nn.Module]:
    """Ablation 1: train separate W_in/W_out per bias-carrying component type.

    Only trains projections for Q, K, V, and mlp_up — the types whose
    (d_model+1)-th dim carries real bias information.  Zero-padded types
    (embeddings, O, mlp_down, LayerNorms, standalone biases) are skipped
    because the identity initialization already reconstructs them perfectly.
    """
    models = {}
    for ctype in sorted(BIAS_CARRYING_TYPES):
        mask = [i for i, l in enumerate(labels) if l == ctype]
        subset = train_data[mask]
        m = train_projection(subset, d_model, tag=f"per-type:{ctype}")
        models[ctype] = m
    return models


def ablation_bias_scaling(train_data: Tensor, d_model: int,
                          scales: list[float] = None) -> dict:
    """Ablation 2: sweep bias magnitude scaling factor c."""
    if scales is None:
        scales = [0.1, 0.5, 1.0, 2.0, 10.0]

    results = {}
    for c in scales:
        m = train_projection(train_data, d_model, tag=f"scale:{c}", bias_scale=c)

        # Evaluate against the scaled representation, but only scale the
        # minibatch bias column to avoid cloning the full training tensor.
        mse, bias_mse = reconstruction_mse_in_batches(m, train_data, bias_scale=c)

        results[f"c={c}"] = {"total_mse": mse, "bias_mse": bias_mse}
        print(f"  [scale c={c}] total_mse={mse:.8f}  bias_mse={bias_mse:.8f}")

        del m
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ── Phase 6: Analysis ────────────────────────────────────────────────────────

# %%
def information_theoretic_analysis(pca_eigenvalues: Tensor,
                                   d_model: int) -> dict:
    """Report variance in the discarded 769th component using pre-computed PCA."""
    total_var = pca_eigenvalues.sum().item()
    top_k_var = pca_eigenvalues[:d_model].sum().item()
    discarded_var = total_var - top_k_var
    S = pca_eigenvalues.clamp(min=0).sqrt()

    return {
        "total_variance": total_var,
        "top_k_variance": top_k_var,
        "discarded_variance": discarded_var,
        "fraction_discarded": discarded_var / (total_var + 1e-12),
        "top_5_singular_values": S[:5].tolist(),
        "last_5_singular_values": S[-5:].tolist() if len(S) >= 5 else S.tolist(),
    }


def bias_contribution_analysis(data: Tensor, labels: list[str],
                               d_model: int) -> dict:
    """Compute ||b|| / ||w|| ratio per component type for bias-carrying vectors."""
    bias_types = {"Q", "K", "V", "mlp_up"}
    results = {}

    for ctype in sorted(bias_types):
        mask = [i for i, l in enumerate(labels) if l == ctype]
        if not mask:
            continue
        subset = data[mask]
        w = subset[:, :d_model]
        b = subset[:, d_model]
        w_norm = w.pow(2).mean().sqrt().item()
        b_norm = b.pow(2).mean().sqrt().item()
        results[ctype] = {
            "weight_rms": w_norm,
            "bias_rms": b_norm,
            "ratio": b_norm / (w_norm + 1e-12),
        }

    return results


def layer_wise_sensitivity(projection: nn.Module, serialized_data: Tensor,
                           config: BertConfig, ref_batch: dict) -> dict:
    """Apply round-trip to one layer at a time, measure per-layer perplexity impact."""
    d = config.hidden_size
    d_ff = config.intermediate_size
    n_layers = config.num_hidden_layers
    vocab = config.vocab_size
    max_pos = config.max_position_embeddings
    n_type = config.type_vocab_size
    global_count = vocab + max_pos + n_type + 2
    per_layer = d * 4 + d_ff * 2 + 6

    # Get original perplexity
    orig_data = serialized_data.clone()
    params_orig = deserialize(orig_data, config)
    model_orig = BertForMaskedLM(config).eval().to(DEVICE)
    model_orig.bert.load_state_dict(params_orig)
    orig_ppl = compute_mlm_perplexity(model_orig, ref_batch)
    del model_orig
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    results = {"original_ppl": orig_ppl}
    projection = projection.eval()

    for layer_i in range(n_layers):
        # Start with original data, only round-trip this layer's vectors
        data_mod = serialized_data.clone()
        start = global_count + layer_i * per_layer
        end = start + per_layer
        layer_vectors = data_mod[start:end].to(DEVICE)
        with torch.no_grad():
            data_mod[start:end] = projection(layer_vectors).cpu()

        params = deserialize(data_mod, config)
        model = BertForMaskedLM(config).eval().to(DEVICE)
        model.bert.load_state_dict(params)
        ppl = compute_mlm_perplexity(model, ref_batch)
        del model
        torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

        ratio = ppl / orig_ppl
        results[f"layer_{layer_i}"] = {"ppl": ppl, "ratio": ratio}
        print(f"  [layer-sensitivity] layer {layer_i:2d}: ppl={ppl:.3f}  ratio={ratio:.4f}")

    return results


# ── Main entry point ─────────────────────────────────────────────────────────

# %%
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()

    print("=" * 70)
    print("BIAS ABSORPTION VALIDATION EXPERIMENT v1")
    print("=" * 70)

    # ── Phase 1: Load data ───────────────────────────────────────────────────
    print("\n▸ Phase 1: Loading and serializing MultiBERTs...")
    serialized, config = load_and_serialize_all()
    d_model = config.hidden_size
    labels = vector_component_labels(config)
    assert len(labels) == serialized[0].shape[0], \
        f"Label count {len(labels)} != vector count {serialized[0].shape[0]}"

    print(f"\n  Config: d_model={d_model}, d_ff={config.intermediate_size}, "
          f"n_layers={config.num_hidden_layers}, vocab={config.vocab_size}")
    print(f"  Vectors per model: {serialized[0].shape[0]:,}")
    print(f"  Vector dimension: {serialized[0].shape[1]}")

    bias_mask = bias_carrying_mask(config)

    # Prepare training data: concatenate only bias-carrying vectors.
    train_data = torch.cat([serialized[s][bias_mask] for s in TRAIN_SEEDS], dim=0)
    print(
        f"  Training data: {train_data.shape[0]:,} bias-carrying vectors "
        f"from {len(TRAIN_SEEDS)} seeds"
    )

    # Prepare reference batch for MLM evaluation
    print("\n▸ Preparing reference text batch for MLM evaluation...")
    tokenizer = BertTokenizer.from_pretrained(
        "google/multiberts-seed_0"
    )
    ref_batch = get_reference_batch(tokenizer)
    print(f"  Reference batch: {ref_batch['input_ids'].shape}")

    # ── Phase 2: Train shared projection ─────────────────────────────────────
    print("\n▸ Phase 2: Training shared linear projection...")
    projection = train_projection(train_data, d_model, tag="shared")
    torch.save(projection.state_dict(), RESULTS_DIR / "projection_shared.pt")

    # ── Phase 3: Evaluation ──────────────────────────────────────────────────
    print("\n▸ Phase 3: Evaluation metrics on held-out seeds...")

    # Metric 1: Per-component reconstruction error
    print("\n  ── Metric 1: Per-component reconstruction error ──")
    for seed in TEST_SEEDS:
        print(f"\n  Seed {seed}:")
        errs = compute_reconstruction_errors(projection, serialized[seed], labels, config)
        results[f"metric1_seed{seed}"] = errs
        for ctype, vals in sorted(errs.items()):
            print(f"    {ctype:16s}  mse={vals['mse']:.2e}  "
                  f"rel={vals['relative_mse']:.2e}  n={vals['count']}")

    # Metric 2: Bias reconstruction accuracy
    print("\n  ── Metric 2: Bias reconstruction accuracy ──")
    for seed in TEST_SEEDS:
        print(f"\n  Seed {seed}:")
        bias_acc = compute_bias_accuracy(projection, serialized[seed], labels, config)
        results[f"metric2_seed{seed}"] = bias_acc
        for ctype, vals in sorted(bias_acc.items()):
            print(f"    {ctype:8s}  corr={vals['correlation']:.6f}  mae={vals['mae']:.2e}")

    # Metric 3: MLM perplexity
    print("\n  ── Metric 3: MLM perplexity of reconstructed models ──")
    results["metric3"] = evaluate_mlm_perplexity(
        projection, serialized, TEST_SEEDS, config, ref_batch, tag="learned"
    )

    # Metric 4: No-bias baseline
    print("\n  ── Metric 4: No-bias baseline ──")
    results["metric4"] = evaluate_no_bias_baseline(serialized, TEST_SEEDS, config, ref_batch)

    # Metric 5: Symmetry preservation
    print("\n  ── Metric 5: Symmetry preservation ──")
    for seed in TEST_SEEDS:
        print(f"\n  Seed {seed}:")
        layer_idx = seed % config.num_hidden_layers  # vary layers across seeds

        neuron_result = test_symmetry_neuron(serialized[seed], config, projection, layer_idx)
        results[f"metric5_neuron_seed{seed}"] = neuron_result
        print(f"    Neuron perm: sets_identical={neuron_result['sets_identical']}  "
              f"outputs_match={neuron_result['outputs_match']}  "
              f"max_diff={neuron_result['max_set_diff']:.2e}")

        head_result = test_symmetry_head(serialized[seed], config, projection, layer_idx)
        results[f"metric5_head_seed{seed}"] = head_result
        print(f"    Head perm:   sets_identical={head_result['sets_identical']}  "
              f"outputs_match={head_result['outputs_match']}  "
              f"max_diff={head_result['max_set_diff']:.2e}")

        chunked_result = test_chunked_bias_symmetry_break(config, serialized[seed], layer_idx)
        results[f"metric5_chunked_seed{seed}"] = chunked_result
        print(f"    Chunked bias: symmetry_broken={chunked_result['symmetry_broken']}  "
              f"max_chunk_diff={chunked_result['max_chunk_diff']:.2e}")

    # ── Phase 4: Baselines ───────────────────────────────────────────────────
    print("\n▸ Phase 4: Baselines...")

    # Baseline 2: PCA — compute eigenvectors once from covariance matrix
    print("\n  ── Baseline 2: PCA projection ──")
    print("    Computing PCA via covariance eigendecomposition...")
    pca_mean, pca_vecs, pca_eigenvalues = compute_pca_from_covariance(
        train_data, d_model
    )
    for seed in TEST_SEEDS:
        _, pca_info = pca_baseline(
            serialized[seed], d_model, pca_mean, pca_vecs, pca_eigenvalues
        )
        results[f"baseline_pca_seed{seed}"] = pca_info
        print(f"    seed {seed}: mse={pca_info['mse']:.8f}  "
              f"var_discarded={pca_info['variance_discarded']:.2e}")

    # Baseline 3: Random projection
    print("\n  ── Baseline 3: Random orthogonal projection ──")
    for seed in TEST_SEEDS:
        _, rand_info = random_projection_baseline(train_data, serialized[seed], d_model)
        results[f"baseline_random_seed{seed}"] = rand_info
        print(f"    seed {seed}: mse={rand_info['mse']:.8f}")

    # PCA and random baseline MLM perplexity (using first test seed as representative)
    print("\n  ── Baseline MLM perplexity (seed 20) ──")
    seed = TEST_SEEDS[0]

    pca_recon, _ = pca_baseline(
        serialized[seed], d_model, pca_mean, pca_vecs, pca_eigenvalues
    )
    pca_params = deserialize(pca_recon, config)
    pca_model = BertForMaskedLM(config).eval().to(DEVICE)
    pca_model.bert.load_state_dict(pca_params)
    pca_ppl = compute_mlm_perplexity(pca_model, ref_batch)
    del pca_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    results["baseline_pca_ppl"] = pca_ppl
    print(f"    PCA perplexity: {pca_ppl:.3f}")

    rand_recon, _ = random_projection_baseline(train_data, serialized[seed], d_model)
    rand_params = deserialize(rand_recon, config)
    rand_model = BertForMaskedLM(config).eval().to(DEVICE)
    rand_model.bert.load_state_dict(rand_params)
    rand_ppl = compute_mlm_perplexity(rand_model, ref_batch)
    del rand_model
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    results["baseline_random_ppl"] = rand_ppl
    print(f"    Random projection perplexity: {rand_ppl:.3f}")

    # ── Phase 5: Ablations ───────────────────────────────────────────────────
    print("\n▸ Phase 5: Ablations...")

    # Ablation 1: Per-component projections
    print("\n  ── Ablation 1: Per-component-type projections ──")
    per_comp_models = ablation_per_component(train_data, labels, d_model)
    # Evaluate per-component models on test data
    for seed in TEST_SEEDS[:1]:  # Representative seed only
        print(f"\n  Seed {seed} per-component reconstruction errors:")
        for ctype, m in per_comp_models.items():
            mask = [i for i, l in enumerate(labels) if l == ctype]
            subset = serialized[seed][mask]
            m = m.to(DEVICE).eval()
            with torch.no_grad():
                recon = m(subset.to(DEVICE)).cpu()
                mse = (subset - recon).pow(2).mean().item()
                norm_sq = subset.pow(2).mean().item()
            m = m.cpu()
            rel = mse / (norm_sq + 1e-12)
            results[f"ablation1_{ctype}_seed{seed}"] = {"mse": mse, "relative_mse": rel}
            print(f"    {ctype:16s}  mse={mse:.2e}  rel={rel:.2e}")

    # Ablation 2: Bias magnitude scaling
    print("\n  ── Ablation 2: Bias magnitude scaling ──")
    results["ablation2"] = ablation_bias_scaling(train_data, d_model)

    # Ablation 3: Nonlinear projection
    print("\n  ── Ablation 3: Nonlinear projection ──")
    nonlinear = train_projection(
        train_data, d_model, model=NonlinearBiasProjection(d_model), tag="nonlinear"
    )
    for seed in TEST_SEEDS[:1]:
        errs = compute_reconstruction_errors(nonlinear, serialized[seed], labels, config)
        results[f"ablation3_seed{seed}"] = errs
        total_mse = sum(v["mse"] * v["count"] for v in errs.values()) / sum(v["count"] for v in errs.values())
        print(f"    seed {seed} total_mse={total_mse:.8f}")

    # Ablation 4: Bias-only vectors (count and symmetry analysis)
    print("\n  ── Ablation 4: Bias-only vectors ──")
    n_layers = config.num_hidden_layers
    d_ff = config.intermediate_size
    bias_carrying = n_layers * (3 * d_model + d_ff)
    extra_tokens = bias_carrying  # each bias scalar becomes its own d_model token
    orig_count = serialized[0].shape[0]
    # Under bias-only: original d_model vectors (no +1 dim) + separate bias tokens
    # Weight-only vectors: same count as now, but d_model dim (not d_model+1)
    # Bias tokens: one per bias-carrying vector
    results["ablation4"] = {
        "original_vector_count": orig_count,
        "bias_only_vector_count": orig_count + bias_carrying,
        "sequence_length_increase": bias_carrying,
        "percent_increase": 100.0 * bias_carrying / orig_count,
        "symmetry_preserved": True,  # each neuron's bias is its own token
    }
    print(f"    Original vectors: {orig_count:,}")
    print(f"    With separate bias tokens: {orig_count + bias_carrying:,}")
    print(f"    Increase: {bias_carrying:,} ({100.0 * bias_carrying / orig_count:.1f}%)")
    print(f"    Symmetry preserved: True (each bias is its own token)")

    # ── Phase 6: Analysis ────────────────────────────────────────────────────
    print("\n▸ Phase 6: Analysis...")

    # Information-theoretic bound (reuse PCA eigenvalues from baseline 2)
    print("\n  ── Information-theoretic bound ──")
    info = information_theoretic_analysis(pca_eigenvalues, d_model)
    results["info_theoretic"] = info
    print(f"    Fraction of variance in discarded component: {info['fraction_discarded']:.2e}")

    # Bias contribution analysis
    print("\n  ── Bias contribution analysis ──")
    bias_contrib = bias_contribution_analysis(train_data, labels, d_model)
    results["bias_contribution"] = bias_contrib
    for ctype, vals in sorted(bias_contrib.items()):
        print(f"    {ctype:8s}  w_rms={vals['weight_rms']:.4f}  "
              f"b_rms={vals['bias_rms']:.4f}  ratio={vals['ratio']:.4f}")

    # Layer-wise sensitivity (one representative test seed)
    print("\n  ── Layer-wise sensitivity ──")
    results["layer_sensitivity"] = layer_wise_sensitivity(
        projection, serialized[TEST_SEEDS[0]], config, ref_batch
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE  ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")

    # Success criteria check
    print("\n▸ Success criteria:")
    if "metric3" in results:
        ratios = [v["ratio"] for v in results["metric3"].values()]
        mean_ratio = sum(ratios) / len(ratios)
        print(f"  MLM perplexity ratio (mean): {mean_ratio:.4f}  "
              f"{'PASS' if mean_ratio < 1.05 else 'FAIL'} (< 1.05)")

    if "metric2_seed20" in results:
        min_corr = min(v["correlation"] for v in results["metric2_seed20"].values())
        print(f"  Min bias correlation (seed 20): {min_corr:.4f}  "
              f"{'PASS' if min_corr > 0.95 else 'FAIL'} (> 0.95)")

    if "info_theoretic" in results:
        frac = results["info_theoretic"]["fraction_discarded"]
        print(f"  Variance discarded: {frac:.2e}  "
              f"{'PASS' if frac < 0.001 else '---'} (< 0.1%)")

    # Save results
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return obj
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int,)):
            return obj
        return str(obj)

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()

# %%
