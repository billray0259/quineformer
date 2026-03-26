# %%
"""Bias Absorption V2: End-to-End Functional Training.

Replaces V1's weight-space MSE objective with a functional KL divergence loss
that backpropagates through the full BERT forward pass. The projection learns
to minimize *functional* distortion rather than *geometric* reconstruction
error.

See experiments/bias_absorption/Experiment_v2.md for full specification.
"""

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from quineformer.bias_absorption import (
    apply_projection_to_bias_rows_with_grad,
    load_multibert_model,
    reconstruct_model,
)
from quineformer.serialization import deserialize, vector_component_labels

# Import shared code from V1 (DRY)
from run_v1 import (
    DEVICE,
    MLM_MAX_LENGTH,
    MLM_NUM_SAMPLES,
    NUM_SEEDS,
    SERIALIZED_CACHE,
    TEST_SEEDS,
    TRAIN_SEEDS,
    BiasProjection,
    compute_bias_accuracy,
    compute_mlm_perplexity,
    compute_reconstruction_errors,
    get_reference_batch,
    test_symmetry_head,
    test_symmetry_neuron,
)

# ── V2-specific configuration ───────────────────────────────────────────────

# Training hyperparameters
V2_LR = 1e-4
V2_MAX_EPOCHS = 100
V2_BATCHES_PER_SEED = 2        # minibatches per seed per epoch
V2_BATCH_SIZE = 4              # sequences per minibatch
V2_TRAIN_SEQ_LEN = 128        # shorter than V1's 512 for memory
V2_TRAIN_NUM_SEQUENCES = 256   # total training sequences from WikiText-103
V2_MASK_PROB = 0.15
V2_VAL_EVERY = 5              # validate every N epochs
V2_PATIENCE = 15              # early stopping patience (epochs)
V2_LOG_EVERY = 1

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results_v2"
LEGACY_RESULTS_DIR = SCRIPT_DIR / "experiments" / "bias_absorption" / "results_v2"
LEGACY_V1_PROJECTION_PATH = (
    SCRIPT_DIR / "experiments" / "bias_absorption" / "results_v1" / "projection_shared.pt"
)
V1_PROJECTION_PATH = SCRIPT_DIR / "results_v1" / "projection_shared.pt"

TIE_WEIGHTS = {
    "cls.predictions.decoder.weight": "bert.embeddings.word_embeddings.weight",
}


def resolve_existing_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def evaluate_mlm_perplexity_silent(
    projection: nn.Module,
    serialized_list,
    test_seeds: list[int],
    config: BertConfig,
    ref_batch: dict,
    tag: str = "",
) -> dict:
    """Evaluate original vs reconstructed perplexity without load spam."""
    results = {}

    for seed in test_seeds:
        data = serialized_list[seed]

        orig_model = load_multibert_model(seed).eval().to(DEVICE)
        orig_ppl = compute_mlm_perplexity(orig_model, ref_batch)
        del orig_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        recon_model = reconstruct_model(projection, data, config, seed, DEVICE).to(DEVICE)
        recon_ppl = compute_mlm_perplexity(recon_model, ref_batch)
        del recon_model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        ratio = recon_ppl / orig_ppl
        results[f"seed_{seed}"] = {
            "original_ppl": orig_ppl,
            "reconstructed_ppl": recon_ppl,
            "ratio": ratio,
        }
        print(
            f"  [{tag}] seed {seed}: orig={orig_ppl:.3f}  "
            f"recon={recon_ppl:.3f}  ratio={ratio:.4f}"
        )

    return results


# ── Phase 1: Training data pipeline ─────────────────────────────────────────

# %%
class LazySerializedList:
    """Load serialized tensors from disk on demand to avoid OOM.

    Behaves like a list but only keeps one tensor in memory at a time.
    """

    def __init__(self, cache_dir: str, num_seeds: int):
        self._cache_dir = Path(cache_dir)
        self._num_seeds = num_seeds
        # Preload shape from first file
        first = torch.load(self._cache_dir / "seed_0.pt", weights_only=True)
        self.shape_per_seed = first.shape
        del first

    def __len__(self):
        return self._num_seeds

    def __getitem__(self, seed: int) -> Tensor:
        return torch.load(
            self._cache_dir / f"seed_{seed}.pt", weights_only=True
        )


def get_training_batches(
    tokenizer: BertTokenizer,
    num_sequences: int = V2_TRAIN_NUM_SEQUENCES,
    seq_len: int = V2_TRAIN_SEQ_LEN,
    batch_size: int = V2_BATCH_SIZE,
    mask_prob: float = V2_MASK_PROB,
) -> list[dict[str, Tensor]]:
    """Prepare training minibatches from WikiText-103 train split.

    Stream-tokenizes dataset rows to avoid loading the entire text into
    memory. Returns a list of batch dicts on CPU.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Stream-tokenize: accumulate token ids row by row, stop early
    content_len = seq_len - 2
    tokens_needed = num_sequences * content_len + content_len  # small buffer
    all_ids: list[int] = []
    for row in ds:
        line = row["text"]
        if not line or not line.strip():
            continue
        all_ids.extend(tokenizer.encode(line, add_special_tokens=False))
        if len(all_ids) >= tokens_needed:
            break
    del ds

    # Build sequences of [CLS] + content + [SEP]
    sequences = []
    for i in range(0, len(all_ids) - content_len + 1, content_len):
        if len(sequences) >= num_sequences:
            break
        ids = ([tokenizer.cls_token_id]
               + all_ids[i : i + content_len]
               + [tokenizer.sep_token_id])
        sequences.append(ids)
    del all_ids

    input_ids = torch.tensor(sequences, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    # MLM masking
    labels = input_ids.clone()
    mask_rand = torch.rand_like(input_ids, dtype=torch.float)
    mask_rand[:, 0] = 1.0   # don't mask [CLS]
    mask_rand[:, -1] = 1.0  # don't mask [SEP]
    mask_positions = mask_rand < mask_prob
    masked_input = input_ids.clone()
    masked_input[mask_positions] = tokenizer.mask_token_id
    labels[~mask_positions] = -100

    # Split into minibatches
    batches = []
    for start in range(0, len(sequences), batch_size):
        end = start + batch_size
        if end > len(sequences):
            break
        batches.append({
            "input_ids": masked_input[start:end],
            "attention_mask": attention_mask[start:end],
            "labels": labels[start:end],
        })

    print(f"  Training corpus: {len(sequences)} sequences × {seq_len} tokens "
          f"→ {len(batches)} minibatches of {batch_size}")
    return batches


# ── Phase 2: V2 functional training ─────────────────────────────────────────

# %%
def _student_forward(
    teacher_model: BertForMaskedLM,
    student_params: dict[str, Tensor],
    batch: dict[str, Tensor],
) -> Tensor:
    """Run a differentiable forward pass using projected parameters.

    Uses functional_call with tie_weights to ensure the projected embedding
    weight flows through the output decoder head.
    """
    return functional_call(
        teacher_model,
        student_params,
        kwargs={
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        },
        strict=False,
        tie_weights=TIE_WEIGHTS,
    ).logits


def _kl_loss_at_mask(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
) -> Tensor:
    """KL(teacher || student) computed only at masked positions."""
    # labels != -100 marks masked positions
    mask = labels != -100  # (batch, seq)
    s_flat = student_logits[mask]  # (n_masked, vocab)
    t_flat = teacher_logits[mask]  # (n_masked, vocab)

    return F.kl_div(
        F.log_softmax(s_flat, dim=-1),
        F.softmax(t_flat, dim=-1),
        reduction="batchmean",
    )


def train_projection_v2(
    projection: BiasProjection,
    serialized_list: list[Tensor],
    config: BertConfig,
    training_batches: list[dict[str, Tensor]],
) -> BiasProjection:
    """Train W_in/W_out via end-to-end functional distillation.

    For each training seed: load teacher, run teacher forward (no grad),
    project serialized vectors through W_in/W_out, deserialize into BERT
    params, run student forward via functional_call, compute KL loss,
    backpropagate into W_in/W_out.
    """
    projection = projection.to(DEVICE).train()
    optimizer = torch.optim.Adam(projection.parameters(), lr=V2_LR)
    n_batches = len(training_batches)

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"V2 Functional Training")
    print(f"  {sum(p.numel() for p in projection.parameters()):,} trainable params")
    print(f"  {len(TRAIN_SEEDS)} train seeds × {V2_BATCHES_PER_SEED} batches/seed")
    print(f"  lr={V2_LR}, max_epochs={V2_MAX_EPOCHS}, patience={V2_PATIENCE}")
    print(f"{'='*60}")

    for epoch in range(V2_MAX_EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0
        t_epoch = time.time()

        # Shuffle seed order each epoch
        seed_order = TRAIN_SEEDS.copy()
        random.shuffle(seed_order)

        for seed in seed_order:
            # ── Load teacher ─────────────────────────────────────────
            teacher = load_multibert_model(seed).eval().to(DEVICE)

            # Freeze teacher parameters (only projection is trained)
            for p in teacher.parameters():
                p.requires_grad_(False)

            # Pick random minibatches for this seed
            batch_indices = random.sample(range(n_batches), V2_BATCHES_PER_SEED)

            for bi in batch_indices:
                batch = {k: v.to(DEVICE) for k, v in training_batches[bi].items()}

                # ── Teacher forward (no grad) ────────────────────────
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits.detach()

                # ── Projection round-trip (gradient flows) ───────────
                x_ser = serialized_list[seed]
                x_recon = apply_projection_to_bias_rows_with_grad(
                    projection, x_ser, config, DEVICE
                )
                student_bert_params = deserialize(x_recon, config)

                # Prefix keys with 'bert.' for functional_call
                student_params = {
                    "bert." + k: v for k, v in student_bert_params.items()
                }

                # ── Student forward (gradient flows) ─────────────────
                student_logits = _student_forward(teacher, student_params, batch)

                # ── KL loss at masked positions ──────────────────────
                loss = _kl_loss_at_mask(
                    student_logits, teacher_logits, batch["labels"]
                )

                # ── Backward + step ──────────────────────────────────
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

                # Free intermediate graph references
                del x_ser, x_recon, student_bert_params, student_params
                del student_logits, teacher_logits, loss

            # ── Cleanup teacher ──────────────────────────────────────
            del teacher
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(epoch_steps, 1)
        dt = time.time() - t_epoch

        if epoch % V2_LOG_EVERY == 0:
            print(f"  epoch {epoch:3d}  loss={avg_loss:.6f}  "
                  f"({dt:.1f}s, {epoch_steps} steps)")

        # ── Validation ───────────────────────────────────────────────
        if (epoch + 1) % V2_VAL_EVERY == 0:
            val_loss = _validate(
                projection, serialized_list, config, training_batches
            )
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best checkpoint
                torch.save(
                    projection.state_dict(),
                    RESULTS_DIR / "projection_v2_best.pt",
                )
            else:
                patience_counter += V2_VAL_EVERY

            print(f"  ── val loss={val_loss:.6f}  best={best_val_loss:.6f}  "
                  f"patience={patience_counter}/{V2_PATIENCE} "
                  f"{'*' if improved else ''}")

            if patience_counter >= V2_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Load best checkpoint
    best_path = resolve_existing_path(
        RESULTS_DIR / "projection_v2_best.pt",
        LEGACY_RESULTS_DIR / "projection_v2_best.pt",
    )
    if best_path.exists():
        projection.load_state_dict(torch.load(best_path, weights_only=True))
        print(f"  Loaded best checkpoint (val_loss={best_val_loss:.6f})")

    projection = projection.cpu()
    return projection


@torch.no_grad()
def _validate(
    projection: BiasProjection,
    serialized_list: list[Tensor],
    config: BertConfig,
    training_batches: list[dict[str, Tensor]],
) -> float:
    """Compute validation KL loss on held-out seeds (fast, no full perplexity)."""
    projection.eval()
    total_loss = 0.0
    total_steps = 0

    for seed in TEST_SEEDS:
        teacher = load_multibert_model(seed).eval().to(DEVICE)

        # Use a single batch for fast validation
        batch = {k: v.to(DEVICE) for k, v in training_batches[0].items()}
        teacher_logits = teacher(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        x_recon = apply_projection_to_bias_rows_with_grad(
            projection, serialized_list[seed], config, DEVICE
        )
        student_bert_params = deserialize(x_recon, config)
        student_params = {"bert." + k: v for k, v in student_bert_params.items()}

        student_logits = _student_forward(teacher, student_params, batch)
        loss = _kl_loss_at_mask(student_logits, teacher_logits, batch["labels"])
        total_loss += loss.item()
        total_steps += 1

        del teacher, x_recon, student_bert_params, student_params
        del student_logits, teacher_logits, loss
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    projection.train()
    return total_loss / max(total_steps, 1)


# ── Phase 3: V1 comparison ──────────────────────────────────────────────────

# %%
def evaluate_v1_comparison(
    v2_projection: BiasProjection,
    serialized_list: list[Tensor],
    config: BertConfig,
    ref_batch: dict,
) -> dict:
    """Metric 4: side-by-side V1 vs V2 perplexity on test seeds."""
    # Load V1 projection
    v1_projection = BiasProjection(config.hidden_size)
    v1_state = torch.load(
        resolve_existing_path(V1_PROJECTION_PATH, LEGACY_V1_PROJECTION_PATH),
        weights_only=True,
    )
    v1_projection.load_state_dict(v1_state)

    print("\n  V1 projection (weight-space MSE):")
    v1_results = evaluate_mlm_perplexity_silent(
        v1_projection, serialized_list, TEST_SEEDS, config, ref_batch, tag="V1"
    )

    print("\n  V2 projection (functional KL):")
    v2_results = evaluate_mlm_perplexity_silent(
        v2_projection, serialized_list, TEST_SEEDS, config, ref_batch, tag="V2"
    )

    combined = {}
    for seed in TEST_SEEDS:
        key = f"seed_{seed}"
        combined[key] = {
            "original_ppl": v2_results[key]["original_ppl"],
            "v1_ppl": v1_results[key]["reconstructed_ppl"],
            "v1_ratio": v1_results[key]["ratio"],
            "v2_ppl": v2_results[key]["reconstructed_ppl"],
            "v2_ratio": v2_results[key]["ratio"],
        }

    return combined


# ── Main entry point ─────────────────────────────────────────────────────────

# %%
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    t0 = time.time()

    print("=" * 70)
    print("BIAS ABSORPTION V2: END-TO-END FUNCTIONAL TRAINING")
    print("=" * 70)

    # ── Phase 1: Load data ───────────────────────────────────────────────────
    print("\n▸ Phase 1: Loading data...")
    # Use lazy loading to avoid holding all 25 tensors (~10 GB) in RAM
    serialized = LazySerializedList(SERIALIZED_CACHE, NUM_SEEDS)
    # Load config from first model
    config = BertConfig.from_pretrained("google/multiberts-seed_0")
    d_model = config.hidden_size
    labels = vector_component_labels(config)

    print(f"\n  Config: d_model={d_model}, d_ff={config.intermediate_size}, "
          f"n_layers={config.num_hidden_layers}, vocab={config.vocab_size}")
    print(f"  Vectors per model: {serialized.shape_per_seed[0]:,}")
    print(f"  Lazy loading: tensors loaded from disk on demand")

    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")

    print("\n▸ Preparing training corpus (WikiText-103 train split)...")
    training_batches = get_training_batches(tokenizer)

    print("\n▸ Preparing evaluation reference batch (WikiText-103 test split)...")
    ref_batch = get_reference_batch(tokenizer)
    print(f"  Reference batch: {ref_batch['input_ids'].shape}")

    # ── Phase 2: Train V2 projection ─────────────────────────────────────────
    print("\n▸ Phase 2: Training V2 projection (end-to-end functional)...")
    projection = BiasProjection(d_model)
    projection = train_projection_v2(
        projection, serialized, config, training_batches,
    )
    torch.save(projection.state_dict(), RESULTS_DIR / "projection_v2.pt")

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
            print(f"    {ctype:8s}  corr={vals['correlation']:.6f}  "
                  f"mae={vals['mae']:.2e}")

    # Metric 3: MLM perplexity of reconstructed models
    print("\n  ── Metric 3: MLM perplexity of reconstructed models ──")
    results["metric3"] = evaluate_mlm_perplexity_silent(
        projection, serialized, TEST_SEEDS, config, ref_batch, tag="V2"
    )

    # Metric 4: V1 comparison
    print("\n  ── Metric 4: V1 vs V2 comparison ──")
    if resolve_existing_path(V1_PROJECTION_PATH, LEGACY_V1_PROJECTION_PATH).exists():
        results["metric4"] = evaluate_v1_comparison(
            projection, serialized, config, ref_batch,
        )
        print("\n  Summary:")
        for key, vals in results["metric4"].items():
            print(f"    {key}: V1 ratio={vals['v1_ratio']:.1f}  "
                  f"V2 ratio={vals['v2_ratio']:.4f}")
    else:
        print(
            "  SKIPPED: V1 projection not found at "
            f"{resolve_existing_path(V1_PROJECTION_PATH, LEGACY_V1_PROJECTION_PATH)}"
        )
        results["metric4"] = "skipped"

    # Metric 5: Symmetry preservation
    print("\n  ── Metric 5: Symmetry preservation ──")
    for seed in TEST_SEEDS:
        print(f"\n  Seed {seed}:")
        layer_idx = seed % config.num_hidden_layers

        neuron_result = test_symmetry_neuron(
            serialized[seed], config, projection, layer_idx
        )
        results[f"metric5_neuron_seed{seed}"] = neuron_result
        print(f"    Neuron perm: sets_identical={neuron_result['sets_identical']}  "
              f"outputs_match={neuron_result['outputs_match']}  "
              f"max_diff={neuron_result['max_set_diff']:.2e}")

        head_result = test_symmetry_head(
            serialized[seed], config, projection, layer_idx
        )
        results[f"metric5_head_seed{seed}"] = head_result
        print(f"    Head perm:   sets_identical={head_result['sets_identical']}  "
              f"outputs_match={head_result['outputs_match']}  "
              f"max_diff={head_result['max_set_diff']:.2e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE  ({elapsed/60:.1f} minutes)")
    print(f"{'='*70}")

    # Success criteria
    print("\n▸ Success criteria:")
    if "metric3" in results:
        ratios = [v["ratio"] for v in results["metric3"].values()]
        mean_ratio = sum(ratios) / len(ratios)
        print(f"  MLM perplexity ratio (mean): {mean_ratio:.4f}  "
              f"{'PASS' if mean_ratio < 1.05 else 'FAIL'} (target < 1.05)")
        if mean_ratio < 1.01:
            print(f"  → GOOD: ratio < 1.01")

    if isinstance(results.get("metric4"), dict):
        v1_ratios = [v["v1_ratio"] for v in results["metric4"].values()]
        v2_ratios = [v["v2_ratio"] for v in results["metric4"].values()]
        print(f"  V1 mean ratio: {sum(v1_ratios)/len(v1_ratios):.1f}")
        print(f"  V2 mean ratio: {sum(v2_ratios)/len(v2_ratios):.4f}")
        print(f"  Improvement: {sum(v1_ratios)/len(v1_ratios) / (sum(v2_ratios)/len(v2_ratios)):.0f}×")

    # Save results
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
        if isinstance(obj, int):
            return obj
        return str(obj)

    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()

# %%
