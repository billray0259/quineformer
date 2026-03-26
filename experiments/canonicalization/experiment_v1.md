# Canonicalization Experiment V1: Interpolation-Aligned Coordinate Canonicalization

## Goal

Train a `CanonicalizationModule` that maps MultiBERT parameters into a shared canonical coordinate system where **linear interpolation of canonical parameters produces functionally valid intermediate models**. The canonicalization operates on the d_model coordinate system (Section 3.3 of the README) and is trained end-to-end with a functional interpolation loss. Pre-trained bias absorption projections from `bias_absorption/run_v1_min.py` are loaded and frozen.

## Motivation

MultiBERT checkpoints share architecture and tokenizer but develop arbitrary residual stream coordinate systems during independent training (different random seeds). Naively interpolating their raw parameters produces non-functional models because dimension 47 in seed 0 may encode an entirely different feature than dimension 47 in seed 5. A learned canonicalization module resolves this coordinate ambiguity by discovering a shared basis in which parameter interpolation is semantically meaningful.

This experiment validates the canonicalization module in isolation before integrating it into the full EigenCritic/QuineFormer pipeline. The interpolation objective is a natural functional test: if canonicalized parameters interpolate well, the canonical space preserves the geometric structure that matters for model behavior.

## Relation to Prior Experiments

- **Bias Absorption V1 (run_v1_min.py):** Trained a shared `BiasProjection` (W_in, W_out) mapping (d_model+1) → d_model → (d_model+1) on bias-carrying vectors across 20 MultiBERT seeds. Achieved mean perplexity ratio ≈ 1.08 on 5 held-out seeds. The learned projection is loaded here and **frozen** — canonicalization training does not modify it.
- **Canonicalization module (quineformer/canonicalization.py):** Implements the architecture from README §3.3: W_down projection, shallow transformer, Sinkhorn normalization → soft permutation matrix P.

## Data

**Source:** 25 MultiBERT checkpoints (seeds 0–24), pre-serialized to `(141,702 × 769)` tensors cached in `bias_absorption/data/multiberts/serialized/`.

**Split:**
- **Training pairs:** All unique unordered pairs from seeds 0–19 → $\binom{20}{2} = 190$ pairs.
- **Validation pairs:** All unique unordered pairs from seeds 20–24 → $\binom{5}{2} = 10$ pairs.
- **Cross-split pairs (held out):** Pairs crossing train/test seeds are not used during training or primary validation, but can be used for additional analysis.

Each training step samples a pair $(i, j)$ and an interpolation factor $\alpha \sim \text{Uniform}(0, 1)$.

## Architecture

### Frozen Components

- **BiasProjection** (W_in, W_out): Loaded from `bias_absorption/results_v1_min/projection_shared.pt`. All parameters frozen.
- **Serialization/deserialization:** Using `quineformer.serialization.{serialize, deserialize}`.

### Trainable Components

- **CanonicalizationModule** (`quineformer.canonicalization`):
  - `W_down`: (30,522 × 768) learned projection from vocab space.
  - Shallow transformer: 2 layers, 4 heads, d_model=768, dim_feedforward=3,072.
  - Learned temperature `log_tau` for Sinkhorn sharpening.
  - **Total trainable parameters:** ~53M.

### Forward Pass (Single Model)

```
serialized_i          # (141702, 769) — cached on disk
    → absorbed_i = W_in(serialized_i)  # (141702, 768) — frozen bias absorption
    → extract word embeddings E_i = absorbed_i[:vocab_size]  # (V, 768)
    → CanonicalizationModule(E_i) → (canonical_E_i, P_i)
    → canonical_i = absorbed_i @ P_i   # (141702, 768) — apply P to all vectors
```

Bias absorption (W_in) maps (d_model+1) → d_model *before* canonicalization, folding each vector's bias scalar into the d_model representation. After interpolation and un-canonicalization, W_out maps d_model → (d_model+1) to recover the bias dimension for deserialization. The bias is never handled as a separate channel — it lives inside the d_model space throughout the canonicalization and interpolation stages.

### Interpolation in Canonical Space

Given models $i$ and $j$ with interpolation factor $\alpha$:

$$\text{canonical\_interp} = (1 - \alpha) \cdot \text{canonical}_i + \alpha \cdot \text{canonical}_j$$

To produce a functional model, the interpolated canonical parameters are mapped back to a coordinate system. Since the interpolated model has no "native" coordinate system, we use the inverse of one of the endpoint models' permutations (arbitrarily $P_i^{-1}$, or equivalently $P_i^T$ since P is approximately orthogonal after Sinkhorn):

$$\text{interp\_params} = \text{canonical\_interp} \cdot P_i^T$$

The interpolated d_model-dimensional vectors are then passed through `W_out` (frozen), which maps d_model → (d_model+1) to recover the weight and bias components, followed by `deserialize()` to reconstruct a full BERT state dict. Because W_in absorbs the bias into d_model space before canonicalization, and W_out extracts it after, there is no need to interpolate the bias dimension separately — it is interpolated implicitly as part of the d_model representation.

## Loss Function

### Primary Loss: Activation MSE

The loss measures whether interpolating in canonical space produces the same intermediate-layer activations as interpolating model outputs directly.

For a reference input batch $x$ (tokenized text), define:

- $A_i^{(\ell)}(x)$: Hidden activations of model $i$ at layer $\ell$ (after the residual connection + LayerNorm), shape `(batch, seq_len, d_model)`.
- $A_j^{(\ell)}(x)$: Same for model $j$.
- $A_{\text{canon-interp}}^{(\ell)}(x)$: Activations of the model reconstructed from interpolated canonical parameters.

**Target activations** (activation-space interpolation — the ground truth):

$$A_{\text{target}}^{(\ell)}(x) = (1 - \alpha) \cdot A_i^{(\ell)}(x) + \alpha \cdot A_j^{(\ell)}(x)$$

Note: target activations require running each endpoint model independently and interpolating their activations. This is the "ideal" behavior — what a perfect canonicalization would achieve — because activation-space interpolation preserves the linear structure of the residual stream at each layer (though it does not account for nonlinear interactions across layers, it serves as a tractable proxy).

**Loss:**

$$\mathcal{L}_{\text{act}} = \frac{1}{L} \sum_{\ell=1}^{L} \frac{\| A_{\text{canon-interp}}^{(\ell)}(x) - A_{\text{target}}^{(\ell)}(x) \|_F^2}{S}$$

where $L$ is the number of layers (12 for BERT-base) and $S = \text{batch\_size} \times \text{seq\_len} \times d_{\text{model}}$ normalizes by the number of activation scalars.

### Sharpness Regularization

$$\mathcal{L}_{\text{sharp}} = \lambda_{\text{sharp}} \cdot \frac{1}{2}\left(H(P_i) + H(P_j)\right)$$

where $H(P)$ is the mean row entropy of P (from `CanonicalizationModule.row_entropy()`). This encourages P to be a near-permutation matrix, keeping $P^{-1}$ well-conditioned.

### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{act}} + \mathcal{L}_{\text{sharp}}$$

## Implementation Details

### Reference Batch

Reuse the same reference batch construction as `bias_absorption/run_v1.py`: WikiText-103 validation set, tokenized with the MultiBERT tokenizer, truncated/padded to a fixed sequence length. A single reference batch is fixed for the entire training run to reduce variance.

### Activation Extraction

Use `torch.func.functional_call()` with the deserialized state dict to run a differentiable forward pass through a BERT shell model. Register forward hooks (or use a wrapper) to capture hidden states at each layer output. The 12 hidden states plus the embedding output (13 total) provide the activation targets.

Since both the interpolated activations (target) and the canon-interp activations (prediction) require forward passes, each training step involves **3 forward passes** through BERT:
1. Model $i$ → activations $A_i$
2. Model $j$ → activations $A_j$
3. Interpolated canonical model → activations $A_{\text{canon-interp}}$

Only pass 3 requires gradients (through the canonicalization module). Passes 1 and 2 are run in `torch.no_grad()`.

### Memory Management

Three BERT forward passes per step on a 12GB GPU is tight. Mitigations:
- **fp16 mixed precision** for all forward passes.
- **Small reference batch** (e.g., 4 sequences × 128 tokens).
- **Gradient checkpointing** on the canonicalization transformer if needed.
- **Detach endpoint activations** (passes 1, 2) — no gradient graph stored.
- **Accumulate gradients** over multiple pairs before optimizer step if single-pair memory is borderline.

### Training Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Optimizer | AdamW | Weight decay 0.01 |
| Learning rate | 1e-4 | With cosine schedule, 10% warmup |
| Epochs | 50 | Over all 190 training pairs |
| Batch size | 1 pair/step | Each step = one (i, j, α) triple |
| Gradient accumulation | 4 steps | Effective batch of 4 pairs |
| α sampling | Uniform(0, 1) | New α each step |
| λ_sharp | 0.1 | Entropy regularization weight |
| τ init | 1.0 | Warm start, annealed by loss |
| Sinkhorn iterations | 20 | |
| Reference batch | 4 × 128 | Sequences × tokens |
| Precision | fp16 mixed | `torch.cuda.amp` |

### Temperature Annealing

τ is learned (via `log_tau`), but we additionally clamp `log_tau` to $[\log(0.05), \log(2.0)]$ to prevent degenerate solutions. The entropy regularization provides the sharpening pressure — as training progresses, the loss naturally drives τ lower for sharper permutations.

## Evaluation

### Primary Metric: Perplexity Ratio of Interpolated Models (α = 0.5)

For each validation pair $(i, j)$ with $\alpha = 0.5$:

1. **Canonical interpolation perplexity:** Construct the interpolated model from averaged canonical parameters (mapped back through $P_i^T$ → `W_out` → `deserialize`). Evaluate MLM perplexity on the reference batch.

2. **Ensemble baseline perplexity:** Run both models independently on the reference batch. Average their final-layer logits (pre-softmax) with equal weight: $\text{logits}_{\text{ensemble}} = 0.5 \cdot \text{logits}_i + 0.5 \cdot \text{logits}_j$. Compute perplexity from the ensembled logits.

3. **Perplexity ratio:**

$$r_{ij} = \frac{\text{PPL}_{\text{canon-interp}}}{\text{PPL}_{\text{ensemble}}}$$

**Report:** Mean perplexity ratio over all 10 validation pairs: $\bar{r} = \frac{1}{10}\sum_{(i,j)} r_{ij}$.

A ratio near 1.0 means canonical interpolation produces a single model that performs as well as a 2-model ensemble — the canonicalization has discovered a coordinate alignment that makes weight averaging functionally meaningful.

### Secondary Metrics

| Metric | Description |
|---|---|
| Per-layer activation MSE | Breakdown of $\mathcal{L}_\text{act}$ by layer, to identify which layers are hardest to align. |
| Permutation sharpness | Mean row entropy of P across validation models. Lower = sharper = more permutation-like. |
| P stability | Frobenius norm of $P_i - P_j$ for pairs within the same split. If canonicalization is consistent, P should be similar for models in the same family. |
| Interpolation curve | Evaluate perplexity at $\alpha \in \{0.0, 0.1, 0.2, \ldots, 1.0\}$ for select pairs. A good canonicalization produces a smooth, monotonic interpolation curve; a failed one produces a sharp perplexity spike at intermediate α. |

### Baselines

1. **Naive interpolation (no canonicalization):** Interpolate raw serialized parameters directly (after bias absorption only). Expected: catastrophic perplexity at α = 0.5 due to coordinate misalignment.

2. **Git Re-Basin (Ainsworth et al., 2023):** Apply the weight-matching algorithm to find a permutation alignment between each pair, then interpolate. This is the established non-learned baseline for model merging.

3. **Random orthogonal P:** Apply a random fixed orthogonal matrix to all models as "canonicalization." Expected: no better than naive, confirming that the learned P captures genuine structure.

## Success Criteria

| Level | Requirement |
|---|---|
| **Minimum** | Canonical interpolation perplexity is finite and significantly better than naive interpolation at α = 0.5. |
| **Good** | Mean perplexity ratio $\bar{r} < 2.0$ (canon-interp within 2× of ensemble). |
| **Great** | Mean perplexity ratio $\bar{r} < 1.5$, smooth interpolation curves. |
| **Excellent** | Mean perplexity ratio $\bar{r} < 1.2$, competitive with Git Re-Basin. |

## Outputs

| File | Contents |
|---|---|
| `results_v1/canonicalization_module.pt` | Trained CanonicalizationModule state dict. |
| `results_v1/results.json` | All metrics: per-pair perplexity ratio, per-layer activation MSE, entropy, P stability. |
| `results_v1/interpolation_curves.json` | Perplexity at 11 α values for each validation pair. |
| `results_v1/training_log.json` | Per-epoch loss, lr, τ, entropy. |

## Pseudocode

```python
# ── Setup ──────────────────────────────────────────────────────────
serialized, config = load_and_serialize_all()          # from bias_absorption/run_v1
projection = BiasProjection(d_model=768)
projection.load_state_dict(torch.load("bias_absorption/results_v1_min/projection_shared.pt"))
projection.eval()
for p in projection.parameters():
    p.requires_grad_(False)

canon = CanonicalizationModule(vocab_size=30522, d_model=768)
optimizer = AdamW(canon.parameters(), lr=1e-4, weight_decay=0.01)
ref_batch = get_reference_batch(tokenizer)
shell_model = BertForMaskedLM(config)  # empty shell for functional_call

vocab_size = config.vocab_size

# ── Training loop ──────────────────────────────────────────────────
for epoch in range(n_epochs):
    for seed_i, seed_j in shuffled(train_pairs):
        alpha = torch.rand(1).item()

        # Absorb biases via frozen W_in: (N, 769) → (N, 768)
        s_i = serialized[seed_i]                                # (N, 769)
        s_j = serialized[seed_j]
        a_i = projection.encode(s_i)                            # (N, 768)
        a_j = projection.encode(s_j)

        # Canonicalize both models
        E_i = a_i[:vocab_size].unsqueeze(0)                     # (1, V, 768)
        E_j = a_j[:vocab_size].unsqueeze(0)
        _, P_i = canon(E_i)                                     # (1, 768, 768)
        _, P_j = canon(E_j)

        canon_i = a_i @ P_i.squeeze(0)                          # (N, 768)
        canon_j = a_j @ P_j.squeeze(0)

        # Interpolate in canonical space, then map back
        canon_interp = (1 - alpha) * canon_i + alpha * canon_j
        interp_768 = canon_interp @ P_i.squeeze(0).T           # back to coord system

        # Recover bias dimension via frozen W_out: (N, 768) → (N, 769)
        interp_769 = projection.decode(interp_768)
        params_interp = deserialize(interp_769, config)

        # Forward passes for activations
        with torch.no_grad():
            A_i = get_all_hidden_states(shell_model, params_i, ref_batch)
            A_j = get_all_hidden_states(shell_model, params_j, ref_batch)
            A_target = [(1 - alpha) * a_i + alpha * a_j for a_i, a_j in zip(A_i, A_j)]

        A_interp = get_all_hidden_states(shell_model, params_interp, ref_batch)

        # Loss
        loss_act = mean([mse(a_pred, a_tgt) for a_pred, a_tgt in zip(A_interp, A_target)])
        loss_sharp = 0.5 * (canon.row_entropy(P_i) + canon.row_entropy(P_j))
        loss = loss_act + lambda_sharp * loss_sharp

        loss.backward()
        # gradient accumulation & optimizer step
```

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| **P collapses to identity:** All models mapped to same P, canonicalization is a no-op. | Monitor P diversity across seeds. The interpolation loss should prevent this since identity P = naive interpolation = high loss. |
| **P becomes diffuse (non-permutation):** Soft assignments → ill-conditioned inverse. | Entropy regularization + τ clamping. Monitor row entropy throughout training. |
| **Activation-space interpolation is a poor target:** Nonlinear interactions across layers mean interpolated activations ≠ activations of an interpolated model, even with perfect coordinate alignment. | This is a known approximation. The per-layer breakdown reveals where the proxy fails. Future work could use a KL-divergence loss on final logits instead. |
| **Memory overflow from 3 BERT passes:** | Small reference batch, fp16, detach endpoint activations, gradient accumulation. |
| **Frozen bias absorption is suboptimal under canonicalization:** The V1 projection was trained without canonicalization; joint fine-tuning might help. | Start frozen. If results are promising, run an ablation unfreezing W_in/W_out with a small learning rate. |

## Future Extensions

- **Joint training with bias absorption:** Unfreeze W_in/W_out and fine-tune end-to-end.
- **KL-divergence loss on logits:** Replace activation MSE with distillation-style KL loss for better functional fidelity.
- **Cross-architecture canonicalization:** Test on model pairs with different architectures sharing the same tokenizer (via vocabulary intersection, README §3.3).
- **Integration with EigenCritic:** Use the trained canonicalization module as the first stage of the meta-transformer pipeline.
