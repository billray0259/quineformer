**Bias Absorption V2: End-to-End Functional Training**

**Motivation.** V1 trained W_in/W_out to minimize weight-space MSE and achieved excellent reconstruction (relative MSE ~0.1%, bias correlation >0.998). Yet the reconstructed models are functionally destroyed — perplexity ratios of 10,000–15,000×. This reveals that weight-space MSE is a poor proxy for functional fidelity: tiny perturbations in weight space, spread across 141,702 vectors and amplified through 12 transformer layers, compound into catastrophic output divergence. V2 replaces the weight-space objective with an end-to-end functional loss that backpropagates through the entire BERT model, training W_in/W_out to minimize the distortion that encoding and decoding introduces to the model's actual behavior.

**Key insight.** The `deserialize()` function uses only autograd-compatible operations (slicing, transposing, indexing). This means we can construct a differentiable pipeline: serialize → project via (W_in, W_out) → deserialize → functional forward pass → loss. Gradients flow from the output distillation loss all the way back through BERT's computation graph, through the deserialized parameter tensors, through the projection round-trip, and into W_in and W_out. The projection learns to allocate its single discarded dimension to minimize *functional* impact rather than *geometric* reconstruction error.

**Data.** Same as V1: 25 MultiBERTs (seeds 0–24), split 20 train / 5 test. Serialized to (141,702 × 769) matrices. Reference text from WikiText-103 for evaluation. Additionally, prepare a *training* text corpus: a set of tokenized batches from WikiText-103 train split for computing the functional loss during projection training.

**Architecture.** Same W_in (769 → 768) and W_out (768 → 769) linear projection as V1. Same identity initialization. The only change is the training objective.

**Training objective.** For each training step, sample a MultiBERT seed *s* and a text minibatch *B*:

1. **Teacher forward pass** (no gradient): Run the original model on *B*, collect teacher logits *z_T* of shape (batch, seq_len, vocab). Detach.

2. **Projection round-trip** (gradient flows through W_in, W_out):
   ```
   x_serialized = serialized[s]                    # (141702, 769), detached
   x_reconstructed = W_out(W_in(x_serialized))     # (141702, 769), in graph
   params = deserialize(x_reconstructed, config)    # dict of tensors, in graph
   ```

3. **Student forward pass** (gradient flows through params, hence through W_in/W_out):
   Use `torch.func.functional_call(model_shell, params, input_batch)` to run the reconstructed parameters through a BERT forward pass without mutating any `nn.Module` buffers. The model shell is a parameter-free `BertForMaskedLM` instance; its architecture provides the computation graph while `params` provides the tensor values.

4. **Loss**: KL divergence from teacher to student logits, computed at the masked positions only:
   ```
   L = KL(softmax(z_T / τ) || softmax(z_S / τ))
   ```
   with temperature τ = 1.0 (no temperature scaling — we want exact output matching, not soft-target distillation). Optionally add a small weight-space MSE regularizer:
   ```
   L_total = L_KL + λ · L_MSE
   ```
   with λ = 0.0 by default (pure functional loss) and swept in ablations.

**Training procedure.**

- **Optimizer**: Adam, lr = 1e-4 (lower than V1's 1e-3 because the loss landscape through the full model is more complex).
- **Training corpus**: 256 sequences of 128 tokens from WikiText-103 train split, with 15% masking. Shorter sequences than V1's 512-token evaluation sequences to reduce memory. Pre-tokenized and cached.
- **Epoch structure**: Each epoch iterates over all 20 training seeds. For each seed:
  - Load serialized vectors (from cache, CPU).
  - Run teacher forward pass on a fresh random subset of training batches (1–4 minibatches of 4 sequences each).
  - Run projection → deserialize → student forward pass → loss → backward → step.
  - Delete model tensors, clear cache.
- **Convergence**: Train for up to 100 epochs. Monitor validation loss on held-out seeds every 5 epochs. Early stop if validation loss doesn't improve for 15 epochs.
- **Memory management**: Only one MultiBERT is loaded at a time. Teacher logits are detached and moved to CPU between teacher and student passes. The student forward pass uses gradient checkpointing if needed. Batch size of 4 (sequences) should keep peak VRAM under 10 GB even with the backward pass through 110M parameters.

**Why this works despite 110M parameters in the forward pass.** The 110M BERT parameters are *not* learned — they are fixed outputs of the (W_in, W_out) projection applied to the fixed serialized vectors. The only trainable parameters are W_in and W_out (~1.2M total). The BERT forward+backward computes gradients of the loss w.r.t. every parameter tensor in `params`, then autograd chains these through the linear projection to get gradients w.r.t. W_in and W_out. This is analogous to neural network distillation where the student architecture is fixed and only a small adapter is learned — the backward pass is expensive in compute but cheap in optimizer state (Adam buffers are only 1.2M × 2 floats).

**The compute cost is dominated by BERT backward passes.** Each training step involves one BERT forward (teacher, no grad) and one BERT forward+backward (student). On an RTX 3060 with batch size 4 and sequence length 128, each forward pass takes ~50ms and backward ~100ms. With 20 seeds × 2 minibatches × 100 epochs = 4,000 steps, total training time is ~15–20 minutes. This is significantly more expensive than V1's convex linear autoencoder but still very manageable.

**Evaluation protocol.** Identical to V1 for direct comparison:

1. **Metric 1 — Per-component reconstruction error.** Same as V1. Report weight-space MSE by component type. Expect this to be *worse* than V1 (the functional objective doesn't optimize for weight-space MSE), but this is acceptable if the functional metrics improve.

2. **Metric 2 — Bias reconstruction accuracy.** Same as V1. Correlation and MAE of reconstructed bias scalars.

3. **Metric 3 — MLM perplexity of reconstructed models.** Same as V1. This is the primary metric. Expect perplexity ratio close to 1.0 (the objective directly optimizes for this).

4. **Metric 4 — Comparison to V1 (weight-space MSE baseline).** Load the V1 projection and report its perplexity side-by-side with V2. This replaces V1's "no-bias baseline" as the primary comparison — V1 showed that even excellent weight-space reconstruction fails functionally, so V2's improvement over V1 is the central result.

5. **Metric 5 — Symmetry preservation.** Same as V1. The projection is still a single shared linear map applied identically to permuted vectors, so permutation equivariance should still hold.

**Ablations.**

**Ablation 1 — MSE regularization weight λ.** Sweep λ ∈ {0.0, 0.001, 0.01, 0.1, 1.0}. At λ = 0 the objective is purely functional; at large λ it converges toward V1's weight-space objective. The sweep maps the Pareto frontier between weight-space fidelity and functional fidelity.

**Ablation 2 — Number of training text minibatches per seed.** Sweep {1, 2, 4, 8} minibatches per seed per epoch. More batches provide a more stable gradient estimate of the functional loss but increase training time. Determine the minimum number of batches needed for the projection to generalize across input texts (the projection should work regardless of what text the model processes).

**Ablation 3 — Learning rate.** Sweep lr ∈ {1e-5, 5e-5, 1e-4, 5e-4, 1e-3}. The loss landscape through a full transformer is non-convex, so the learning rate may matter more than in V1's convex problem.

**Ablation 4 — Sequence length.** Sweep sequence length ∈ {32, 64, 128, 256}. Shorter sequences are cheaper but provide less signal; longer sequences stress memory. Determine the minimum context length for effective training.

**Ablation 5 — Hidden state matching.** Replace the output KL loss with MSE on the final hidden state: L = ||h_T - h_S||² averaged over masked positions. This tests whether matching intermediate representations is cheaper or more effective than matching output logits (which project through the 768 → 30522 unembedding matrix and may amplify small differences).

**Analysis.**

**Weight-space vs. function-space Pareto.** Plot per-component weight-space MSE (x-axis) against perplexity ratio (y-axis) for V1, V2, and the λ-sweep ablation. V1 occupies the low-MSE / high-perplexity corner; V2 should occupy the higher-MSE / low-perplexity corner. The λ sweep traces the curve between them.

**Gradient analysis.** After training, examine the learned W_in and W_out. Compute the direction in 769-dim space that V2 discards (the null space of W_in) and compare to V1's discarded direction (the 769th principal component). If they differ, plot the component loadings of both directions: V1 discards the direction of least variance; V2 discards the direction of least functional importance. The difference reveals which weight-space dimensions matter for function despite carrying measurable variance.

**Per-layer impact decomposition.** Apply V2's projection to only one layer at a time (other layers unmodified) and measure perplexity, as in V1's layer-wise sensitivity. Compare the per-layer sensitivity profile between V1 and V2. V2 should show more uniform (and smaller) per-layer impact because its loss forced the projection to account for amplification through downstream layers.

**Compute budget.** Training: ~20 minutes on RTX 3060 (100 epochs × 20 seeds × 2 batches × 150ms per step). Evaluation: same as V1 (~80 minutes for MLM perplexity across all seeds and ablations). Ablation sweeps: 5 × 20 minutes = ~2 hours. Total: ~4 hours. Fits in a single session.

**Success criteria.**

Minimum: V2 projection achieves perplexity ratio < 1.05 on held-out models (5 seeds), demonstrating that end-to-end functional training closes the V1 gap. Symmetry preservation tests continue to pass.

Good: Perplexity ratio < 1.01. Weight-space MSE is worse than V1 but perplexity is orders of magnitude better, cleanly demonstrating that weight-space MSE is a misleading objective.

Great: The gradient analysis reveals that V2's discarded direction differs meaningfully from V1's (the 769th PC), showing that functional importance and variance are not aligned. The λ sweep traces a clean Pareto frontier. Hidden-state matching (Ablation 5) is competitive with output KL, suggesting the distortion is amplified primarily by the unembedding projection.

Publishable: All of the above, plus V2's projection generalizes perfectly across unseen text (Ablation 2 shows 1–2 batches suffice), confirming that the functional distortion from bias absorption is a property of the weight geometry, not the input distribution. The per-layer decomposition reveals which layers concentrate the functional sensitivity, providing actionable guidance for future layer-specific or component-specific projections.
