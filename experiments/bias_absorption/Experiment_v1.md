**Bias Absorption Validation Experiment Using MultiBERTs**

**Goal.** Validate the learned input/output projection with bias absorption (Section 4.1) in isolation, before integrating it into the full EigenCritic pipeline. The projection W_in: (d_model+1) → d_model should compress each weight-bias pair into d_model dimensions with negligible functional information loss, and the round-trip through W_out: d_model → (d_model+1) should reconstruct models that remain performant. The experiment also verifies that per-vector bias absorption preserves neuron permutation symmetry while the alternative chunked-bias serialization breaks it.

**Data.** Use all 25 MultiBERTs checkpoints (BERT-base, d_model=768, 12 layers, 12 heads, d_ff=3072, vocab_size=30522, identical hyperparameters, different random seeds). Split: seeds 0–19 for training, seeds 20–24 for test. Prepare a reference text batch for evaluating reconstructed models — a few thousand tokens from Wikipedia, matching BERT's pretraining distribution.

**Serialization: per-vector bias absorption for BERT-base.** Each BERT layer contains attention (Q, K, V, O projections with biases) and MLP (intermediate dense with bias, output dense with bias) and two LayerNorms (γ, β each). Under bias absorption:

- Q, K, V weight columns (768 each per layer): each column pairs with its corresponding bias entry → (d_model+1)-dimensional vector `[w; b]`. Total: 2304 bias-carrying vectors per layer.
- O weight columns (768 per layer): no per-vector bias pairing → padded as `[w; 0]`.
- O bias: 1 standalone d_model vector → padded as `[b_O; 0]`.
- MLP intermediate (up) rows (3072 per layer): each row pairs with its b₁ entry → `[w; b]`. Total: 3072 bias-carrying vectors per layer.
- MLP output (down) columns (3072 per layer): no per-vector bias pairing → padded as `[w; 0]`.
- MLP b₂: 1 standalone d_model vector → padded as `[b₂; 0]`.
- LayerNorm γ, β (2 norms × 2 vectors = 4 per layer): padded as `[v; 0]`.

Per layer: 2304 + 768 + 1 + 3072 + 3072 + 1 + 4 = 9222 vectors (5376 carry a paired bias scalar, 3846 are zero-padded).
Global: word embeddings (30522) + position embeddings (512) + token type embeddings (2) + embedding LayerNorm γ, β (2) = 31038 vectors, all zero-padded.
Total: 9222 × 12 + 31038 = 141,702 vectors. Each is (d_model+1) = 769-dimensional. Verify this accounts for all ~110M BERT-base parameters with zero remainder.

**Trainable module.** W_in of shape (769 × 768) and W_out of shape (768 × 769). Total: 769 × 768 + 768 × 769 = 1,181,184 parameters (~1.2M). Initialized as W_in = [I₇₆₈ | 0] (identity with zero column for the bias slot) and W_out = [I₇₆₈; 0ᵀ] (identity on top, zero row on bottom). At initialization, the round-trip discards all bias information and passes weight vectors through unchanged — a no-bias baseline that training improves upon.

**Training objective.** Joint reconstruction loss over all (d_model+1)-dimensional vectors from the training models. For each vector x_i of dimension 769:

```
x̂_i = W_out @ (W_in @ x_i)
L_recon = (1/N) Σ_i ||x_i - x̂_i||²
```

This is a linear autoencoder with a d_model bottleneck. The optimal W_in and W_out span the top d_model principal components of the (d_model+1)-dimensional data distribution, discarding the single least-variance direction. Training by gradient descent converges to this optimum (or a rotation of it) without explicitly computing PCA. Use Adam, lr=1e-3, train until convergence (the loss landscape is convex for linear autoencoders, so convergence is fast — minutes on CPU).

Train a single shared (W_in, W_out) pair across all 20 training models (~2.8M vectors). The projection must work for all component types simultaneously — Q columns, MLP rows, embeddings, LayerNorms — so the learned subspace must accommodate the statistics of all vector types.

**Evaluation protocol.**

**Metric 1 — Reconstruction error by component type.** For each held-out model (seeds 20–24), compute mean squared reconstruction error ||x - x̂||² separately for each component type: Q (with bias), K (with bias), V (with bias), O (zero-padded), MLP up (with bias), MLP down (zero-padded), standalone biases (b_O, b₂), LayerNorms, embeddings. Report the ratio of reconstruction error to original vector norm: ||x - x̂||² / ||x||². This reveals which component types lose the most information under the d_model bottleneck. Vectors that were zero-padded have trivially zero error in the bias slot; the interesting signal is whether bias-carrying vectors lose meaningful bias information.

**Metric 2 — Bias reconstruction accuracy.** For each bias-carrying vector, extract the reconstructed bias scalar (the (d_model+1)-th element of x̂) and compare to the original. Report correlation coefficient and mean absolute error between original and reconstructed bias values, broken down by component type (Q bias, K bias, V bias, MLP b₁). This directly measures how much bias information survives the round-trip.

**Metric 3 — MLM perplexity of reconstructed models.** For each held-out model, reconstruct all weight vectors via the round-trip x̂ = W_out(W_in(x)). Reassemble into a full BERT model (extract the d_model weight portion and the bias scalar from each 769-dim reconstructed vector). Run the reconstructed model on the reference text batch and compute masked language modeling perplexity. Compare to the original model's perplexity. Report the ratio (reconstructed perplexity / original perplexity) for all 5 held-out models. A ratio close to 1.0 means the bottleneck lost negligible functional information.

**Metric 4 — Comparison to bias-discarding baseline.** The identity initialization W_in = [I | 0] simply discards biases. Reassemble a BERT model using the original weights but with all biases set to zero. Measure MLM perplexity. This is the "no bias" baseline — the performance floor that learned W_in/W_out must beat. The gap between this baseline and the original model quantifies how much functional information lives in the biases. The gap between this baseline and the learned reconstruction quantifies how much of that information W_in/W_out recovers.

**Metric 5 — Symmetry preservation test.** For each held-out model, select a random layer. Apply a random permutation π to the MLP neurons in that layer (permute rows of intermediate dense, corresponding entries of b₁, and columns of output dense together). This produces a functionally identical model. Serialize both the original and permuted models using bias absorption. Verify:
- (a) The set of (d_model+1)-dimensional vectors for the MLP intermediate in that layer is identical under permutation — the vectors are merely reordered, not changed in content. This is true by construction (each neuron's bias travels with its weight vector).
- (b) After round-trip reconstruction, the reconstructed models produce identical outputs (up to floating point). This confirms W_in/W_out preserves the permutation equivariance.

Repeat for attention head permutation: permute heads within a layer (move all Q, K, V, O vectors for a head together). This should similarly produce an identical set of serialized vectors, merely reordered.

**Baselines.**

**Baseline 1 — Chunked bias vectors.** The alternative serialization that Section 4.1 argues against. Instead of per-vector bias absorption, serialize bias vectors as separate tokens: chunk each d_ff-dimensional b₁ into d_ff/d_model = 4 vectors of dimension d_model (for d_ff=3072, d_model=768). Similarly chunk b_Q, b_K, b_V (each d_model-dimensional, so 1 vector each). All weight vectors remain d_model-dimensional (no +1 dimension, no W_in/W_out needed). Measure: (a) total vector count (should be slightly different), (b) MLM perplexity of the reconstructed model (trivially perfect since no projection is needed — all information is preserved), (c) symmetry test — apply MLP neuron permutation π and verify that the chunked b₁ vectors change *content*, not just order. Specifically, if b₁ is chunked into 4 vectors of 768 entries each, permuting the 3072 MLP neurons shuffles entries *across* chunks, producing different chunk contents. This breaks equivariance: the meta-transformer would see different token content for functionally identical models. Document this failure mode explicitly.

**Baseline 2 — PCA projection (non-learned).** Compute the top 768 principal components of all (d_model+1)-dimensional vectors from the training set. Project via PCA encode/decode. Compare reconstruction error and MLM perplexity to the learned W_in/W_out. The learned projection should converge to approximately the same solution (both are linear bottleneck autoencoders), but the comparison validates that training dynamics reach the optimum.

**Baseline 3 — Random projection.** Project (d_model+1) → d_model via a random orthogonal projection (random 768 rows of a 769×769 orthogonal matrix). Reconstruct via pseudoinverse. This baseline quantifies how much structure the learned/PCA projections exploit beyond random compression.

**Ablations.**

**Ablation 1 — Per-component-type W_in/W_out.** Instead of a single shared projection, train separate (W_in, W_out) pairs for each component type (one for Q vectors, one for K vectors, one for MLP up, etc.). This has ~10× more parameters but allows each projection to specialize. Compare reconstruction error and MLM perplexity to the shared projection. If the shared projection is nearly as good, that supports the architectural choice of a single W_in for all vector types.

**Ablation 2 — Bias magnitude scaling.** The bias scalars and weight vectors may have very different magnitudes. Test whether scaling biases by a constant factor c before concatenation (and dividing by c after reconstruction) improves the projection's allocation of capacity. Sweep c ∈ {0.1, 0.5, 1.0, 2.0, 10.0}. If c=1.0 is suboptimal, this suggests a normalization step should precede bias absorption.

**Ablation 3 — Nonlinear projection.** Replace the linear W_in/W_out with a shallow MLP: (d_model+1) → d_model → (d_model+1) with a single hidden layer and GELU activation. This tests whether the information loss from the linear bottleneck is fundamental (rank deficiency) or whether a nonlinear encoder can pack more information into d_model dimensions. If the nonlinear version is significantly better, the linear projection may need rethinking; if comparable, the linear design is validated.

**Ablation 4 — Bias-only vectors.** Instead of absorbing biases into weight vectors, add each bias as its own separate d_model-dimensional token (no concatenation, no projection needed). Compare vector count, symmetry preservation (b₁ entries are individual scalars zero-padded to d_model — each neuron's bias is its own token, so neuron permutation just reorders tokens, preserving symmetry). This is a viable alternative design; the experiment quantifies the tradeoff: more tokens (higher sequence length for the meta-transformer) vs. no projection information loss.

**Analysis.**

**Information-theoretic bound.** The (d_model+1) → d_model linear bottleneck necessarily discards one dimension of information per vector. Compute the variance explained by the top d_model principal components across all training vectors. Report the fraction of total variance in the discarded 769th component. If this fraction is very small (e.g., < 0.1%), the bottleneck is near-lossless in practice regardless of what the downstream perplexity metrics show.

**Bias contribution analysis.** For each component type, measure the ratio of bias magnitude to weight vector magnitude: ||b|| / ||w||. Components where biases are relatively large (high ratio) are where bias absorption matters most and where reconstruction error matters most. Cross-reference with Metric 1 (per-component reconstruction error) and Metric 3 (perplexity impact).

**Layer-wise sensitivity.** Reconstruct models with the round-trip applied to only one layer at a time (all other layers use original weights). Measure per-layer perplexity impact. This identifies which layers are most sensitive to the bias absorption bottleneck and may inform future layer-specific projection strategies.

**Compute budget.** The 25 MultiBERTs checkpoints are ~10GB (shared with the canonicalization experiment). Training the linear autoencoder is trivially fast (convex loss, ~1.2M parameters, convergence in minutes on CPU). The expensive part is MLM perplexity evaluation: 5 held-out models × (original + reconstructed + no-bias + PCA + random + ablations) ≈ 40 forward passes on the reference batch. At ~2 minutes per forward pass on a 3060, evaluation takes ~80 minutes. Including ablations and analysis, the full experiment fits in a single day.

**Success criteria.**

Minimum: learned W_in/W_out reconstruction achieves perplexity ratio < 1.05 (less than 5% degradation) on held-out models, and bias reconstruction correlation > 0.95 for all component types.

Good: perplexity ratio < 1.01 (negligible degradation), variance in the discarded 769th component < 0.1% of total, and the symmetry preservation test passes exactly for both neuron and head permutations. Chunked-bias baseline demonstrably fails the symmetry test.

Great: shared W_in/W_out matches per-component-type projections (Ablation 1) within noise, confirming that a single shared projection suffices. Linear projection matches nonlinear (Ablation 3), confirming the information loss is fundamental to the rank deficiency and not to the choice of linear compression. Layer-wise sensitivity analysis reveals that early layers (with larger relative bias magnitudes) are most affected, providing actionable guidance for the full pipeline.

Publishable: all of the above, plus the information-theoretic analysis shows the 769th principal component carries < 0.01% of total variance across all 25 models, establishing that the d_model bottleneck is near-lossless as an empirical fact for BERT-scale models. The bias-only alternative (Ablation 4) provides a clean comparison of the sequence-length vs. information-loss tradeoff, with concrete numbers for the meta-transformer's attention cost.