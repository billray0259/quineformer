**Canonicalization Module Validation Experiment Using MultiBERTs**

**Goal.** Train and validate the learned canonicalization module (Section 3.3) in isolation, before building the full EigenCritic. The module should learn to map independently trained BERT models into a shared coordinate system such that their weights become directly compatible — verified by the functional test that weight-averaged models remain performant.

**Data.** Download all 25 MultiBERTs checkpoints from HuggingFace (BERT-base, d_model=768, 12 layers, 12 heads, 30522 vocab, identical hyperparameters, different random seeds). Split: seeds 0–19 for training (190 unique pairs), seeds 20–24 for test (10 unique pairs). Also prepare a reference text batch for evaluating averaged models — a few thousand tokens from Wikipedia or BookCorpus, the same data distribution BERT was trained on.

**Trainable module.** W_down (30522 × 768, ~23.4M parameters), canonicalization transformer (2 layers, 4 heads, d_model=768, ~14M parameters), learned temperature τ. Total ~37M parameters. The module takes a model's embedding matrix E, computes M = (1/|vocab|) · E^T @ W_down, feeds M's rows through the canonicalization transformer, and produces a soft permutation matrix P = softmax(attention_logits / τ).

**Training loop.** Each step: sample two models A and B from different training seeds. Extract their embedding matrices. Run both through the canonicalization module to get P_A and P_B. Apply P_A to every d_model-dimensional vector in model A (all Q, K, V, O, MLP, LayerNorm, and embedding parameters — ~160K vectors). Apply P_B to every vector in model B. Sample an interpolation coefficient α ~ Uniform(0, 1). Compute interpolated weights: W_interp = α · P_A(A) + (1-α) · P_B(B). Run a forward pass through the interpolated BERT model on the reference text batch. Compute the loss as KL divergence between the interpolated model's output distribution and the average of both parent models' output distributions (computed with the original, untransformed models): KL(interp_output || 0.5 · A_output + 0.5 · B_output). Add sharpness regularization: λ_sharp · mean row entropy of P_A and P_B. Backprop through the BERT forward pass, through the interpolated weights, through the P applications, into the canonicalization module. Only the module's parameters are updated.

The gradient signal from this loss is rich: every weight in the model contributes to the output, so every weight votes on what P should be. The module learns the coordinate transformation that makes all parameters — not just embeddings — simultaneously compatible across models.

**Memory management.** The backward pass stores the computational graph linking P to ~160K weight applications to the BERT forward pass. On a 3060 (12GB), use a small input batch (4–8 sequences of length 128), and gradient checkpoint the BERT forward pass to trade compute for memory. One model pair per optimizer step. If still too tight, randomly include only a subset of weight matrices in the computational graph each step (apply P to all weights for the forward pass, but detach a random fraction from autograd). This gives an unbiased but noisier gradient estimate.

**Evaluation metrics on held-out seeds 20–24.**

Metric 1 — Linear mode connectivity. For each pair of held-out models, canonicalize both, interpolate at α = 0, 0.1, 0.2, ..., 1.0, evaluate MLM perplexity at each point. Plot the perplexity curve. A successful canonicalization produces a flat or gently convex curve (no loss barrier at the midpoint). Compare against three baselines: naive interpolation (no canonicalization), random P (random permutation applied to each model), and Git Re-Basin (Hungarian algorithm on activation correlations). The learned module should match or beat Re-Basin.

Metric 2 — Midpoint perplexity. The single most important number. Take the α=0.5 averaged model after canonicalization, measure its MLM perplexity. Compare to the average perplexity of the two parent models. The ratio (midpoint perplexity / mean parent perplexity) should be close to 1.0 for good canonicalization. Naive averaging gives a ratio >> 1 (the midpoint model is broken). Report this ratio for all 10 held-out pairs.

Metric 3 — Permutation recovery on synthetic data. Generate 10 random coordinate permutations per held-out model. Canonicalize the original and each permuted copy. Compare the argmax of each row of P to the known ground-truth permutation. Report fraction of coordinates correctly recovered. This tests whether the module has learned the specific mechanism of undoing coordinate permutations, not just finding some transformation that makes averaging work.

Metric 4 — P conditioning. Report the condition number of P across all test models. Should be close to 1 (well-conditioned, near-permutation). If large, the P⁻¹ needed for QuineFormer's output mapping will be numerically unstable.

Metric 5 — Cross-seed weight cosine similarity. After canonicalizing two models from different seeds, compute mean cosine similarity between corresponding weight vectors across all ~160K positions. Higher means the coordinate systems are better aligned. Compare to cosine similarity without canonicalization (should be low due to arbitrary coordinate systems) and with ground-truth permutation alignment on synthetic copies (should be ~1.0).

**Ablations.**

Ablation 1 — No sharpness regularization. Does P become diffuse? Does conditioning degrade? Does midpoint perplexity worsen?

Ablation 2 — Factored W_down. Replace with A @ B at rank r = 64, 128, 256. How much compression before midpoint perplexity degrades?

Ablation 3 — Restricted vocabulary. Use only the top-k tokens (k = 1000, 3000, 5000, 10000, full). Measures robustness to the vocabulary-intersection scenario for cross-family generalization.

Ablation 4 — Fixed α=0.5 vs. sampled α ~ Uniform(0,1) during training. Does sampling across the full interpolation range improve the flatness of the connectivity curve, or is midpoint-only training sufficient?

Ablation 5 — Comparison to Re-Basin. Run Git Re-Basin (activation matching with Hungarian algorithm) on the same held-out pairs. Compare midpoint perplexity and linear mode connectivity curves. The learned module should be competitive despite never seeing activations — only embeddings.

**Compute budget.** The 25 MultiBERTs checkpoints total ~10GB to download. The canonicalization module is ~37M parameters, trained on 190 pairs with a small input batch per step. Estimated training time: a few hours on a 3060. Evaluation (computing perplexity curves for 10 pairs × 11 interpolation points × 4 conditions) takes a few hours more. Including ablations, the full experiment fits in 2–3 days.

**Success criteria.**

Minimum: midpoint perplexity ratio < 2.0 (averaged model is degraded but not broken) and permutation recovery > 90% on synthetic data.

Good: midpoint perplexity ratio < 1.2 (averaged model nearly matches parents) and linear mode connectivity curve is flat.

Great: learned module matches or beats Re-Basin on midpoint perplexity while using only the embedding matrix (no activation matching, no reference data forward passes).

Publishable: all of the above, plus ablations showing vocabulary restriction degrades gracefully (supporting cross-family generalization), and the conditioning analysis confirms P⁻¹ is numerically stable (confirming viability for QuineFormer).