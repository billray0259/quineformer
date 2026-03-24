**Goal:** Verify that the learned canonicalization module can map independently trained BERT models — which compute similar functions in different coordinate systems — to a consistent canonical representation.

**Phase 1: Data preparation.**

Download all 25 MultiBERTs final checkpoints from HuggingFace. These are 25 BERT-base models (d_model=768, 30K vocab, 12 layers, 12 heads) trained identically except for random seed. Extract the embedding matrix E (shape 30522 × 768) from each.

For each model, also generate K=10 synthetic permuted copies: sample a random permutation of the 768 coordinates, apply it to every weight vector in the model (every row and column indexed by d_model, every LayerNorm parameter, every embedding row). This gives you 25 real models + 250 synthetic permutations = 275 total weight configurations. The synthetic copies are functionally identical to their parent — you know the ground truth permutation.

Split: 20 seeds for training (200 synthetic + 20 real = 220), 5 seeds for test (50 synthetic + 5 real = 55).

**Phase 2: Train the canonicalization module.**

The module consists of W_down (30522 × 768, ~23.4M parameters) and a shallow canonicalization transformer (2 layers, 4 heads, d_model=768, ~14M parameters), plus the temperature scalar τ.

Training objective has three terms:

*Term 1 — Permutation consistency.* For each real model and its K permuted copies, run all K+1 versions through the canonicalization module to produce K+1 canonical embedding matrices. Minimize pairwise MSE between all canonical embeddings of the same underlying model. This teaches the module to undo coordinate permutations.

*Term 2 — Cross-seed discriminability.* Canonical embeddings from different seeds should remain distinct. Take canonical embeddings from two different seeds, compute their MSE. Add a hinge loss: L_discrim = max(0, margin - MSE_cross_seed). This prevents the degenerate solution of mapping everything to the same canonical form regardless of function. The margin should be set empirically based on the natural variation between seeds.

*Term 3 — Sharpness regularization.* Mean row entropy of the soft permutation matrix P, weighted by λ_sharp. Keeps P close to a hard permutation for numerical stability.

Total loss: L = L_consistency + α·L_discrim + λ_sharp·L_sharp

Training is cheap. Each forward pass through the module processes one embedding matrix (30K × 768 matrix multiply with W_down, then a 768-length sequence through a small transformer). No need to serialize the full model weights — only the embedding matrix is input to the canonicalization module. Batch size of 8–16 models fits easily in 12GB.

**Phase 3: Evaluation metrics.**

On the held-out 5 seeds:

*Metric 1 — Consistency error.* For each held-out model and its 10 permutations, compute mean pairwise MSE between canonical representations. Perfect score is 0. Report as a fraction of the mean pairwise MSE between uncanonicalized permuted copies (which should be large). This gives you a normalized "percent of permutation noise removed."

*Metric 2 — Permutation recovery accuracy.* For each synthetic permutation, you know the ground truth π. The module produces P. For each row of P, take the argmax — this is the module's best guess of where that coordinate should map. Compare to π⁻¹. Report fraction of coordinates correctly recovered (out of 768). A random baseline scores 1/768 ≈ 0.13%. A perfect module scores 100%.

*Metric 3 — Discriminability ratio.* Compute mean within-model distance (across permutations of the same seed) and mean between-model distance (across different seeds), both after canonicalization. Report the ratio between/within. Should be large — ideally orders of magnitude. If this ratio is small, the module is collapsing distinct models together.

*Metric 4 — Downstream functional test.* This is the strongest validation. Take two permuted copies of the same model, canonicalize both, then compute the cosine similarity between corresponding weight vectors (not just embeddings — all Q, K, V, O, MLP weights after applying P). If canonicalization works, corresponding vectors from the two copies should be nearly identical. Report mean cosine similarity across all ~160K weight vectors. Do the same for two different seeds — this should be lower, reflecting genuine functional differences.

*Metric 5 — P conditioning.* Report the mean condition number of P across test models. Well-conditioned (close to 1, as it would be for a true permutation matrix) means the inverse will be numerically stable for QuineFormer's output mapping. Poorly conditioned means sharpness regularization needs tuning.

**Phase 4: Ablations.**

*Ablation 1 — No sharpness regularization.* Does P become diffuse? Does permutation recovery drop? Does conditioning degrade?

*Ablation 2 — No discriminability loss.* Does the module collapse distinct seeds to the same representation?

*Ablation 3 — Random W_down (frozen).* How much does learned W_down contribute versus random projections from vocab space? If random projections work almost as well, the specific vocabulary-concept structure matters less than the shared indexing.

*Ablation 4 — Factored W_down.* Replace W_down (30522 × 768) with A @ B where A is (30522 × r) and B is (r × 768). Test r = 64, 128, 256. How much compression can you tolerate before permutation recovery degrades?

*Ablation 5 — Fewer vocabulary tokens.* Restrict to the top-k most frequent tokens (k = 1000, 3000, 5000, 10000, 30522). At what point does alignment quality drop? This directly tests the vocabulary-intersection generalization story.

**Phase 5: Stretch goal — cross-seed weight alignment.**

If metrics 1–4 look good, apply P from each model to *all* of its weight vectors (not just embeddings), producing fully canonicalized models. Then test linear interpolation between canonicalized models from different seeds. Evaluate the interpolated model on MLM perplexity at interpolation points α = 0, 0.25, 0.5, 0.75, 1. If canonicalization truly resolves the coordinate degeneracy, the interpolated model should maintain reasonable performance (no loss barrier at α=0.5). This replicates the Re-Basin linear mode connectivity result but using your learned module instead of the Hungarian algorithm — a direct comparison point.

**Compute budget.**

The canonicalization module is ~37M parameters. Training data is 220 embedding matrices. Each epoch processes 220 forward passes through the module — seconds on a 3060. The bottleneck is downloading the 25 MultiBERTs checkpoints (~400MB each, ~10GB total). Training itself should converge in under an hour. The whole experiment, including ablations, fits in a day.