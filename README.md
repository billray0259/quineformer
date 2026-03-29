# QuineFormer Project: Research Notes & Plan

> *Meta-transformers that read, evaluate, and generate neural network parameters as native token sequences, exploiting the d_model decomposition as a structured serialization for same-family models.*
>
> **EigenCritic** — a bidirectional transformer that predicts model quality from serialized weights. The critic. The early experiment.
>
> **QuineFormer** — a bidirectional transformer that reads a transformer's weights and outputs improved weights. The generator. The endgame.

---

## Table of Contents

1. [**Core Insight**](#1-core-insight)
   - The Labeled Bag of $d_{model}$ Vectors
   - Structured Serialization for Same-Family Models
   - The Necessity of Coordinate Canonicalization
2. [**Prior Work Landscape**](#2-prior-work-landscape)
   - 2.1 Diffusion Models over Neural Network Weights
   - 2.2 Weight Nowcasting / Training Trajectory Prediction
   - 2.3 Transformer-Based Weight Generation
   - 2.4 Comparison of Symmetry-Aware Backbones
3. [**The $d_{model}$ Decomposition**](#3-the-d_model-decomposition)
   - 3.1 [Vector Counting: GPT-2 Small](#31-vector-counting-gpt-2-small)
   - 3.2 [Scaling to Larger Models](#32-scaling-to-larger-models)
   - 3.3 [Weight-Space Canonicalization](#33-weight-space-canonicalization)
     - The Problem: Arbitrary Coordinate Systems
     - Solution: Differentiable Canonicalization Module
     - Cross-Family Generalization via Vocab Intersection
4. [**Architecture Design Decisions**](#4-architecture-design-decisions-shared-by-eigencritic-and-quineformer)
   - 4.1 Input/Output Projections with Bias Absorption
   - 4.2 Bidirectional Attention for Global Circuit Detection
   - 4.3 Predicting Full Weights vs. Parameter Deltas
   - 4.4 Structural Metadata: Hybrid Additive & Attention Bias Encodings
   - 4.5 Attention Efficiency and Structured Sparsity
5. [**First Experiment: EigenCritic**](#5-first-experiment-eigencritic--validation-loss-prediction)
   - Motivation: Proving Weight-Space Readability
   - Design Requirements and Performance Baselines
   - Test Protocols: Weak vs. Strong (Cross-Domain) Generalization
6. [**Data Augmentation Strategy**](#6-data-augmentation-strategy-eigencritic-training)
   - 6.1 Label-Preserving: Per-Head Rescaling & Group Shuffling
   - 6.2 Structural Negatives: Shuffled Weights & Norm-Matched Noise
   - 6.3 Controlled Damage: Residual Stream Rotation & Head Ablation
7. [**Future Direction: QuineFormer**](#7-future-direction-quineformer--weight-generation--trajectory-prediction)
   - Three Tiers of Success: Interpolation to Extrapolation
   - Weight Generation as Iterative Refinement
8. [**Future Direction: Learned Regularization Signal**](#8-future-direction-eigencritic-as-learned-regularization-signal)
   - EigenCritic as a Differentiable Prior
   - Initialization Pre-Shaping and RLHF for Weight Space
9. [**Hardware-Constrained Proof of Concept**](#9-hardware-constrained-proof-of-concept-eigencritic)
   - Target Model and Meta-Transformer Scaling on RTX 3060
   - Dataset Generation and Variational Axes
   - 7-Day Implementation Timeline
10. [**Naming & Identity**](#10-naming--identity)
11. [**References**](#11-references)
---

## 1. Core Insight

Every learned parameter in a standard transformer is dimensioned by `d_model` — the width of the residual stream. Token embeddings are `(vocab_size × d_model)`. Attention projections are `(d_model × d_model)`. MLP weights are `(d_model × d_ff)`. Layer norms are vectors of length `d_model`. Even positional embeddings are `(max_seq_len × d_model)`.

This means every weight matrix can be decomposed into a collection of `d_model`-dimensional vectors. A key projection matrix isn't one opaque tensor — it's `d_head` vectors, each defining a direction in residual stream space. An MLP's up-projection is `d_ff` vectors, each a "read direction" from the residual stream. Its down-projection is another `d_ff` vectors, each a "write direction" back into the stream.

A transformer's parameters are therefore a **labeled bag of `d_model` vectors**, each annotated with metadata about where it plugs into the architecture (which layer, which matrix, which position within that matrix).

Since every parameter vector has the same dimensionality as a token embedding, the entire parameter set is **structurally compatible as input to another transformer of the same width** after a learned canonicalization stage (Section 3.3) resolves the arbitrariness of the residual stream coordinate system. A thin learned projection handles bias absorption and distribution normalization (see Section 4.1), initialized to identity so the model starts from the raw weight representation and learns to deviate only where needed.

### What's novel in this synthesis

Individual ingredients all exist in the literature — parameters-as-token-sequences (G.pt), parameters-as-architectural-primitives (TokenFormer), weight-rows-as-directions (Transformer Circuits), transformer-based weight generation (HyperTransformer). **Our contribution is recognizing that d_model creates a convenient and structured serialization for same-family models that makes all of these the same idea.** Within a controlled model family (shared architecture, tokenizer, and training distribution), dot products between parameter-vectors in the meta-transformer correspond to functional interactions in the target model's residual stream. Across independently trained models, this correspondence holds only after the learned canonicalization module (Section 3.3) resolves residual stream coordinate permutations and scaling degeneracies.

Raw weight coordinates are not a canonical semantic space: two functionally equivalent networks can differ by residual stream coordinate permutations, per-head scaling symmetries, and optimizer path effects. The d_model decomposition provides a natural tokenization, but a learned canonicalization module (Section 3.3) is required to ensure that geometric relationships between weight vectors reflect functional similarity rather than parameterization artifacts.

---

## 2. Prior Work Landscape

### 2.1 Diffusion Models over Neural Network Weights

| Paper | Year | Key Idea |
|---|---|---|
| **p-diff** (Wang et al.) | Feb 2024 | Autoencoder + DDPM generates high-performing neural network parameters from noise. Operates on BatchNorm subsets only. Uses 1D CNN encoder/decoder. |
| **D2NWG** (Soro et al.) | Feb 2024, ICLR 2025 | Dataset-conditioned latent diffusion for weight generation. Set Transformer encodes dataset, UNet generates weights. Scaled to LLaMA-3-2-1B (+3% on math reasoning). |
| **W-Diff** (NeurIPS 2024) | 2024 | Captures evolving pattern of classifier weights across domains via diffusion. Uses FIFO queue of historical domain weights. Closest to "learning training trajectories." |
| **Denoised Neural Weights** (Gong et al.) | Jul 2024, ECCV 2024 | UNet diffusion generates GAN weights conditioned on text + block index. 15× training speedup for image translation. |

### 2.2 Weight Nowcasting / Forecasting (Training Trajectory Prediction)

| Paper | Year | Approach | Horizon | Key Limitation |
|---|---|---|---|---|
| **Introspection** (Sinha et al.) | 2017 | Shallow feedforward predictor from weight history | Short | Tiny models only |
| **WNN** (Jang et al., ICML) | 2023 | Two-stream FC network on weight values + temporal diffs | ~5 steps | Treats each weight independently |
| **NiNo** (Knyazev et al., ICLR) | 2025 | GNN on neural graphs capturing neuron connectivity | ~1000 steps | Graph construction nontrivial for transformers |
| **LFD-2** (Shou et al.) | 2024 | Lightweight feedforward, predicts final weights from init + early params | Full trajectory | No dynamics modeling |
| **GFM** (May 2025) | 2025 | Conditional flow matching modeling weight trajectories as structured flows | Full trajectory | Doesn't use diffusion specifically |
| **KNOWN** (Jang et al.) | Aug 2025 | Meta-learned hyper-model that reverses forgetting to predict enhanced weights | **Beyond training** | First to extrapolate past convergence |

### 2.3 Transformer-Based Weight Generation

| Paper | Year | Key Idea |
|---|---|---|
| **G.pt** (Peebles et al.) | Sep 2022 | Conditional diffusion transformer over sequences of neural net parameters. Loss-conditioned: prompts desired loss, generates matching weights in one update. **Max 10K parameters.** |
| **HyperTransformer** (Zhmoginov et al., ICML) | 2022 | Transformer generates CNN weights layer-by-layer from support samples. Autoregressive, one transformer per layer. |
| **MotherNet** (Müller et al.) | 2023 | Hypernetwork transformer generates child network weights from training set via in-context learning in a single forward pass. |
| **TokenFormer** (Wang et al., ICLR Spotlight) | 2025 | Replaces all linear projections with token-parameter attention (Pattention). Parameters ARE tokens within the architecture. Scales 124M→1.4B by appending parameter tokens. |

### 2.4 Architectures for Processing Weight Spaces

| Architecture | Symmetry-aware? | Scales? | Cross-architecture? | Used by |
|---|---|---|---|---|
| Flat MLP/FC | No | Yes (simple) | No | WNN, LFD-2, Introspection |
| 1D CNN on flattened weights | No | Moderate | No | p-diff, Deep Meta Classifier |
| UNet + diffusion | No (arbitrary blocks) | Moderate | No | D2NWG, Denoised Neural Weights |
| GNN on neural graphs | **Yes** | Challenging | **Yes** | NiNo, Graph Metanetworks |
| Flow matching | Depends on backbone | Moderate | Potentially | GFM |
| DeepSets | Over-symmetric | Yes | Yes | Baseline in various papers |
| **Transformer on d_model vectors** | **Yes (hybrid: additive type tags + attention bias for groups + learned coordinate canonicalization)** | **Yes** | **Same tokenizer family (via vocab intersection)** | **EigenCritic / QuineFormer (proposed)** |

### 2.5 Key Related Concepts

- **Mechanistic Interpretability / Transformer Circuits** (Elhage et al., 2021): The residual stream as bottleneck; weight rows/columns as semantically meaningful directions in d_model space.
- **Analyzing Transformers in Embedding Space** (Dar et al., 2022): Translating weight matrices into vocabulary-space operations via embedding projection.
- **ICLR 2025 Workshop**: "Neural Network Weights as a New Data Modality" — dedicated workshop building community around this research direction.

---

## 3. The d_model Decomposition

### 3.1 Vector Counting: GPT-2 Small

For GPT-2 Small (`d_model=768`, 12 layers, 12 heads, `d_ff=3072`):

**Per-layer components (9,222 vectors × 12 layers = 110,664):**
- Q, K, V, O weight matrices: 4 × 768 = 3,072 vectors. Q/K/V columns each carry their per-vector bias scalar as a (d_model+1)-th dimension before input projection (see Section 4.1).
- O bias: 1 vector (shared output-space offset, no per-vector pairing)
- MLP W₁, W₂: 2 × 3,072 = 6,144 vectors. W₁ rows carry their b₁ scalar as a (d_model+1)-th dimension before input projection.
- MLP b₂: 1 vector (shared output-space offset)
- Layer norms (γ, β × 2): 4 vectors

**Global components (51,283 vectors):**
- Token embeddings: 50,257
- Positional embeddings: 1,024
- Final layer norm (γ, β): 2

**Total: 161,947 vectors. 161,947 × 768 + 64,512 absorbed bias scalars = 124,439,808 parameters (124.4M). Zero remainder.**

### 3.2 Scaling to Larger Models

| Model | d_model | Vectors | Context needed | Fits in 1M window? |
|---|---|---|---|---|
| GPT-2 Small (124M) | 768 | 162K | 162K | YES |
| GPT-2 XL (1.6B) | 1,600 | 974K | 974K | YES |
| LLaMA-2 7B (6.7B) | 4,096 | 1.6M | 1.6M | Barely no |
| LLaMA-3 8B (8.8B) | 4,096 | 2.2M | 2.2M | No |
| LLaMA-2 70B (69B) | 8,192 | 8.4M | 8.4M | No |
| LLaMA-3 405B (406B) | 16,384 | 24.8M | 24.8M | No |

Key observation: **the ratio of vectors to parameters shrinks as d_model grows** — a 70B model has ~570× the parameters of GPT-2 Small but only ~52× the vectors. Bigger models are more token-efficient under this decomposition.

### 3.3 Weight-Space Canonicalization

#### The problem

Raw weight coordinates are not a canonical representation of network function. Multiple distinct weight configurations compute identical input-output mappings. If weight vectors are fed directly to the meta-transformer, geometric relationships between tokens (distances, dot products) carry parameterization artifacts rather than purely functional content, and the model must waste capacity learning to ignore irrelevant variation.

Three sources of non-canonicality:

1. **Residual stream coordinate permutation.** The d_model basis — the coordinate system shared by every weight vector in the entire model — is arbitrary. Two independently initialized models develop different internal coordinate systems: dimension 47 in model A might encode the same feature as dimension 302 in model B. Permuting the d_model coordinates across all weight vectors simultaneously leaves the network function unchanged (because elementwise nonlinearities like GELU and LayerNorm's learned gain/bias commute with coordinate permutations, unlike arbitrary rotations). This changes the numerical values inside every token the meta-model sees.

2. **Continuous scaling symmetry.** Within each attention head, W_V can be scaled by any nonzero α with a compensating 1/α applied to W_O, producing an infinite family of equivalent weight configurations per head (see Section 6.1). This is handled by a simple preprocessing step: normalize W_V to unit Frobenius norm and absorb the factor into W_O. No learned module needed.

3. **Optimizer path dependence.** Two independent training runs on the same task converge to functionally near-identical networks that occupy different regions of weight space, related by no clean symmetry group. This is the residual variation after all known symmetries are removed, managed by training data diversity within a controlled model family.

Note: within-layer permutation symmetries (head reordering, MLP neuron reordering, within-head QK/VO dimension permutation) do NOT require canonicalization. These symmetries reorder which tokens appear at which set positions, and the meta-transformer's attention-based architecture is natively equivariant to this reordering (Section 4.4). Only the residual stream coordinate permutation changes the numerical content of token vectors and is therefore visible to the meta-model.

#### The solution: learned differentiable canonicalization

Rather than aligning models to a fixed external reference (which introduces an arbitrary artifact and requires offline preprocessing), we build the canonicalization into the meta-model itself as a learned, differentiable first stage. The module computes a soft permutation matrix P from the target model's own embedding matrix, applies P to every weight vector, and the rest of the meta-model operates in the resulting canonical space. For QuineFormer's generation task, P⁻¹ maps output vectors back to the original coordinate system.

The key insight: the embedding matrix E (shape vocab_size × d_model) bridges a *shared* coordinate system (vocabulary, which is fixed by the tokenizer) and the *arbitrary* coordinate system (d_model basis, which varies per model). Two models using the same tokenizer both have an embedding for the word "the" — the row position in E is the same, but the d_model values differ according to each model's internal coordinate system. This shared vocabulary indexing provides the cross-model correspondence needed for alignment, without any external reference model.

**Step 1: Project the embedding to a d_model × d_model representation.**

E is (vocab_size × d_model), rank d_model. A learned projection W_down of shape (vocab_size × d_model) reduces the vocabulary dimension:

```
M = (1 / |vocab|) · Eᵀ @ W_down
```

M has shape (d_model × d_model). Row i of M describes how residual stream coordinate i relates to d_model learned "concept probes" in vocabulary space. The columns of W_down are shared parameters of the canonicalization module — fixed across all target models. Each column is a direction in vocabulary space that the module has learned to be informative for alignment.

The normalization by |vocab| ensures that the magnitude of M is independent of vocabulary size. This is critical for cross-family generalization (see below).

The projection is lossless in the sense that E has only d_model linearly independent directions, and W_down projects onto d_model probe directions. Gradient descent ensures the probes span the space of E, since non-spanning probes waste capacity.

**Step 2: Canonicalization transformer.**

Feed M's d_model rows as a sequence of d_model tokens, each of dimension d_model, into a shallow transformer. This transformer operates in the concept-probe space (which IS shared across models because W_down is a shared parameter), looking at a sequence of residual stream coordinates (which are NOT yet aligned). Its attention pattern — a (d_model × d_model) matrix — directly yields the alignment logits.

Apply Sinkhorn normalization with learned temperature τ to produce P:

```
P = Sinkhorn(attention_logits / τ)
```

Alternating row and column normalization makes P approximately doubly stochastic. When τ is small and the attention is sharp, each row and column of P concentrates on a single entry, so P approaches a permutation matrix that maps "coordinate i in this model corresponds to position j in canonical space."

**Step 3: Apply P to every weight vector.**

Multiply every d_model-dimensional vector in the serialized model by P. This is a simple matrix-vector product applied token-wise. The meta-transformer then processes all vectors in the learned canonical space.

**Step 4 (QuineFormer only): Inverse mapping.**

The meta-model's output vectors are in canonical space. Multiply each by P⁻¹ to recover weights in the original model's coordinate system. Matrix inversion is differentiable:

```
∂(P⁻¹)/∂P = −P⁻¹ (∂P) P⁻¹
```

So gradients flow from the loss backward through P⁻¹, through the entire meta-model, through the P application, through the canonicalization transformer, and into W_down and τ. The downstream task loss teaches the canonicalization module what coordinate system makes the meta-model's job easiest.

For EigenCritic (scalar output), P⁻¹ is not needed — there are no output weight vectors to un-permute.

#### Sharpness regularization

If P becomes diffuse (rows close to uniform), it is ill-conditioned and P⁻¹ amplifies noise, causing gradient explosion. Two complementary mitigations:

1. **Entropy penalty.** Add L_sharp = λ_sharp · mean_over_rows H(softmax(logits/τ)) to the training loss, where H is Shannon entropy. This directly encourages each row to concentrate on one coordinate.

2. **Temperature annealing.** Initialize τ warm (τ=1, soft assignments, easy gradients) and anneal toward a small value (τ~0.1, sharp near-permutations, well-conditioned inverse) over the course of training.

Together, these keep P close to a permutation matrix throughout training, ensuring that the inverse is numerically stable and the canonicalization is interpretable.

#### Cross-family generalization via vocabulary intersection

The canonicalization module has a "vocabulary" — the row-indexing of W_down corresponds to specific tokens. For a different model family with a different tokenizer, the full vocabulary may not match. However, the module generalizes naturally:

1. Take the intersection of the canonicalization module's vocabulary with the new model family's vocabulary.
2. Select the corresponding rows of W_down and the corresponding rows of E.
3. Compute M = (1 / |intersection|) · E_intersect^T @ W_down_intersect.
4. Proceed as before. The normalization by |intersection| keeps magnitudes calibrated.

The intersection must contain at least d_model tokens for M to be full rank. In practice, 4–8× d_model tokens are needed for stability. For d_model=768, this means ~3K–6K shared tokens — easily met by the most common BPE tokens across any pair of Latin-script tokenizers.

Using the top-k most frequent tokens across many model families as the canonical vocabulary maximizes intersection coverage. Rare tokens contribute little signal anyway (their embeddings are typically undertrained), so restricting to common tokens may actually improve alignment quality.

The approach degrades gracefully: smaller intersections produce noisier M matrices and weaker alignment, with hard failure only when the intersection drops below d_model. This is the correct failure mode — two models with nearly zero shared vocabulary genuinely lack a natural shared coordinate system via embeddings.

#### Parameter cost

W_down: vocab_size × d_model. For a 50K vocabulary with d_model=768: ~38.6M parameters. This can be initialized from the top d_model right singular vectors of any reference embedding matrix for a strong starting point.

Canonicalization transformer (2 layers, 4 heads, d_model=768): ~14M parameters.

Temperature τ: 1 scalar.

Total: ~53M parameters. Substantial but a one-time cost shared across all target models, and the W_down matrix can be factored as W_down = A @ B (with A of shape vocab_size × r and B of shape r × d_model, r << d_model) if the parameter count needs to be reduced.

#### The full differentiable pipeline

```
Raw weights
  → Normalize scaling (W_V unit norm, absorb into W_O)
  → Extract embedding matrix E
  → M = (1/|vocab|) · Eᵀ @ W_down
  → Canonicalization transformer → attention logits
  → P = softmax(logits / τ)
  → Multiply every weight vector by P
  → Meta-transformer (EigenCritic or QuineFormer)
  → [QuineFormer only: multiply output vectors by P⁻¹]
  → Loss
  → Gradients flow end-to-end through entire pipeline
```

The canonicalization module is not preprocessing — it is the first stage of the meta-model, trained jointly by the downstream objective. It learns what canonical coordinate system makes the meta-model's task easiest.

---

## 4. Architecture Design Decisions (Shared by EigenCritic and QuineFormer)

Both EigenCritic and QuineFormer share the same foundational architecture — a bidirectional transformer that ingests serialized d_model vectors with structural metadata. They differ only in their output heads and training objectives.

### 4.1 Learned Input/Output Projection with Bias Absorption

Most weight vectors are natively d_model-dimensional, but several target model parameters have a natural per-vector bias scalar: each Q, K, V column pairs with one entry of the corresponding bias vector, and each MLP up-projection row pairs with one entry of b₁. Concatenating these gives (d_model+1)-dimensional vectors `[w; b]`. Vectors without a paired bias (O columns, MLP down-projection columns, embeddings, layer norms, standalone biases b_O and b₂) are padded: `[w; 0]`.

A shared learned projection `W_in` of shape `(d_model+1) × d_model` maps all vectors into the meta-transformer's working space. This has several properties:

1. **Meta-transformer width stays at d_model.** Using d_model+1 as the width would break multi-head divisibility (e.g. 769 is prime for GPT-2 Small).
2. **Bias information is mixed across all dimensions** rather than quarantined in a reserved slot. Information loss from discarding one dimension is negligible under a learned projection that discards the least informative direction.
3. **All vectors live in the same space.** Every vector — with or without a paired bias — passes through `W_in`, so attention dot products are geometrically coherent.
4. **Eliminates chunked bias vectors entirely.** The original design chunked the d_ff-dimensional b₁ into d_ff/d_model vectors, which arbitrarily bound unrelated neurons' biases together and violated the MLP neuron permutation symmetry that Section 4.4 is designed to preserve. Under bias absorption, each neuron's bias scalar travels with its own weight vector.

For QuineFormer's generation task, a corresponding `W_out` of shape `d_model × (d_model+1)` recovers the weight vector and bias scalar from each output position.

**Initialization:** `W_in = [I | 0]` (identity with a zero column for the bias input). `W_out = [I; 0ᵀ]` (identity on top, zero row on bottom). At step 0, the model ignores all biases and passes weight vectors through unchanged — identical to a no-projection baseline. Training discovers how much to deviate from this. An ablation comparing the learned projection against a frozen identity initialization quantifies how much the projection earns beyond bias absorption — e.g., adapting to distribution differences between component types or learning complementary normalization beyond what the canonicalization module (Section 3.3) provides.

**EigenCritic** adds a small readout head for regression (linear projection from a [CLS] token to a scalar). **QuineFormer** uses `W_out` at each position — the output vectors are the weights.

### 4.2 No Causal Attention Mask — Full Bidirectional Attention

Weight vectors have no temporal ordering. Layer 8's Q matrix isn't "after" layer 3's MLP in any causal sense — they coexist simultaneously. Full bidirectional attention is essential for cross-layer circuit detection.

This makes the architecture BERT-like, not GPT-like. For weight generation, this means iterative parallel refinement rather than autoregressive generation — arguably better suited since there's no natural left-to-right ordering of weight vectors.

### 4.3 Predict Full Weights, Not Deltas

The residual stream already computes `input + learned_residual`. If we train the model to predict **full target weights**, the model only needs to learn the residual correction in its internal parameters — exactly what transformers are optimized to do.

If we trained on deltas, the model would need to suppress the residual connection's identity contribution (learn `f(x) = delta - x`), wasting capacity fighting the architecture.

Additional benefit: when input weights are close to target (near convergence), the full-weights target is close to the input, so loss is naturally small and gradients are well-behaved.

### 4.4 Structural Metadata Encoding

#### The reconstruction principle

The metadata encoding must carry exactly enough information to reassemble a functionally equivalent transformer from the serialized vectors — no less, no more. If any tag is missing, there exists some weight configuration where reconstruction becomes ambiguous in a way that changes the model's function. If any tag is redundant, it encodes a distinction that doesn't exist, injecting noise and breaking a true symmetry.

#### Symmetry analysis

A transformer's parameters live on a 5-axis grid: `(layer, component_type, head_index, neuron_index, token_id)`. Each axis is either **ordered** (positions are non-interchangeable) or **symmetric** (positions are interchangeable under coordinated permutation):

| Axis | Symmetric? | Encode? | Reason |
|---|---|---|---|
| **Layer** | No | Yes | Layers are ordered; depth determines function. Swapping layer 3 and layer 7 changes the model. |
| **Component type** | No | Yes | Q ≠ K ≠ V ≠ O ≠ MLP_up ≠ MLP_down. Each serves a distinct computational role. |
| **Head index** | Yes | Group tag only | Heads are summed after projection; reordering heads (with their O blocks) leaves the output identical. |
| **Neuron index** | Yes | Group tag only | Within each head, permuting the d_head neuron indices (Q/K rows together, V/O rows together) preserves QK^T and VO products. Similarly for MLP neurons (up rows, down columns together). |
| **Token ID** | Yes* | No | Vocabulary ordering is arbitrary; embedding quality is geometric. (* Embeddings are not functionally symmetric — each maps to a specific word — but their quality is a property of collective geometry, not individual identity.) |
| **d_model coordinate** | Yes | Canonicalize | Permuting the residual stream basis across all weight vectors simultaneously preserves network function (elementwise nonlinearities commute with coordinate permutations). Unlike the within-layer symmetries above, this changes the numerical content of every token, so the meta-model cannot be architecturally invariant to it. Handled by the learned canonicalization module (Section 3.3). |

The key insight: **self-attention is natively permutation-equivariant.** If no positional encoding is added for an axis, the transformer automatically treats that axis as symmetric — permuting the input along that axis produces the same permutation in the output. This is the exact behavior we want for head indices and neuron indices. We don't need to *build* symmetry-awareness; we just need to *not break it* with unnecessary encodings.

#### Interaction groups

Head and neuron symmetry requires coordinated permutation — you can only swap head 3 with head 5 if you move *all* of head 3's parameters together. A K row from head 3 cannot be swapped with a K row from head 7; they participate in different attention computations (head 3's K rows are dotted with head 3's Q rows specifically). This means **group membership is functional information that must be encoded**, even though the group labels themselves are arbitrary.

The complete interaction structure per layer:

```
Layer (ordered, needs unique encoding)
├── Attention
│   ├── Head A (symmetric with other heads — needs group tag, not unique ID)
│   │   ├── QK neuron-pair 0: {Q_col_0 [+bias], K_col_0 [+bias]}   (symmetric with other QK pairs)
│   │   ├── QK neuron-pair 1: {Q_col_1 [+bias], K_col_1 [+bias]}
│   │   ├── VO neuron-pair 0: {V_col_0 [+bias], O_col_0}            (symmetric with other VO pairs)
│   │   ├── VO neuron-pair 1: {V_col_1 [+bias], O_col_1}
│   │   └── ...
│   ├── Head B (symmetric with Head A)
│   │   └── (same internal structure)
│   ├── b_O (standalone d_model vector, shared output-space offset)
│   └── ...
├── MLP
│   ├── Neuron 0: {up_row_0 [+bias], [gate_row_0,] down_col_0}  (symmetric with others)
│   ├── Neuron 1: {up_row_1 [+bias], [gate_row_1,] down_col_1}
│   ├── b₂ (standalone d_model vector, shared output-space offset)
│   └── ...
└── LayerNorms (unique, no internal symmetry)
```

`[+bias]` denotes vectors whose paired bias scalar is concatenated as a (d_model+1)-th dimension before input projection (see Section 4.1). Critically, each neuron's bias travels with its own weight vector, preserving permutation symmetry — the original design of chunking b₁ into d_model-sized vectors violated this by arbitrarily binding unrelated neurons' biases.

Note: within each head, QK and VO form **independent** permutation groups. Q dimension 3 pairs with K dimension 3, and V dimension 3 pairs with O dimension 3, but Q dimension 3 has no special relationship with V dimension 3.

#### The hybrid encoding design

Adding metadata as additive vectors to the weight content corrupts the content signal — the cross terms in attention dot products couple content with metadata in unpredictable ways. But MLP layers in the meta-transformer need *some* content-level signal to know what type of vector they're processing. The solution is a hybrid:

**Additive embeddings** for the two non-symmetric axes (layer + component type). These are few in number (~35 vectors), high-information, and trainable — the model can learn to place them in subspaces that minimally interfere with weight content. They give the MLP layers the context they need.

**Attention bias** for group membership (head tags, neuron pairing). These carry binary information ("same group" or "different group") and would be most damaging as additive modifications to content. A scalar bias added directly to the attention logits perfectly captures group membership without touching the weight vectors at all.

Concretely:

```
projected_i = [w_i; bias_i] @ W_in          // (d_model+1) → d_model
content_i   = projected_i + e_layer[layer_id] + e_type[component_type]

attention(i,j) = softmax(
    (content_i @ W_Q)(content_j @ W_K)^T / sqrt(d)
    + b_same_head[head_i == head_j]
    + b_same_neuron[neuron_i == neuron_j]
)
```

The attention biases `b_same_head` and `b_same_neuron` are learned scalars (per attention head in the meta-transformer). They nudge attention toward or away from within-group pairs without modifying the content vectors. During training, the assignment of head and neuron group labels is randomly permuted per example — the model learns "shared label = same interaction group" without learning "label 5 = induction head."

#### Parameter count

- **Canonicalization module (Section 3.3):** W_down: vocab_size × d_model (~38.6M for GPT-2 vocabulary). Canonicalization transformer (2 layers, 4 heads): ~14M. Temperature τ: 1 scalar. Total: ~53M.
- **Input/output projections:** W_in: (d_model+1) × d_model. W_out: d_model × (d_model+1) (QuineFormer only). ~591K parameters each for d_model=768.
- **Additive embeddings:** 15 layer + 20 component type = 35 learned vectors of dimension d_model (~27K)
- **Attention biases:** 2 scalars per attention head in the meta-transformer (same_head, same_neuron). For a meta-transformer with 4 layers × 4 heads = 16 heads: 32 scalars.
- **Total metadata + canonicalization parameters (EigenCritic):** ~53.6M. **(QuineFormer):** ~54.2M.

The canonicalization module dominates, driven by the W_down projection matrix. This can be factored (W_down = A @ B with intermediate rank r << d_model) if the parameter budget is constrained. The structural metadata parameters remain ~200× smaller than naive unique-per-position encoding.

#### Why this is not DeepSets

DeepSets is permutation-invariant over the *entire* input. Our model is permutation-invariant only along the symmetric axes (head index, neuron index). Along the non-symmetric axes (layer, component type), elements interact richly through multi-layer self-attention. Q-vectors from layer 5 attend to K-vectors from layer 3, MLP vectors from layer 8, etc. The attention is content-dependent and cross-group, with multi-hop reasoning across meta-transformer layers. DeepSets processes each element independently before aggregation; our model lets all vectors talk to each other.

#### Architectural consequence: neuron permutation augmentation is unnecessary

With this encoding, the architecture is inherently equivariant to neuron and head permutations. There are no position-dependent features to destroy because the encoding never distinguishes neuron or head positions. This eliminates Category 2 augmentation (random neuron permutation) entirely — baking symmetry into the architecture is strictly better than training it away with data augmentation.

### 4.5 Attention Efficiency

At GPT-2 scale: 162K² ≈ 26 billion attention scores per layer. For larger models, structured sparse attention is needed. The structural metadata provides natural sparsity: dense attention within same layer/matrix, sparse attention across layers.

---

## 5. First Experiment: EigenCritic — Validation Loss Prediction

### 5.1 Motivation

EigenCritic validates the readability of weight space before we attempt to write to it with QuineFormer. If a meta-transformer can learn to predict validation loss from weight vectors — and does so via structural features rather than simple statistics — this is evidence that "intelligence patterns" exist in weight space.

### 5.2 Critical Design Requirements

**Diversity of training data:** Same architecture trained on multiple data domains (Shakespeare, code, Wikipedia, multilingual, etc.). Variation across learning rates, batch sizes, optimizers, random seeds.

**Strong baselines:** Linear regression and random forest on ~20 simple weight statistics (per-layer norms, spectral norms, means, variances, kurtosis, sparsity). The meta-transformer must beat these, and beat them in ways attributable to structural attention patterns.

**Two evaluation protocols:**
- **Weak test:** Hold out random 20% of checkpoints (same data sources)
- **Strong test:** Hold out entire data source (e.g., train on 4 domains, test on code). This is the real result.

### 5.3 Pitfalls to Avoid

1. **Trivial baselines:** Simple weight statistics predict loss well. Unterthiner et al. (2020) showed this. Must beat them convincingly.
2. **Learning "training progress" not "intelligence":** A model that learns "these weights look like step 8000" predicts loss without learning anything structural. Mitigate with diverse hyperparameter configurations that reach the same loss at different steps.
3. **"Task-independent" is where the claim lives or dies:** Must demonstrate cross-domain transfer, not just in-distribution prediction.

---

## 6. Data Augmentation Strategy (EigenCritic Training)

### Principle

Design augmentations that are **adversarial against trivial baselines**: transformations that change simple statistics while preserving (or controllably destroying) functional structure. These ensure EigenCritic learns complex structural patterns, not simple heuristics.

### 6.1 Label-Preserving Augmentations (Always On)

**Per-head value/output rescaling (α ~ LogUniform[0.1, 10]):**
Within each attention head, scale W_V → αW_V and W_O → (1/α)W_O. The head's output contribution is W_O · softmax(QK⊤/√d) · W_V · x, so the α factors cancel exactly at the head output — before reaching any skip connection, LayerNorm, or other head. This is a genuine per-head gauge symmetry (one free scalar per head). It destroys per-matrix norms, spectral norms, and weight magnitude statistics while preserving the network's function exactly.

*Note: a whole-layer rescaling (multiply all weights by α, rely on LayerNorm absorption) does NOT work. Skip connections pass unscaled activations around LayerNorm, and attention logits scale as α², so the network's function changes. The W_V/W_O form is the narrowest exact invariance available.*

*Note: the scaling normalization step in the canonicalization module (Section 3.3) normalizes W_V to unit Frobenius norm before the meta-model sees the data. The augmentation re-introduces scaling variation at training time, teaching the model to be robust to any residual scaling mismatch. Both steps are complementary: normalization cleans the input, augmentation ensures robustness.*

**Random permutation of group labels:**
Head-tag and neuron-tag assignments are shuffled randomly per training example. The model learns "same tag = same interaction group" but cannot memorize which tag corresponds to which head or neuron. (Note: neuron/head *permutation of actual weight vectors* need not be applied as a data augmentation because the architecture is natively equivariant to these permutations — see Section 4.4.)

### 6.2 Structural Negatives (Mixed Into Training Set)

**Shuffled-weight copies:**
Take a trained model, shuffle weight values within each matrix randomly. Preserves all marginal statistics (mean, variance, norms, kurtosis, sparsity) but destroys all relational structure. Val loss goes to random chance. Label with actual (terrible) val loss. This is the perfect hard negative: identical simple features, destroyed complex features, catastrophically different performance.

**Norm-matched random models:**
Generate random weights scaled to match real models' per-layer statistics. Val loss is terrible. Any predictor relying on simple statistics assigns the same score to both real and norm-matched models.

### 6.3 Controlled Damage (Additional Labeled Data)

**Random orthogonal rotation of the residual stream:**
Apply a random orthogonal matrix R to every weight vector in the target model, transforming directions in the residual stream. This preserves all dot products between interacting weight vectors (norms, angles), so simple statistics look identical. However, elementwise nonlinearities (GELU, LayerNorm's learned gain/bias) break rotational invariance: `GELU(Rx) ≠ R·GELU(x)` because the activation treats each coordinate independently, and rotating moves values across coordinate boundaries. The result is a network with identical weight statistics but measurably different (usually worse) performance. Forward-pass the rotated network to obtain the actual val loss as the new label. This cheaply manufactures diverse training pairs — one per rotation per model — with honestly re-measured labels, and provides hard examples where norm-based baselines cannot detect the damage.

*Note: for RoPE-based models, R must be block-diagonal respecting the RoPE pair structure. For learned positional embeddings (GPT-2 style), R can be full if positional embeddings are also rotated. The intersection of constraints from RoPE, GELU, and LayerNorm collapses general rotations to neuron permutations (already handled architecturally — see Section 4.4), which is why this augmentation requires re-measured labels rather than being label-preserving.*

**Head ablation variants:** Zero out individual attention heads (set O_proj to zero). Measure actual val loss. Teaches the predictor that specific cross-layer structures matter.

**Per-layer Gaussian noise with measured impact:** Add N(0, σ) to one layer at a time, evaluate loss. Vary which layer and σ. Teaches layer-level importance.

**Weight interpolation between good and bad models:** `W = α * W_good + (1-α) * W_bad`. Evaluate at each α. Creates smooth trajectories with known performance.

**Rank truncation per matrix:** SVD → zero out smallest singular values. Evaluate loss. Directly teaches about effective rank vs. performance.

### 6.4 Augmentation Ablation

Run experiments with each augmentation independently enabled/disabled. If a particular augmentation *hurts* EigenCritic's cross-domain performance, that tells us something interesting about what features actually predict generalization. This analysis is itself part of the scientific contribution.

---

## 7. Future Direction: QuineFormer — Weight Generation & Trajectory Prediction

### 7.1 The Vision

Where EigenCritic reads weights and predicts a scalar, QuineFormer reads weights and **outputs improved weights**. This is the training-trajectory prediction problem — learning "how models improve" rather than "what good weights look like."

### 7.2 Three Tiers of Success

**Tier 1 — Interpolation:** Given an undertrained model, predict what continued SGD would have done. Useful for training acceleration but not revolutionary.

**Tier 2 — Cross-run transfer:** Model recognizes that a "converged at loss 0.10" state looks structurally similar to an intermediate state from another run that eventually reached loss 0.03. Predicts the continuation. Genuine generalization.

**Tier 3 — True extrapolation:** Predicts productive weight updates beyond anything in its training distribution. Would require learning the causal structure of why certain weight changes improve performance. This is the holy grail.

### 7.3 Denoising vs. Trajectory Framing

Existing diffusion approaches learn: **noise → good weights** (static distribution).
The trajectory approach learns: **weights at step t → weights at step t+k** (dynamic process).

The second learns "what learning itself looks like" — an operator that maps worse models to better models. If this operator generalizes, it could be applied iteratively past convergence, effectively learning a better optimizer than gradient descent.

### 7.4 Why This Could Work

- SGD is local (gradient at current point only); a learned update operator could be non-local.
- SGD is memoryless (even Adam's momentum is shallow); a trajectory model could learn deep temporal patterns.
- SGD is bounded by the training data; a model trained on diverse trajectories could learn general patterns of "how neural networks improve."

### 7.5 Connection to Existing Weight Generation Architectures

For the generation task, the QuineFormer architecture naturally supports iterative refinement: feed in current weights → get out updated weights → feed those back in → repeat. The bidirectional attention and full-weight prediction (not deltas) design decisions from Section 4 directly support this. QuineFormer shares EigenCritic's backbone but replaces the [CLS] regression head with `W_out` projections — the output at each position is mapped back to (d_model+1) dimensions, recovering the weight vector and its paired bias scalar (see Section 4.1). Output vectors are then multiplied by P⁻¹ (Section 3.3) to map from canonical space back to the original model's coordinate system.

**Open question for iterative refinement:** the canonicalization module computes P from the input model's embedding matrix. When feeding output weights back as input for a second refinement pass, should P be recomputed from the new model's embeddings (which have changed), or held fixed from the first pass? Recomputing is more principled but adds cost; holding fixed assumes the coordinate system hasn't shifted too much in one update step. Worth testing both.

---

## 8. Future Direction: EigenCritic as Learned Regularization Signal

### 8.1 Concept

Use the trained EigenCritic as a differentiable regularizer during training of new models. Two gradient sources: the usual task loss (from data) and EigenCritic's opinion of weight quality (from learned structural priors).

### 8.2 Promising Applications

**Initialization:** Use EigenCritic gradients to pre-shape random weights before training begins, without seeing training data. Could produce initializations better than Kaiming/Xavier.

**Reward model for QuineFormer:** Score candidate weight updates with EigenCritic, use scores as training signal for QuineFormer's generator. This is "RLHF for weight space" — directly connecting the first paper (EigenCritic) to the second paper (QuineFormer).

### 8.3 Risks

- **Mode collapse:** Model could learn to produce weights that "look good" to EigenCritic without actually performing well (analogous to GAN mode collapse).
- **Distribution shift:** Once EigenCritic gradients modify training, weights enter a distribution the predictor has never seen.
- **Fighting the task loss:** EigenCritic may penalize intermediate weight configurations that are necessary waypoints toward good solutions.

### 8.4 Mitigations

- Small regularization coefficient (λ = 0.01–0.1), possibly annealed.
- Apply infrequently (every 100-1000 steps).
- Freeze EigenCritic during target model training.
- Validate against standard regularizers on equal compute budgets.

---

## 9. Hardware-Constrained Proof of Concept (EigenCritic)

### 9.1 Hardware

NVIDIA RTX 3060, 12GB VRAM.

### 9.2 Target Model

4-layer transformer, d_model=128, 4 heads, d_ff=512, character-level vocab (256 tokens).
- Parameters: ~820K
- Serializes to: **~6,400 vectors of dim 128**
- Memory for input: ~2 MB (fp16)

### 9.3 EigenCritic Architecture

Bidirectional transformer, d_model=128, 4 layers, 4 heads. Learned input projection W_in: 129 → 128 (bias absorption, initialized to [I | 0]). No causal mask. [CLS] token → linear → scalar for regression.

Learned canonicalization module (Section 3.3): W_down (256 × 128 = 32K parameters), shallow canonicalization transformer (2 layers, 4 heads, d=128, ~400K parameters), learned temperature τ. Produces a 128 × 128 soft permutation matrix P applied to all weight vectors before they enter the main meta-transformer.

Structural metadata: additive learned embeddings for layer index (4 layer + 3 global = 7 embeddings) and component type (~12 types). Attention bias scalars for head group membership and neuron group membership (2 scalars per meta-transformer attention head). Group labels randomly permuted each training example.

- Meta-transformer parameters: ~800K
- Canonicalization module parameters: ~430K
- Total parameters: ~1.2M
- Attention memory (6.4K² × 4 layers × 4 heads): ~1.6 GB
- **Total estimated: ~2 GB. Fits easily on 3060.**

### 9.4 Dataset Generation

**5 data sources** (same architecture, different data):
1. TinyShakespeare — character-level English literature
2. TinyStories — simple English stories
3. Code (Python subset) — from The Stack
4. Wikipedia (Simple English)
5. Multilingual (de/fr/es)

**Variation axes per data source:**
- Learning rates: [1e-4, 3e-4, 1e-3, 3e-3]
- Random seeds: 5 per configuration
- Checkpoints: every 50 steps, 0–2000 (40 per run)
- Batch sizes: [16, 32, 64]
- Optimizers: [Adam, SGD+momentum]

**~100 runs per source × 5 sources = 500 runs × 40 checkpoints = 20,000 training examples.**

Storage: ~36 GB in fp16. Load on-the-fly from disk during training.

**Preprocessing (offline, after all checkpoints are generated):**
Normalize per-head W_V to unit Frobenius norm and absorb the scaling factor into W_O for all checkpoints. This removes the continuous scaling degeneracy (Section 3.3). Residual stream coordinate canonicalization is handled online by the learned canonicalization module as the first stage of the meta-model forward pass.

**Each checkpoint evaluated on:**
1. Val loss on its own data source (in-distribution)
2. Val loss on a shared cross-domain eval set

### 9.5 Baselines

1. Linear regression on 20 weight statistics (norms, spectral values, means, variances per layer)
2. Random forest on same features
3. MLP on per-layer summary statistics concatenated
4. EigenCritic WITHOUT augmentations (ablation)
5. EigenCritic WITH augmentations (full model)
6. EigenCritic with canonicalization module frozen to identity (ablation — quantifies the value of learned coordinate alignment)

### 9.6 Train/Test Protocol

- **Weak test:** Hold out random 20% of checkpoints (all data sources present in train)
- **Strong test:** Hold out entire data source (e.g., train on 4, test on code)

### 9.7 Success Criteria

| Level | Requirement |
|---|---|
| **Minimum** | EigenCritic > all baselines on weak test |
| **Good** | EigenCritic > baselines on strong test (cross-domain) |
| **Great** | Above + attention analysis shows meaningful cross-layer patterns |
| **Publishable** | All of above + augmentation ablation showing each augmentation independently helps on strong test |

### 9.8 Timeline

| Day | Task |
|---|---|
| 1 | Implement target model + checkpoint saving. Run sanity check: 50 models, linear baseline. |
| 2 | Generate full checkpoint dataset (500 runs). Apply W_V/W_O scaling normalization. Implement baselines (linear, RF, MLP). |
| 3 | Implement EigenCritic: canonicalization module + bidirectional transformer, d=128, 4L/4H, structural metadata, [CLS] + regression head. |
| 4 | Train EigenCritic without augmentations. Compare to baselines. Debug. |
| 5 | Add augmentations one at a time. Ablation study (including frozen-canonicalization ablation). |
| 6 | Strong test (held-out data source). Attention analysis / visualization. |
| 7 | Write up results. |

### 9.9 Day 1 Sanity Check (Do This First)

Train 50 copies of the tiny transformer on TinyShakespeare with varied learning rates and seeds. Save final checkpoints. Compute 20 simple statistics per checkpoint. Fit linear regression → val loss.
- If R² > 0.9: baselines are very strong, augmentations are critical.
- If R² < 0.5: the problem may be too noisy, reconsider experimental design.
- Takes 30 minutes. Don't build EigenCritic until you've seen these numbers.

### 9.10 Implementation Notes

- Use nanoGPT (Karpathy) as base for target model: clean, minimal, well-tested at small scales.
- Character-level tokenizer shared across all data sources (avoids BPE alignment issues).
- fp16 mixed precision throughout (essential on 3060).
- Gradient checkpointing if memory is tight.
- Batch size for meta-transformer: 1–4 models per batch.
- AdamW with cosine schedule, ~50–100 epochs over dataset.

---

## 10. Naming & Identity

**QuineFormer** — a transformer that takes a transformer as input and outputs a transformer.

- **Quine:** A program that outputs its own source code. Self-referential computation.
- **Former:** Transformer lineage.
- **Immediately parseable:** Anyone in CS knows both halves.
- **Says what it does:** Self-referential computation in a shared representational space.
- **Googleable:** Zero prior hits.

```
mkdir quineformer
```

---

## 11. References

### Diffusion over Neural Network Weights
- Wang, K. et al. (2024). "Neural Network Parameter Diffusion (p-diff)." arXiv:2402.13144
- Soro, B. et al. (2024). "Diffusion-Based Neural Network Weights Generation (D2NWG)." arXiv:2402.18153, ICLR 2025
- Gong, Y. et al. (2024). "Efficient Training with Denoised Neural Weights." ECCV 2024. arXiv:2407.11966
- NeurIPS 2024. "Weight Diffusion for Future (W-Diff)."

### Weight Nowcasting / Forecasting
- Sinha, A. et al. (2017). "Introspection Network."
- Jang, J. et al. (2023). "Learning to Boost Training by Periodic Nowcasting Near Future Weights (WNN)." ICML 2023
- Knyazev, B. et al. (2025). "Accelerating Training with Neuron Interaction and Nowcasting Networks (NiNo)." ICLR 2025. arXiv:2409.04434
- Shou, X. et al. (2024). "Less is More: Efficient Weight Farcasting with 1-Layer Neural Network (LFD-2)." arXiv:2505.02714
- arXiv:2505.20221 (May 2025). "Gradient Flow Matching (GFM)."
- Jang, J. et al. (2025). "Learning from Oblivion: Predicting Knowledge Overflowed Weights via Retrodiction of Forgetting (KNOWN)." arXiv:2508.05059

### Transformer-Based Weight Generation
- Peebles, W. et al. (2022). "Learning to Learn with Generative Models of Neural Network Checkpoints (G.pt)." arXiv:2209.12892
- Zhmoginov, A. et al. (2022). "HyperTransformer." ICML 2022
- Müller, A. et al. (2023). "MotherNet: Fast Training and Inference via Hyper-Network Transformers." arXiv:2312.08598
- Wang, H. et al. (2024). "TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters." ICLR 2025 Spotlight. arXiv:2410.23168
- arXiv:2501.11587 (2025). "Scaling Up Parameter Generation: A Recurrent Diffusion Approach."

### Weight Space as Data Modality
- Kofinas, M. et al. (2024). "Graph Metanetworks for Processing Diverse Neural Architectures." arXiv:2312.04501
- ICLR 2025 Workshop. "Neural Network Weights as a New Data Modality."
- Unterthiner, T. et al. (2020). "Predicting Neural Network Accuracy from Weights." arXiv:2002.11448
- Ying, H. et al. (2024). "Enhancing deep neural network training efficiency and performance through linear prediction." Scientific Reports.

### Mechanistic Interpretability
- Elhage, N. et al. (2021). "A Mathematical Framework for Transformer Circuits." Transformer Circuits Thread.
- Dar, G. et al. (2022). "Analyzing Transformers in Embedding Space."
- Geva, M. et al. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories."

### Learned Optimizers
- Andrychowicz, M. et al. (2016). "Learning to learn by gradient descent by gradient descent."
- Metz, L. et al. (2022). "VeLO: Training Versatile Learned Optimizers."
- Thérien, B. et al. (2024). "μLO: Compute-Efficient Meta-Generalization of Learned Optimizers." arXiv:2406.00153

### Architecture References
- Peebles, W. & Xie, S. (2022). "Scalable Diffusion Models with Transformers (DiT)." arXiv:2212.09748
- Ha, D. et al. (2016). "HyperNetworks." arXiv:1609.09106
- Karpathy, A. "nanoGPT." https://github.com/karpathy/nanoGPT

### Weight-Space Canonicalization
- Ainsworth, S. et al. (2023). "Git Re-Basin: Merging Models Modulo Permutation Symmetries." ICLR 2023. arXiv:2209.04836
- Dar, G. et al. (2022). "Analyzing Transformers in Embedding Space." arXiv:2209.02535
- Boufalis, O. et al. (2025). "Symmetry-Aware Graph Metanetwork Autoencoders: Model Merging through Parameter Canonicalization." TAG-DS 2025. arXiv:2511.12601

---

*Last updated: March 2026. Research directory: `quineformer/`. Revision in progress.*