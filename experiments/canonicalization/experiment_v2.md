# Canonicalization Experiment V2

## Goal

Train the simplified `CanonicalizationModule` with a layer-local surrogate loss instead of a full interpolated-model forward on every step.

The canonicalization is still model-level: each endpoint model gets one permutation-like matrix `P`. The difference is that training supervises only one sampled encoder layer at a time.

## Core Idea

For a sampled pair of models `(i, j)` and a sampled encoder layer `l`:

1. Compute `P_i` and `P_j` from the endpoint word embeddings.
2. Canonicalize the input and output residual streams of layer `l` for both endpoint models.
3. Canonicalize that layer's absorbed parameter rows, interpolate them in canonical space, and map the interpolated layer back into model `i`'s basis for execution.
4. Run only the sampled `BertLayer` on the interpolated residual input.
5. Canonicalize the predicted output and compare it to the interpolated canonical endpoint output with MSE.

This keeps the optimization target tied to activation behavior while avoiding a full BERT forward for the interpolated model on every training step.

## Training Loss

Let `h_i^l` and `h_i^(l+1)` be the cached endpoint residual states on a fixed reference batch.

We define:

- canonical input: `(1 - alpha) * (h_i^l @ P_i) + alpha * (h_j^l @ P_j)`
- canonical target output: `(1 - alpha) * (h_i^(l+1) @ P_i) + alpha * (h_j^(l+1) @ P_j)`

The interpolated sampled layer is executed in model `i`'s basis, then its output is mapped back to canonical space before the loss:

`loss = mse(predicted_canonical_output, target_canonical_output) + lambda_sharp * entropy(P)`

## Minimal Scope

`run_v2.py` intentionally omits:

- W&B logging
- checkpoint sweeps
- interpolation curves
- dynamic imports from `run_v1.py`
- full-model MLM loss in the training objective

It keeps only:

- one best-checkpoint save
- a compact JSON training log
- a small post-training perplexity check on held-out pairs

## Outputs

- `results_v2/canonicalization_module.pt`
- `results_v2/training_log.json`
- `results_v2/perplexity_results.json`