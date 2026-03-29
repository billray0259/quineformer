# Canonicalization Experiment V3

## Goal

Use the RotationSymmetry matching algorithm as a closed-form shared-reference canonicalization baseline for MultiBERT.

Unlike V1 and V2, V3 does not train a learned canonicalization module. Instead, it:

1. picks one train-side MultiBERT seed as a shared reference basis,
2. matches every other model to that reference using closed-form FFN permutation and attention rotation matching,
3. interpolates the matched models directly in native parameter space, and
4. evaluates whether the matched interpolation is more functional than naive interpolation.

## Relation To Prior Experiments

- V1 and V2 learn a model-level canonicalizer from embeddings and optimize interpolation behavior through training.
- V3 removes training entirely and tests whether local transformer symmetries alone are enough to make interpolation between MultiBERT seeds materially better.
- V3 is therefore a baseline against the learned canonicalization line, not a continuation of the Sinkhorn-based module.

## Method

### Shared Reference Basis

The experiment chooses one seed from the train split as a fixed reference basis. Every other seed is matched to that basis.

The default reference selection rule follows the upstream library more closely: choose the train-side seed whose projected parameters are closest to the average projected model over seeds 0-19. The seed can also be fixed manually.

### FFN Matching

For each encoder-layer FFN, solve the same Hungarian assignment objective used in the RotationSymmetry implementation:

- align `intermediate.dense.weight`, `intermediate.dense.bias`, and `output.dense.weight`
- permute hidden FFN neurons into the reference ordering

### Attention Matching

For each attention head independently:

- solve a closed-form orthogonal Procrustes problem for the Q/K block
- solve a closed-form orthogonal Procrustes problem for the V/O block
- optionally apply the paper's post-rotation rescaling step

LayerNorms, embeddings, and the MLM head are left unchanged in V3.

## Evaluation

### Primary Comparison

On held-out pairs from seeds 20-24, compare:

1. naive midpoint interpolation,
2. shared-reference matched midpoint interpolation, and
3. endpoint ensemble baseline.

The main metric is MLM perplexity on the same fixed WikiText validation batches used in the canonicalization experiments.

### Secondary Diagnostics

- parameter distance to the reference before and after matching,
- interpolation curves over alpha,
- attention-only / FFN-only ablations,
- with and without rescaling,
- subset-of-layers matching via `--layer-indices`.

## Outputs

- `results_v3/reference_choice.json`
- `results_v3/distance_summary.json`
- `results_v3/perplexity_comparison.json`
- `results_v3/interpolation_curves.json`
- optional cached matched serializations in `results_v3/matched_serialized/`

## Command

```bash
python experiments/canonicalization/run_v3.py
```

Smoke test:

```bash
python experiments/canonicalization/run_v3.py --smoke-test
```