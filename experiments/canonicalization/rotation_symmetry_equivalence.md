# RotationSymmetry Equivalence Check

## Goal

Verify that the RotationSymmetry adaptation in this repo produces the same matched tensors and the same matched MultiBERT models as the upstream library when the anchor seed is fixed.

## What It Compares

The verification harness compares three levels:

1. per-layer primitive outputs for
   - FFN matching,
   - attention Q/K matching,
   - attention V/O matching,
2. full matched-model BERT state dicts after applying the same fixed-anchor procedure, and
3. MLM losses for matched endpoints and their midpoint interpolation.

## Scope

This experiment verifies the matching implementation under a fixed anchor seed.

It does not try to prove equivalence of anchor-selection logic, because V3 adapts the upstream merger to a shared-reference MultiBERT canonicalization setting rather than reusing their full merging framework unchanged.

## Command

```bash
python experiments/canonicalization/verify_rotation_symmetry_equivalence.py \
  --upstream-repo /tmp/RotationSymmetry \
  --anchor-seed 3 \
  --local-seeds 20 21 \
  --layer-indices 0 5 11
```

Smoke test:

```bash
python experiments/canonicalization/verify_rotation_symmetry_equivalence.py \
  --upstream-repo /tmp/RotationSymmetry \
  --smoke-test
```

## Output

The experiment writes:

- `results_v3_equivalence/equivalence_report.json`

The report includes per-layer tensor diffs, model-level state-dict diffs, and endpoint plus midpoint MLM-loss diffs.