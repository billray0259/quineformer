# Soft-to-Hard Sharpness Note

## Context

The current `run_v2.py` training objective uses a sharpness regularizer based on permutation entropy:

`loss_total = loss_pred + lambda_sharp * loss_sharp`

where `loss_sharp` encourages the Sinkhorn transport matrices to become low-entropy and therefore more permutation-like.

However, the current experiment also manually schedules `tau`, which already pushes the soft transport matrix `P` toward sharper assignments over training.

That raises a question:

- if `tau` is already manually annealed
- and evaluation uses a hardened permutation produced by Hungarian projection

then entropy may not be the most direct sharpness objective.

## Idea

Instead of penalizing entropy, define sharpness by how much the soft permutation changes when it is hardened.

Let:

- `P` be the soft Sinkhorn matrix
- `H(P)` be the hard permutation returned by Hungarian projection

The proposed sharpness penalty is:

$$
L_{\text{hard-gap}} = \frac{1}{d^2} \| P - H(P) \|_F^2
$$

with `H(P)` treated as a detached target.

In words:

- build the hard permutation from the current soft matrix
- stop gradients through the hardening step
- penalize the distance between the soft matrix and its hardened version

## Motivation

This objective is appealing because it matches the actual train/eval mismatch more directly than entropy does.

Current evaluation already hardens the learned permutation before reconstructing interpolated models. If the soft matrix and hard matrix are very different, then training is optimizing one object while evaluation uses another.

This proposal tries to reduce that gap directly.

## Sketch

Conceptually:

```python
hard_i = project_to_hard_permutation(P_i.detach())
hard_j = project_to_hard_permutation(P_j.detach())

loss_sharp = 0.5 * (
    F.mse_loss(P_i, hard_i) +
    F.mse_loss(P_j, hard_j)
)
```

This would replace the entropy-based sharpness loss, not necessarily be added on top of it.

## Why It Might Help

- It directly encourages stability under hardening.
- It aligns training pressure with the exact object used at evaluation time.
- It may be easier to interpret than entropy because the target is concrete: the nearest hard permutation.

## Risks

- The Hungarian projection is discrete, so the detached target can jump abruptly when the best assignment changes.
- Early in training, when `P` is diffuse, the hard target may be noisy or misleading.
- If weighted too strongly, this loss can still overpower the prediction loss, just like entropy can.

## Possible Training Strategy

If this idea is tested later, it probably makes sense to use it conservatively:

- warm up with prediction loss only
- enable the soft-to-hard penalty only after `P` has some structure
- start with a small `lambda_sharp`
- compare against `entropy` and `none` as clean ablations

## Variants Worth Comparing

1. `none`
2. `entropy`
3. `soft_to_hard_gap`
4. `orthogonality`, for example penalizing `||P P^T - I||_F^2`

The soft-to-hard gap is the most directly tied to current evaluation, while orthogonality would be a smoother surrogate that avoids the discrete Hungarian target.

## What To Watch If Implemented

- `val_ppl_ratio`
- `val_roundtrip_ppl_ratio`
- mean `||P - H(P)||`
- row entropy of `P`
- whether the best checkpoint under perplexity occurs earlier or later than under surrogate loss

## Summary

The core intuition is simple:

- entropy says "make P sharp"
- soft-to-hard gap says "make P close to the permutation we actually use at eval time"

This may be a better sharpness objective in a setting where `tau` is already scheduled manually and hardened permutations are used downstream.