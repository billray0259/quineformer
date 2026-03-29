# Head-Local Gauge Fixer Note

## Motivation

The current canonicalization approach focuses on a global residual-stream permutation-like matrix `P`.

That addresses one important symmetry:

- different models may use different residual-stream coordinate orderings

However, this is probably not the only exact or near-exact symmetry that makes model parameters hard to compare.

In particular, an attention head has additional internal change-of-basis freedoms in its Q/K and V/O factorizations. Two heads can implement the same computation while using different internal coordinate systems.

This suggests that global canonicalization alone may be insufficient. We may also need a head-local gauge fixer.

## What Is a Gauge Fixer?

A gauge fixer chooses one canonical representative from a family of parameterizations that compute the same function.

If many parameter settings are functionally equivalent, then the model has gauge freedom. A gauge fixer removes that freedom by imposing a deterministic convention.

The core idea here is:

- identify the exact symmetry family
- transform each head into a canonical representative
- compare and interpolate only after this canonicalization

This could improve:

- interpolation quality
- cross-seed consistency
- numerical stability
- interpretability of canonicalized heads

## Q/K Symmetry

For one attention head, logits depend on dot products between queries and keys.

If activations are written as right-multiplied row vectors, then for any invertible matrix `A` in `GL(d_head)` we can transform a head as:

$$
W_Q' = W_Q A^{-T}, \qquad W_K' = W_K A
$$

This preserves the QK dot products, because the inverse transformations cancel inside the bilinear form.

So the internal basis of one head's Q/K coordinates is not unique. There is a whole family of equivalent parameterizations.

A global residual permutation does not remove this freedom.

## V/O Symmetry

The same issue appears for values and outputs.

For any invertible matrix `B` in `GL(d_head)`, we can transform:

$$
W_V' = W_V B, \qquad W_O' = B^{-1} W_O
$$

up to row/column convention.

This preserves the head output because the value coordinates are changed internally and then undone at the output projection.

Again, this is a head-local exact symmetry that global residual canonicalization does not remove.

## Why This Matters

Two models may already be aligned in the global residual-stream basis, but their heads can still use different internal Q/K and V/O bases.

That means:

- functionally similar heads can still look very different in raw parameters
- interpolation across seeds can remain poor
- QuineFormer may see artificially multimodal targets
- the current canonicalization module may be solving only part of the nuisance variation

In short:

- global `P` handles outer residual coordinate ambiguity
- a head-local gauge fixer would handle inner head-factorization ambiguity

## Deterministic Rule vs Learned Gauge Fixer

There are two broad approaches.

### 1. Deterministic Rule

Use an analytic rule to choose a canonical representative.

Examples:

- whitening a head-local covariance
- SVD or polar decomposition with sorted singular values
- sign conventions
- alignment to fixed probe directions
- canonicalization after global `P` is applied

Pros:

- stable
- interpretable
- harder to overfit
- easier to debug

Cons:

- optimizes a hand-designed proxy
- may not choose the representation that best helps interpolation or generation

### 2. Learned Gauge Fixer

Use a small network or learned module to predict the head-local gauge transform from the parameters.

Pros:

- can optimize the representation that best helps the downstream task
- more flexible than a hand-designed rule

Cons:

- easier to destabilize
- harder to interpret
- larger search space
- may require strong regularization

A good development path is probably:

1. deterministic baseline first
2. learned version only if the deterministic rule helps but is not enough

## Conditioning on the Parameters

A useful gauge fixer almost certainly needs to depend on the parameters.

A fixed matrix shared across all heads and all models would usually just define another arbitrary basis, not a canonicalization.

The right pattern is model-dependent and head-dependent:

- inspect the current head parameters
- infer which member of the symmetry family this head is using
- map it to a shared representative

This is the same pattern used by the current global canonicalization module, which predicts a model-specific `P` from the embedding matrix.

So the head-local version would likely look like:

$$
A_h = g_{QK}(\phi_{QK,h}), \qquad B_h = g_{VO}(\phi_{VO,h})
$$

where:

- `A_h` is the Q/K gauge for head `h`
- `B_h` is the V/O gauge for head `h`
- `phi` are features derived from the current head parameters
- `g` is either an analytic procedure or a small learned module

## How Global P Can Help

The global canonicalization matrix `P` can be used first, and the head-local gauge rule can then operate in the already globally canonicalized residual basis.

This is appealing because it separates the problem into two stages:

1. global residual alignment
2. head-local internal gauge fixing

A practical pipeline could be:

- apply global `P`
- extract globally canonicalized Q/K/V/O head parameters
- compute a head-local canonical representative from those transformed parameters
- pass the fully canonicalized parameters to the downstream model

This is likely better than running the head-local rule directly on raw parameters, because the outer residual ambiguity has already been removed.

## Candidate Parameterizations

If we want a learned head-local gauge, the main design question is how to parameterize the invertible matrices.

### Diagonal Gauge

The simplest option:

$$
A_h = \mathrm{diag}(\exp(a_h)), \qquad B_h = \mathrm{diag}(\exp(b_h))
$$

where `a_h` and `b_h` are free vectors in `R^{d_head}`.

Pros:

- always invertible
- cheap
- stable
- extends the current scalar rescaling idea to per-coordinate scaling

Cons:

- cannot mix coordinates
- weaker than a full basis change

This is the safest first experiment.

### Orthogonal Gauge

Parameterize a skew-symmetric matrix and exponentiate it:

$$
S_h = -S_h^T, \qquad A_h = \exp(S_h)
$$

and similarly for `B_h`.

Pros:

- always invertible
- well-conditioned
- supports basis rotations

Cons:

- no anisotropic scaling

This is a good medium-complexity option.

### Orthogonal + Diagonal Gauge

Combine the two:

$$
A_h = Q_h \mathrm{diag}(\exp(a_h)), \qquad B_h = R_h \mathrm{diag}(\exp(b_h))
$$

Pros:

- richer than either alone
- still more controlled than full `GL(d_head)`

This may be the best tradeoff if diagonal-only is too weak.

### Full Invertible Gauge

Use a full `GL(d_head)` parameterization via LU, QR, polar, or triangular factors.

Pros:

- most expressive
- can represent the full exact symmetry family

Cons:

- numerically dangerous
- large search space
- easy to overfit or become ill-conditioned

Probably not the right place to start.

## Recommended First Parameterization

If this is explored experimentally, the first try should probably be:

- one diagonal Q/K gauge per head
- one diagonal V/O gauge per head
- conditioned on the head's current parameters after global `P`

This is simple, exact, stable, and already much richer than the current single-scalar head scaling story.

## What Would Be Optimized?

A gauge fixer should not be optimized to change the model's function.

Instead, it should optimize which representative inside the exact symmetry family is easiest for the downstream canonicalization task.

That could mean optimizing for:

- interpolation perplexity ratio
- round-trip perplexity ratio
- canonicalization consistency across seeds
- lower surrogate loss
- softer train/eval mismatch after hardening
- lower condition number or more stable inverses

Schematically:

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{canon}}
+ \lambda_{\text{stab}} \mathcal{L}_{\text{stability}}
+ \lambda_{\text{cons}} \mathcal{L}_{\text{consistency}}
+ \lambda_{\text{cond}} \mathcal{L}_{\text{conditioning}}
$$

where `L_canon` is the main downstream canonicalization objective.

## Useful Regularization

Any learned gauge fixer will likely need strong regularization.

### Identity Prior

Keep the gauge near identity unless there is evidence to move:

$$
\mathcal{L}_{\text{id}} = \|A_h - I\|_F^2 + \|B_h - I\|_F^2
$$

### Conditioning Penalty

Discourage badly conditioned transforms:

$$
\mathcal{L}_{\text{cond}} = \log \kappa(A_h) + \log \kappa(B_h)
$$

### Cross-Seed Consistency

Encourage comparable heads from different models to land in similar local bases.

This is especially useful if the goal is interpolation or model merging.

## Interpretability-Aware Gauge Fixing

One interesting possibility is to choose the canonical representative not just for numerical stability, but also for mechanistic interpretability.

That is legitimate only if the optimization remains inside the exact symmetry orbit.

In that case, interpretability can be a secondary criterion for choosing among equivalent representations.

Possible proxies:

- sparsity
- low feature overlap
- alignment with global probe directions
- stability across seeds
- simpler head-local factor structure
- cleaner token or activation probes

This would turn gauge fixing into both:

- a canonicalization tool
- a possible interpretability aid

But this should likely be treated as a secondary objective, not the first thing to optimize.

## Deterministic Rule Conditioned on Global P

A useful middle ground is:

- use the global `P` matrix first
- then define a deterministic head-local rule on the globally canonicalized head parameters

This gives a parameter-conditioned rule without immediately introducing another learned network.

Examples:

- whiten K or V coordinates after global canonicalization
- sort singular directions after global canonicalization
- choose signs and order by deterministic tie-break rules
- align head-local directions to fixed global probes derived from `P`

This is a plausible first experimental step.

## Risks

- The exact symmetry family may be smaller than expected once biases and implementation details are included.
- A too-flexible gauge parameterization may introduce numerical instability.
- If the gauge fixer leaves the exact symmetry family, it becomes model editing rather than canonicalization.
- A learned gauge fixer may overfit to the downstream objective without improving actual alignment.
- A deterministic rule may be too weak to resolve the real ambiguity.

## Practical Recommendation

If this direction is explored, a reasonable order is:

1. Keep the current global residual canonicalization.
2. Add a deterministic head-local rule after global `P`.
3. Test whether this improves interpolation and round-trip behavior.
4. If it helps but is not sufficient, replace the deterministic rule with a small learned head-local gauge module.
5. Start with diagonal gauges before trying richer parameterizations.

## Main Hypothesis

The current canonicalization module may be underpowered because it only fixes global residual-coordinate ambiguity.

A second stage that fixes head-local Q/K and V/O gauge freedoms could:

- make parameters more comparable across seeds
- reduce artificial multimodality in weight space
- improve interpolation quality
- improve QuineFormer's conditioning target
- possibly yield more interpretable canonicalized heads

## Summary

The key idea is:

- global canonicalization fixes outer residual-basis ambiguity
- head-local gauge fixing would fix inner head-factorization ambiguity

A strong first version would be:

- parameter-conditioned
- applied after global `P`
- deterministic or very small
- strongly regularized
- optimized only within exact symmetry families

If this works, it could become a major extension of the current canonicalization story rather than just a small implementation detail.