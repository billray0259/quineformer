# Canonicalization Rollout Curriculum

## Motivation

The current V2 objective in experiments/canonicalization/run_v2.py supervises only a single sampled encoder layer at a time.

This is useful early in training because it gives dense, local gradients, but it does not force the learned canonicalization to remain stable under composition across multiple layers.

This may explain a failure mode where:

- one-layer activation MSE looks reasonable
- validation loss looks reasonable
- full-model interpolation perplexity is still terrible

The hypothesis is that the canonicalization module is learning locally plausible alignments that are not yet compositionally valid across multiple successive layers.

## Core Idea

Replace the fixed one-layer surrogate with a rollout-length curriculum.

For a sampled pair of models and a sampled start layer l, choose a rollout length k and unroll k consecutive interpolated layers:

- input target comes from the canonicalized hidden state at layer l
- output target comes from the canonicalized hidden state at layer l + k
- the interpolated model must stay coherent across multiple successive layer applications

This directly pressures the canonicalization to support compositional reasoning, not just one-step matching.

## Basic Formulation

For endpoint models i and j, and rollout length k:

- canonical input at layer l:
  (1 - alpha) times h_i at l times P_i
  plus
  alpha times h_j at l times P_j

- canonical target at layer l + k:
  (1 - alpha) times h_i at l + k times P_i
  plus
  alpha times h_j at l + k times P_j

- run the sequence of interpolated layers from l through l + k - 1

- loss:
  MSE between predicted canonical hidden state and canonical target hidden state
  plus
  lambda_sharp times the permutation sharpness penalty

## Curriculum Proposal

### Fixed Probabilistic Version

Start with a mixture of one-layer and two-layer rollouts.

Let q be the probability of using k = 1.
Let 1 - q be the probability of using k = 2.

Schedule q from 1 to 0 over training:

- early training:
  only one-layer rollouts
- later training:
  increasingly more two-layer rollouts
- eventually:
  mostly or entirely two-layer rollouts

This is simple and cheap to implement.

### Generalized Version

Extend the curriculum so the maximum rollout length increases over time.

Example:

- phase 1:
  only k = 1
- phase 2:
  k in 1, 2
- phase 3:
  k in 1, 2, 3, 4
- phase 4:
  grow toward full-depth rollouts

The model begins with easy local matching and only later is required to maintain alignment over long compositions.

## Adaptive Unlock Version

A better curriculum may be to condition rollout length on demonstrated mastery rather than epoch count.

Maintain a validation metric for each rollout length k, ideally a smoothed held-out metric rather than instantaneous training loss.

Unlock rollout length k + 1 only when rollout length k is stable and sufficiently good.

Example rule:

- keep an EMA validation loss for each k
- if EMA loss for current frontier length stays below a threshold for M checks
  then unlock the next rollout length

This creates a self-paced curriculum:

- the model only sees harder rollout lengths after mastering shorter ones
- the curriculum adapts to actual learning progress rather than wall-clock training time

## Recommended Sampling Strategy

Once rollout lengths up to k_max are unlocked, sample lengths with a frontier-heavy distribution.

Example:

- 60 percent from k_max
- 30 percent from k_max - 1
- 10 percent from shorter lengths

This keeps pressure on the hardest currently learnable regime while still rehearsing easier cases.

A small amount of exploration on k_max + 1 before full unlock may also help.

## Suggested Mastery Signal

Do not use raw per-step training MSE alone to decide unlocks.

Better options:

- EMA validation MSE at rollout length k
- relative MSE normalized by target hidden-state norm
- cosine similarity between predicted and target hidden states
- improvement over a naive baseline

The unlock signal should be stable, comparable across layers, and difficult to game.

## Expected Benefits

- reduces the gap between low local MSE and bad full-model perplexity
- encourages canonicalizations that remain valid under repeated composition
- preserves easy optimization early in training
- gradually shifts the objective toward what we actually care about

## Risks

- longer rollouts increase compute and memory cost
- if the canonical-space execution path is still misspecified, longer rollouts may amplify the problem
- an overly strict unlock threshold may stall progress at short rollout lengths
- an overly aggressive curriculum may destabilize early training

## Practical Recommendation

Start with the smallest useful change:

- implement k = 1 versus k = 2 rollouts
- use a simple q schedule from all one-layer to mixed one-layer and two-layer
- monitor whether two-layer validation loss tracks full-model perplexity better than one-layer loss

If that helps, move to the adaptive unlock version and gradually extend k_max.

## Interpretation

If increasing rollout length improves full-model behavior, then the current problem was likely under-supervision rather than a fundamentally wrong canonicalization target.

If longer rollouts immediately collapse, that is evidence that the current canonical-space execution path is not yet compositionally faithful enough to support the intended objective.