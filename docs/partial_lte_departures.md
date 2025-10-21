# Design Note: Partial-LTE Snap Strategy

## Motivation

Many chemistry networks contain a small subset of species that equilibrate rapidly while the remaining components evolve on much slower timescales. The current steady-state shortcut now performs a snap-only rebasing of the *entire* state: solve for `y*` with `steady_state_gth`, check detailed balance/timescales, and either overwrite the solution or fall back to VODE. This breaks down when only a sub-block is near LTE: forcing the slow species to snap can introduce large defects, yet continuing with a fully implicit integrator wastes effort resolving rapidly damped modes. We need a mixed strategy that:

1. Reuses the existing `steady_state_gth` machinery to lock just the stiff subset into steady state.
2. Applies the snap-or-fallback logic to the fast block while leaving the slow block untouched.
3. Seamlessly hands control back to the parent integrator when the LTE assumption deteriorates.

This document sketches how the partial snap controller should work and how it fits within the simplified steady-state handling.

## High-Level Workflow

1. **Detection:** Monitor VODE for stiffness symptoms *localized* to a subset of species: tiny steps driven by select Jacobian rows, error-test failures dominated by one block, or `‖f_S(y)‖` below an LTE threshold while the complement remains dynamic.
2. **Partition:** Identify index sets `S` (fast/LTE candidates) and `F` (slow). The partition may be static (problem-provided) or dynamic (diagnostics-driven). Build permutation vectors so vectors/matrices can be rearranged into `(S,F)` block form.
3. **Partial steady state solve:** Hold `y_F` fixed and call a restricted `steady_state_gth_partial` to obtain `y_S*` satisfying `f_S(y_S*, y_F) = 0` with the appropriate conservation constraint over `S`.
4. **Snap gate:** Construct the generator restricted to the `S` block and reuse the global snap heuristics in miniature:
   * Detailed balance per pair `(i,j) ∈ S` via `φ_{i→j} = y*_i · (−G_S[i][j])`.
   * Relaxation timescales `τ_i = 1 / G_S[i][i]` compared against the caller’s hydro timestep.
   * Optional weighted distance `|y_i − y*_i| / (atol_i + rtol_i · max(|y_i|, |y*_i|))` for `i ∈ S`.
5. **Apply snap:** If all gates pass, overwrite `y_S` with `y_S*` (leave `y_F` untouched), reset VODE’s internal history for those indices, and resume integration. Otherwise, record the rejection (for diagnostics) and fall back to the standard VODE step.
6. **Exit / blacklist:** If the partial solve fails, the gates reject repeatedly, or the fast set changes dramatically, revert to full integration and optionally blacklist the current partition until the system re-enters a comparable LTE regime.

## Detailed Components

### 1. Partition Management

- **Static configuration:** Problems can expose `Problem::lte_partition(state, mask)` returning a boolean mask/ordering for LTE candidates.
- **Dynamic detection:** Use Jacobian spectral info or defect ratios to mark fast species on the fly. Provide heuristics (e.g., relative stiffness ratio, persistent tiny departures) to guard against thrashing.
- **Permutation utilities:** Extend `integrator_types` with helpers to permute arrays into `(S,F)` block order and back. This enables reuse of the dense matrix operations without introducing mixed indexing.

### 2. Partial Steady-State Solve

- Implement `steady_state_gth_partial<Problem>(y, mask, config)` that:
  - Permutes the state into `(S,F)` order and freezes the `F` components.
  - Builds the generator restricted to `S` using the mask/permutation.
  - Enforces normalization on the fast abundance (sum over `S` or conserved quantities supplied by the problem).
  - Returns `y_S*`, the restricted generator, and diagnostics (iterations, residual, permutation).
- For problems with invariants spanning both subsets, project the conservation law onto `S` (e.g., treat `∑_{i∈S} y_i` as the fast conserved quantity while the remainder stays fixed).

### 3. Snap Gate Computation

- Reuse the global helper’s detailed-balance logic on the restricted generator.
- Timescale gate: compute `τ_i = 1 / max(G_S[i][i], ε)` for `i ∈ S`; require `max τ_i ≤ safety · dt_hydro`.
- Distance gate: `|y_i − y*_i| / (atol_i + rtol_i · max(|y_i|, |y*_i|))` for `i ∈ S`.
- Aggregate the worst violations to generate diagnostics and to decide whether to accept or reject the partial snap.

### 4. Coordination with VODE

Implement a controller (`PartialLTESnapController`) that:

1. Intercepts VODE failures/diagnostics and proposes a partition.
2. Runs `steady_state_gth_partial` and the snap gate.
3. On acceptance, overwrites `y_S`, resets the Nordsieck history for those indices, and continues VODE with the same timestep.
4. On rejection, restores the original state (or keeps it unchanged) and lets VODE retry normally.
5. Tracks cooldown/blacklist intervals to avoid repeated snap attempts on unstable partitions.

### 5. Exit Criteria

Re-use the snap gates to decide acceptance/rejection. Additional exit policies:

- Partial steady-state solve failure.
- Hydrodynamic timestep too small relative to the fast timescales.
- Slow block changed beyond tolerance since the last accepted snap (indicating the fast subset is no longer near LTE).

### 6. Implementation Roadmap

1. **Partition utilities:** Add permutation helpers and optional `Problem::lte_partition` hook.
2. **Partial steady-state API:** Implement and unit-test `steady_state_gth_partial`.
3. **Gating logic:** Extend `SteadyStateSnapConfig` or add a dedicated structure for partial snaps (balance/timescale/distance tolerances).
4. **Controller integration:** Embed into VODE’s driver with entry points mirroring the full-state snap pre-check.
5. **Testing:** Build regression networks where only a subset is near LTE, and verify snap acceptance, rejection, and blacklist behaviour.
6. **Documentation and examples:** Update this note and add an example (`examples/partial_lte.cpp`) demonstrating selective snapping.

With this blueprint, the partial-LTE controller reuses the simplified snap infrastructure while addressing realistic scenarios where only a handful of species hug steady state.

## Relationship to QSS Methods

The partial snap approach and classical Quasi Steady-State (QSS) integrators both exploit fast/slow separation, but they are not equivalent:

- **Conditional vs. continuous enforcement:** The snap controller only projects the fast block onto steady state when diagnostics pass; QSS rewrites the governing equations so the algebraic constraints hold throughout the integration.
- **State evolution:** After a snap, the fast species remain fixed until VODE advances again. QSS substitutions keep the fast species evolving continuously alongside the slow block.
- **Fallback behaviour:** Snap keeps the full stiff ODE system intact and falls back to VODE when the gates reject. QSS typically has no built-in fallback—its accuracy depends on how well the algebraic approximation captures the dynamics.

Think of the partial snap controller as an optional projection layered atop VODE, while QSS produces a reduced system where fast species are eliminated entirely. Keeping that distinction clear helps prevent scope creep when extending the snap infrastructure.
