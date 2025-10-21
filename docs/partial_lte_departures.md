# Design Note: Partial-LTE Departure Integration

## Motivation

Many chemistry networks contain a small subset of species that equilibrate rapidly while the remaining components evolve on much slower timescales. The current steady-state departure path assumes the *entire* state is close to LTE, rebases to the nonlinear steady state `y*`, and advances all departures with a frozen Jacobian exponential stepper. This breaks down when only a sub-block is near LTE: forcing the slow species to rebase can introduce large defects, yet continuing with a fully implicit integrator wastes effort resolving rapidly damped modes. We need a mixed strategy that:

1. Reuses the existing `steady_state_gth` machinery to lock just the stiff subset into steady state.
2. Evolves departures only for that subset, keeping the slow block under the control of the parent integrator (e.g., VODE).
3. Seamlessly hands control back when the LTE assumption deteriorates.

This document describes the partitioned departure approach and how it fits within the current library.

## High-Level Workflow

1. **Detection:** Monitor the full integrator (VODE) for stiffness symptoms *localized* to a subset of species: tiny steps driven by select Jacobian rows, error-test failures dominated by a single block, or a weighted residual `‖f_S(y)‖` below an LTE threshold while the complement remains dynamic.
2. **Partition:** Identify index sets `S` (fast/LTE candidates) and `F` (slow). This may be static (provided by the problem) or dynamic (based on diagnostics). Build permutation vectors so vectors/matrices can be rearranged into block form.
3. **Partial steady state:** Hold `y_F` fixed at its current value and call `steady_state_gth` (or a variant) on the restricted system to solve `f_S(y_S*, y_F) = 0` with the appropriate conservation constraint for the `S` block. The result `y_S*` is a fully nonlinear steady state for the fast species relative to the frozen slow block.
4. **Departure formation:** Define `δ_S = y_S - y_S*`, set `δ_F = 0`, and cache `A_SS = J_SS(y_S*, y_F)` along with the cross block `A_SF = J_SF(y_S*, y_F)` if needed for diagnostics.
5. **Departure propagation:** Advance `δ_S` using the existing exponential stepper infrastructure, operating on the reduced system:
   ```
   δ̇_S ≈ A_SS δ_S  (since y_F is fixed during the departure step and f_S(y_S*, y_F)=0).
   ```
   The exponential cache, defect tests, and rebase triggers now apply to the `S` block only.
6. **Slow block coupling:** After each accepted departure step, reconstruct the full state:
   ```
   y = [ y_S* + δ_S ]
       [ y_F       ].
   ```
   The parent integrator can either (a) skip its own update for `S` while the departure solver is active, or (b) treat the updated `y` as a new starting point before continuing its usual step for `F`.
7. **Exit / rebase:** If `‖δ_S‖` or the weighted defect exceeds thresholds, or if the partial steady-state solve fails, revert to the full integrator and evolve everyone until partial LTE is detected again.

## Detailed Components

### 1. Partition Management

- **Static configuration:** Problems can expose `Problem::lte_partition(state, mask)` returning a boolean mask/ordering for LTE candidates.
- **Dynamic detection:** Use Jacobian spectral info or defect ratios to mark fast species on the fly. Provide heuristics (e.g., relative stiffness ratio, persistent tiny departures) to guard against thrashing.
- **Permutation utilities:** Extend `integrator_types` with helpers to permute arrays into `(S,F)` block order and back. This enables reuse of the dense matrix operations without introducing mixed indexing.

### 2. Partial Steady-State Solve

- Add `steady_state_gth_partial<Problem>(y_S, y_F, config, mask)`:
  - Freezes `y_F`.
  - Builds the generator restricted to `S` (using the mask/permutation).
  - Enforces normalization on the partial abundance (sum over `S` or individual conserved quantities supplied by the problem).
  - Returns `y_S*`, status, iteration count, and optionally the permutation used.
- For problems with coupled invariants spanning both subsets, supply an option to project the conservation equation into the `S` variables (e.g., treat `∑_{i∈S} y_i` as the fast conserved quantity while the remainder is held constant).

### 3. Departure State Structures

Extend `DepartureState` to track block-specific views:

```cpp
struct PartialDepartureState {
    // Global data
    Real current_time;
    State y;

    // Partition description
    std::array<size_type, N> order;  // (S,F) permutation
    size_type n_fast;                // |S|

    // Fast block cache
    State y_star_fast;
    State delta_fast;
    Matrix jacobian_ss;
    State rhs_fast;                  // f_S(y_S*, y_F) == 0 (stored for diagnostics)

    // Slow block snapshot
    State y_slow;
    State atol_fast;
    State rtol_fast;
    Real defect_last;
};
```

`DepartureWorkspace` can be reused as-is, operating on `n_fast × n_fast` matrices after permuting into block form.

### 4. Exponential Stepper Adaptation

- Before calling `integrate_departure_step`, restrict the cached matrices and vectors to the fast block via the stored permutation.
- After each step, scatter the updated fast block back to the full state; the slow block preserves its previous value.
- Weighted defect computation uses the fast tolerances only. Optionally evaluate the cross defect `r_S = f_S(y_S* + δ_S, y_F) - (A_SS δ_S)` to ensure coupling from `F` does not introduce drift beyond tolerance.

### 5. Coordination with VODE

Implement a controller object (`PartialLTEController`) that:

1. Intercepts VODE failures/diagnostics.
2. Runs the partition detection and partial steady-state solve.
3. Invokes the reduced departure integrator for one or more macro-steps (advancing `y_S` only).
4. Returns control to VODE with the updated full state and a suggestion for the next `dt` for the slow integrator.

Key policy decisions:

- **Step sequencing:** Either (a) lock VODE out while the departure solver advances `t` to the requested `tout`, or (b) interleave small departure steps between standard VODE steps so that `F` continues to evolve.
- **Time synchronization:** If the departure path advances the global time, the slow block must be extrapolated or held constant over the same interval. A simple first phase keeps `y_F` fixed; future work can use an explicit prediction for `y_F` to include first-order coupling terms (`J_SF δ_F`) if needed.

### 6. Exit Criteria

Re-use the existing thresholds with block-specific scopes:

- `‖δ_S‖_∞` exceeding `max_departure_norm_fast`.
- Weighted defect `max_i |r_S,i|/(atol_S,i + rtol_S,i |y_i|)` exceeding `rebase_defect_fast`.
- Partial steady-state solver failure or requirement to rotate species in/out of the LTE set.
- Large changes detected in `y_F` while the departure solver is active (indicating the assumption of a frozen slow block no longer holds).

When any trigger fires, flush the cache, restore control to VODE, and optionally blacklist the current partition until the system re-enters a comparable LTE regime.

### 7. IMEX Relationship

Although the split resembles IMEX integration, note the conceptual differences:

- The departure solver operates in coordinates shifted by the nonlinear steady state of the fast block.
- The slow RHS is not integrated additively during the departure step; the slow species are treated as parameters.
- To emulate an IMEX formulation, we would need to add explicit coupling terms for the slow update; that remains out-of-scope for the first iteration but is an avenue for future refinement.

### 8. Implementation Roadmap

1. **Partition infrastructure:** Add permutation utilities and problem hooks (`lte_partition`). Provide default no-op implementations for problems that do not expose LTE subsets.
2. **Partial steady-state API:** Implement `steady_state_gth_partial` with unit tests covering simple block-partitioned systems.
3. **State extensions:** Introduce `PartialDepartureState`/`PartialDepartureWorkspace` wrappers that sit atop the existing departure machinery.
4. **Integrator loop plumbing:** Create `integrate_partial_departure_step` that slices the fast block, calls the existing exponential stepper, and assembles the full state afterward.
5. **Controller integration:** Embed into VODE’s driver logic with entry/exit hooks mirroring the full-state departure path. Add diagnostics to log partition changes and exit causes.
6. **Testing:** Craft regression tests where only part of the system is near LTE (e.g., synthetic two-time-scale kinetics) and verify the solver stabilizes the fast block while the slow block keeps evolving.
7. **Documentation and examples:** Update the steady-state departure docs to cover partial LTE and add an example (e.g., `examples/partial_lte.cpp`) demonstrating the workflow.

With this blueprint, the partial-LTE departure integrator can reuse most of the infrastructure built for the full-state case while addressing realistic scenarios where only a handful of species hug steady state.
