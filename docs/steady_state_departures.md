# Research Note: Steady-State Departure Integration Strategy

## Motivation

Reactive networks frequently linger near chemical steady state, where traditional implicit integrators (e.g. `VODE<Problem>`) are forced to take extremely small steps to resolve rapidly decaying transients. Rather than evolving the full composition vector `y(t)` in these regimes, we can track departures from an equilibrium state `y\*`, allowing the stiff (fast) modes to remain small while the slow evolution is captured efficiently.

This note sketches how to achieve that using the existing solver stack:

* `steady_state_gth` (introduced in `include/integrators/steady_state_gth.hpp` and exercised by `tests/test_steady_state_gth.cpp`) supplies a robust steady-state solve backed by the GTH factorization.
* `VODE<Problem>` (see `include/integrators/vode.hpp`) provides the stiff time integrator already wired into the library.

## Workflow Overview

1. **Detect Near-Steady Dynamics**
   * Two entry paths are supported:
     * **Default failover:** Run VODE (or the existing stiff solver) as usual. If it succeeds, continue. If it fails with `TOO_MANY_STEPS`, repeated error-test failures, or timestep underflow, attempt a steady-state rebase: compute `y\*`, `A = J(y\*)`, `b = f(y\*)`, and evaluate the weighted defect `r = f(y) - (A δ + b)` at the current state. Enter the steady-state departure workflow only if `max_i |r_i| / (atol_i + rtol_i · |y_i|)` falls below a user-tuned entry tolerance, signalling that VODE stalled because of near-LTE stiffness rather than a real instability.
     * **Optional pre-check:** When a runtime flag (e.g. `precheck_defect`) is enabled, perform the steady-state solve and defect test before calling VODE. If the weighted defect already satisfies the entry tolerance, skip the VODE attempt and start directly in exponential mode; otherwise fall back to VODE.

2. **Compute the Reference State**
   * Call `steady_state_gth<Problem>(y, config)` with the current composition as the initial guess.
   * The helper requires `Problem::steady_state_generator(state, matrix)` to build a continuous-time generator and, optionally, `Problem::steady_state_order(order)` to provide a GTH-friendly ordering.
   * The returned vector `y\*` matches the conserved abundance (`config.norm_target`) and satisfies `J(y\*)^T y\* ≈ 0`.

3. **Form Perturbations**
   * Define `δ = y - y\*`. When `||δ||` is small, the kinetic equation linearizes to `δ̇ ≈ J(y\*) δ + higher-order terms`.
   * Store `y\*` separately; all subsequent integration operates on the departure `δ`.

4. **Integrate the Departures**
   * Freeze the Jacobian at the LTE reference: evaluate `A = J(y\*)` and the residual vector `b = f(y\*)`.
* Propagate `δ` with an exponential integrator for the linear system `δ̇ = A δ + b`, reusing factorizations of `A` so each macro-step costs one dense solve.
* For the small systems we target (`N < 40`), use a real-Schur decomposition to precompute `exp(hA)` and the `φ₁(hA)` action efficiently.
* After each exponential step, form the nonlinear defect `r = f(y\* + δ) - (A δ + b)`; accept the step while `‖r‖` stays below a tolerance tied to the LTE tube.
* This strategy assumes the RHS is autonomous (no explicit `t` dependence) and that the problem exposes an analytic Jacobian.
* If both the departure `δ` and the residual `f(y)` fall below chemistry tolerances, bypass the exponential step entirely and snap the state to `y\*`, since the update would be smaller than the desired accuracy.
    * When deciding to snap, reuse the generator matrix that `steady_state_gth` already evaluated at `y\*` (`Problem::steady_state_generator(y\*, generator)`). For each unordered pair `(i, j)` compute forward/backward fluxes `φ_{i→j} = y\*_i · (−generator[i][j])` and `φ_{j→i} = y\*_j · (−generator[j][i])`; require `|φ_{i→j} − φ_{j→i}| / (ε + max(φ_{i→j}, φ_{j→i})) ≤ τ_balance` (e.g. `τ_balance ≈ 1e−3`). This enforces detailed balance directly from the converged generator coefficients.
    * The same generator diagonals give per-species relaxation times (`τ_i = 1 / max(generator[i][i], ε)`); gate the snap on `max_i τ_i ≤ safety · dt_hydro`, ensuring all reaction timescales remain comfortably below the hydrodynamic step.

5. **Adaptive Re-basing**
   * Periodically (or when `||δ||` exceeds a threshold) recompute the steady state:
     1. Recover the full state `y = y\* + δ`.
     2. Re-run `steady_state_gth` to obtain a refreshed `y\*`.
     3. Reset `δ = y - y\*` and continue integrating.
   * If `||δ||` grows significantly (system exits steady state), switch back to the original `Problem` and let VODE drive the full composition directly.

6. **Interfacing with Sources/Sinks**
   * The approach assumes the kinetic system is homogeneous or that source terms are slow compared to the fast equilibration. For problems with explicit source vectors (cf. Cloudy's `source[level]` handling in `source/atom_leveln.cpp`), subtract the steady-state contribution before constructing `δ`.

## Implementation Sketch

```cpp
using integrators::ProblemTraits;
using integrators::SteadyStateGthConfig;
using integrators::SteadyStateStatus;
using integrators::steady_state_gth;
using Real = integrators::Real;

using Problem = MyKinetics;
using State = typename ProblemTraits<Problem>::state_type;
using Matrix = typename ProblemTraits<Problem>::jacobian_type;

// Assumptions: autonomous RHS and analytic Jacobian available at y*
Real current_time = /* current simulation time */;
State y_full = /* current composition */;
State y_star = y_full;
SteadyStateGthConfig cfg;
cfg.norm_target = std::accumulate(y_full.begin(), y_full.end(), 0.0);
State atol{}; // per-species absolute tolerances (reuse values from the VODE chemistry driver)
State rtol{}; // per-species relative tolerances
initialize_tolerances(atol, rtol);
Real defect_tolerance = Real{0.3}; // loosen or tighten to match chemistry accuracy targets

// 1. Steady-state solve
auto ss_result = steady_state_gth<Problem>(y_star, cfg);
if (ss_result.status != SteadyStateStatus::Success) {
    // fall back to standard VODE integration
}

// 2. Form perturbations about the steady state y*
State delta{};
for (std::size_t i = 0; i < delta.size(); ++i) {
    delta[i] = y_full[i] - y_star[i];
}

// 3. Build the frozen linear model δ̇ = A δ + b
Matrix A{};
Problem::jacobian(current_time, y_star, A);
State b{};
Problem::rhs(current_time, y_star, b);

// 4. Real-Schur decomposition of A (small dense matrix: N < 40)
SchurDecomposition<Matrix> schur = schur_decompose(A); // returns Q, T
Matrix Q = schur.Q;
Matrix T = schur.T;

// Rotate into Schur space
State delta_hat = mul_transpose(Q, delta);
State b_hat = mul_transpose(Q, b);

Real h = choose_step_size(/* heuristics based on LTE window */);
Matrix exp_hT = matrix_exponential_scaled(T, h);
Matrix phi1_hT = matrix_phi1(T, h);

Real t = current_time;
while (t < target_time) {
    // Exact propagation for the frozen linear system
    State delta_hat_trial = mat_vec(exp_hT, delta_hat);
    if (!is_near_zero(b_hat)) {
        State correction = mat_vec(phi1_hT, b_hat);
        for (std::size_t i = 0; i < correction.size(); ++i) {
            delta_hat_trial[i] += h * correction[i];
        }
    }

    State delta_trial = mat_vec_transpose(Q, delta_hat_trial);
    State y_trial{};
    for (std::size_t i = 0; i < y_trial.size(); ++i) {
        y_trial[i] = y_star[i] + delta_trial[i];
    }

    // Defect-based accept/reject
    State rhs_full{};
    Problem::rhs(t + h, y_trial, rhs_full);
    State defect{};
    for (std::size_t i = 0; i < defect.size(); ++i) {
        defect[i] = rhs_full[i] - (mat_vec(A, delta_trial)[i] + b[i]);
    }

    Real max_weighted_defect = Real{0};
    for (std::size_t i = 0; i < defect.size(); ++i) {
        Real weight = atol[i] + rtol[i] * std::abs(y_trial[i]);
        max_weighted_defect = std::max(max_weighted_defect, std::abs(defect[i]) / std::max(weight, Real{1e-30}));
    }
    if (max_weighted_defect <= defect_tolerance) {
        delta_hat = delta_hat_trial;
        delta = delta_trial;
        y_full = y_trial;
        t += h;
        // optional: consider increasing h modestly when successive accepts occur
    } else {
        h *= Real{0.5};
        exp_hT = matrix_exponential_scaled(T, h);
        phi1_hT = matrix_phi1(T, h);
        // retry without advancing time
        continue;
    }

    Real defect_norm = norm(defect);
    if (should_rebase(delta, defect_norm)) {
        break; // hand control back to the main integrator or refresh y*
    }
}
```

The helpers (`initialize_tolerances`, `schur_decompose`, `matrix_exponential_scaled`, `matrix_phi1`, `mat_vec`, `mat_vec_transpose`, `mul_transpose`, `norm`, `is_near_zero`) are thin wrappers around dense linear algebra routines (Eigen/LAPACK) that operate on the small `N × N` matrices in this regime. `should_rebase` encapsulates user logic for switching back to the nonlinear integrator. The defect test mirrors VODE’s error-weighting strategy by dividing each component by `atol + rtol · |y|`, so the exponential step is accepted exactly when the unmodelled nonlinear drift would stay within the familiar chemistry tolerances.

A complementary change is planned for the JAFF network generator so that the C++ output can synthesize the required steady-state hooks automatically:

*Symbolic Generator Synthesis* — During network parsing, retain the stoichiometric matrices `ν⁻` and `ν⁺`. Use SymPy to assemble per-reaction rate expressions `k_r(y)` (the same objects already used for the Jacobian). Build the generator symbolically via `G[i][j] = -Σ_r ν⁻_{r,i}·ν⁺_{r,j}·k_r(y)·y^{ν⁻_r}`, with diagonals reset to the negative row sum. Apply SymPy CSE to emit compact C++ for `steady_state_generator`. Compute a permutation from the reaction graph (edges where both `ν⁻` and `ν⁺` are non-zero) to populate `steady_state_order`. The KokkoS template will include these functions behind an optional flag, so deployed networks can provide the generator/order pair without hand-written code.

## Transition Criteria

* **Enter departure mode** when VODE reports repeated error test failures due to tiny steps, or the linear solve residual `||savf||` falls below `ε_steady`.
* **Leave departure mode** when `||δ||` surpasses a user-defined limit or when `steady_state_gth` fails to converge.
* **Re-base** whenever the defect-based acceptance test starts rejecting steps repeatedly; refresh `y\*`, recompute `A`, and rebuild the Schur/exponential factors.

## Future Work

* Expose diagnostic hooks in `VODEState` so callers can query the local residual and make the steady vs. departure decision automatically.
* Extend `steady_state_gth` to support generator updates in-place, reducing memory traffic during repeated calls.
* Explore low-rank updates in the Picard loop to accelerate the steady-state recomputation.

## Remaining Tasks

* Implement the frozen-Jacobian exponential stepper (Schur factorization, cached `exp(hA)`/`φ₁(hA)` actions, defect-weighted accept/reject, and re-basing).
* Ensure chemistry problems expose `steady_state_generator`/`steady_state_order`, including automatic emission from the JAFF network generator.
* Wire the defect-based entry/exit logic so VODE can hand control to the departure solver when it stalls and resume full integration once LTE breaks.
* Pursue the performance polish items (in-place generator refresh, low-rank Picard updates) after the core path lands.

### Exponential Propagator Implementation Steps

1. **State preparation** — Solve the full nonlinear steady-state problem with `steady_state_gth` to obtain `y_star` satisfying `f(y_star) = 0`, then assemble `delta`, `A = J(y_star)`, and `b = f(y_star)` inside a dedicated entry point (e.g., `integrate_departure_step`), and stash the tolerances/defect threshold that gate acceptance.
2. **Real-Schur factorization** — Factor `A` once per rebase into orthogonal `Q` and quasi-upper-triangular `T`; surface a compact struct that caches `Q`, `T`, and the problem dimension.
3. **Matrix function cache** — For a proposed macro-step `h`, form `exp_hT` and `phi1_hT` (using scaling-and-squaring or a Padé/Lanczos routine suitable for the small dense `T`). Expose helpers to refresh these when `h` changes.
4. **Linear-system propagation** — Rotate `delta` and `b` into Schur space (`delta_hat = Q^T delta`, `b_hat = Q^T b`), apply the frozen linear update (`delta_hat_trial = exp_hT * delta_hat + h * phi1_hT * b_hat`), and map back with `Q`.
5. **Defect evaluation** — Reconstruct `y_trial = y_star + delta_trial`, call the full RHS, and build the weighted defect `r = f(y_trial) - (A delta_trial + b)` using the chemistry tolerances.
6. **Accept/reject control** — Accept the step when the maximum weighted defect falls below the tolerance, update `delta`, `y`, and time, and consider enlarging `h`; otherwise reduce `h`, recompute the matrix functions, and retry without advancing.
7. **Rebase triggers** — Monitor `||delta||` and defect trends to decide when the approximation fails; on trigger, exit so the caller can recompute `y_star`, `A`, and `b` before restarting the loop.
8. **Public API** — Wrap the above in a reusable driver that integrates until either success, rebase request, or exit back to VODE, returning status codes and updated state for seamless handoff.

This departure-based integration path keeps the existing solver infrastructure intact while reusing steady-state Jacobians to drive an exponential propagator through stiff, near-equilibrium phases.
