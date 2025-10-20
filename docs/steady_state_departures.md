# Research Note: Steady-State Departure Integration Strategy

## Motivation

Reactive networks frequently linger near chemical steady state, where traditional implicit integrators (e.g. `VODE<Problem>`) are forced to take extremely small steps to resolve rapidly decaying transients. Rather than evolving the full composition vector `y(t)` in these regimes, we can track departures from an equilibrium state `y\*`, allowing the stiff (fast) modes to remain small while the slow evolution is captured efficiently.

This note sketches how to achieve that using the existing solver stack:

* `steady_state_gth` (introduced in `include/integrators/steady_state_gth.hpp` and exercised by `tests/test_steady_state_gth.cpp`) supplies a robust steady-state solve backed by the GTH factorization.
* `VODE<Problem>` (see `include/integrators/vode.hpp`) provides the stiff time integrator already wired into the library.

## Workflow Overview

1. **Detect Near-Steady Dynamics**
   * Monitor the relative correction produced by VODE during Newton updates (`s.acor`/`acnrm_last`) or the residual of the chemical RHS.
   * Once the normalized residual falls below a threshold (e.g. `||f(y)|| < ε_steady`), freeze the standard time integration and switch to the steady-state pathway.

2. **Compute the Reference State**
   * Call `steady_state_gth<Problem>(y, config)` with the current composition as the initial guess.
   * The helper requires `Problem::steady_state_generator(state, matrix)` to build a continuous-time generator and, optionally, `Problem::steady_state_order(order)` to provide a GTH-friendly ordering.
   * The returned vector `y\*` matches the conserved abundance (`config.norm_target`) and satisfies `J(y\*)^T y\* ≈ 0`.

3. **Form Perturbations**
   * Define `δ = y - y\*`. When `||δ||` is small, the kinetic equation linearizes to `δ̇ ≈ J(y\*) δ + higher-order terms`.
   * Store `y\*` separately; VODE will now operate on `δ`.

4. **Integrate the Departures**
   * Instantiate a lightweight wrapper `ProblemDelta` whose state vector is `δ` and whose RHS evaluates `f(y\* + δ) - f(y\*)`.
   * Pass `ProblemDelta` and an initial `δ` into `VODE<ProblemDelta>` using the existing API (`integrate(problem_state, state)`).
   * Because `δ` evolves near zero, stiffness is reduced and VODE's step sizes remain large even in steady regimes.

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
using Problem = MyKinetics;
using State = typename ProblemTraits<Problem>::state_type;

State y = /* current abundance vector */;
SteadyStateGthConfig cfg;
cfg.norm_target = std::accumulate(y.begin(), y.end(), 0.0);

// 1. Steady-state solve
auto ss_result = steady_state_gth<Problem>(y, cfg);
if (ss_result.status != SteadyStateStatus::Success) {
    // fall back to standard VODE integration
}

State y_star = y;                 // steady state
State delta{};                    // perturbation
for (size_type i = 0; i < delta.size(); ++i) {
    delta[i] = current_state[i] - y_star[i];
}

// 2. Wrap RHS around delta
ProblemDelta wrapped{y_star};
VODE<ProblemDelta> vode_delta;
VODEState<ProblemDelta::neqs> vode_state;
vode_state.y = delta;
/* configure tolerances, time span, etc. */

auto status = vode_delta.integrate(wrapped, vode_state);
if (status == IntegratorResult::SUCCESS) {
    // reconstruct full state
    for (size_type i = 0; i < delta.size(); ++i) {
        current_state[i] = y_star[i] + vode_state.y[i];
    }
}
```

Where `ProblemDelta` simply forwards to the original RHS:

```cpp
struct ProblemDelta {
    static constexpr size_type neqs = Problem::neqs;
    using state_type = std::array<Real, neqs>;
    using rhs_type   = state_type;
    State y_star;

    explicit ProblemDelta(const State& y_ss) : y_star(y_ss) {}

    static void rhs(Real t, const state_type& delta, rhs_type& ddelta_dt) {
        State full{};
        for (size_type i = 0; i < neqs; ++i) {
            full[i] = y_star[i] + delta[i];
        }
        rhs_type f_full{};
        Problem::rhs(t, full, f_full);
        rhs_type f_star{};
        Problem::rhs(t, y_star, f_star);
        for (size_type i = 0; i < neqs; ++i) {
            ddelta_dt[i] = f_full[i] - f_star[i];
        }
    }
};
```

## Transition Criteria

* **Enter departure mode** when VODE reports repeated error test failures due to tiny steps, or the linear solve residual `||savf||` falls below `ε_steady`.
* **Leave departure mode** when `||δ||` surpasses a user-defined limit or when `steady_state_gth` fails to converge.
* **Re-base** every time `||δ||` falls below a tighter tolerance, ensuring accumulated numerical error does not corrupt `y\*`.

## Future Work

* Expose diagnostic hooks in `VODEState` so callers can query the local residual and make the steady vs. departure decision automatically.
* Extend `steady_state_gth` to support generator updates in-place, reducing memory traffic during repeated calls.
* Explore low-rank updates in the Picard loop to accelerate the steady-state recomputation.

This departure-based integration path keeps the existing solver infrastructure intact while enabling robust time evolution through stiff, near-equilibrium phases. A companion Matplotlib script for the phase-portrait illustration lives in `docs/scripts/steady_state_phase.py`; running it will produce `steady_state_departures_phase.png` for use in presentations or extended documentation.
