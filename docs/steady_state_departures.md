# Research Note: Steady-State Snap Strategy

## Motivation

Reactive networks often linger near detailed thermodynamic balance while the hydrodynamic driver advances on much longer timesteps. Traditional stiff integrators (`VODE<Problem>`) respond by shrinking their internal step size until the fast chemistry modes are resolved, even though the true solution is already pinned near a steady state. The previous “departure” integrator attempted to evolve linearized perturbations with a frozen Jacobian, but in practice the additional infrastructure offered little advantage over simply rebasing to the steady solution. The library now provides a lighter-weight helper that:

1. Solves for the steady state `y*` with the existing GTH-based nonlinear solver.
2. Checks whether the converged generator matrix indicates detailed balance and reaction timescales short relative to the hydro timestep.
3. Either snaps the solution to `y*` or falls back to the full VODE integrator. If VODE has already failed, the helper returns a hard failure so the caller can surface the error.

This note records the gating logic, configuration knobs, and integration points for the new snap-or-fallback path.

## Workflow Overview

1. **Detect a candidate LTE regime**
   * Run VODE as usual. If it rejects steps, underflows `dt`, or hits the step limit, hand the current state to the snap helper before retrying.
   * Optional: perform the snap attempt proactively whenever diagnostics (e.g., residual norms) suggest near-LTE behavior to avoid even starting VODE.

2. **Solve for `y*` using GTH**
   * Call `steady_state_gth<Problem>(y_guess, config)` with the current composition as the initial guess. Problems must expose `steady_state_generator(state, matrix)` and may supply a permutation via `steady_state_order(order)`.
   * If the steady-state solve fails, immediately fall back to VODE (or abort if VODE has already failed).

3. **Evaluate detailed balance from the generator**
   * Reuse `Problem::steady_state_generator(y*, generator)` to obtain the continuous-time generator at the converged state. Off-diagonal entries carry negative transition rates, diagonals store the total outflow.
   * For every unordered pair `(i, j)`, compare the forward and backward fluxes `φ_{i→j} = y*_i · (−generator[i][j])` and `φ_{j→i} = y*_j · (−generator[j][i])`. The helper declares detailed balance satisfied when  
     `|φ_{i→j} − φ_{j→i}| / (ε + max(φ_{i→j}, φ_{j→i})) ≤ balance_tolerance`.

4. **Check reaction timescales against the hydro step**
   * Interpret the diagonal entries as outflow rates: `τ_i = 1 / generator[i][i]`. The snap is allowed only if `max_i τ_i ≤ timescale_safety · dt_hydro`, ensuring chemistry relaxes much faster than the transport driver.
   * Species with vanishing outflow (diagonal below `min_diagonal`) force a fallback.

5. **Optional departure bound**
   * The helper reports the max weighted difference between the current state and `y*` (`|y_i − y*_i| / (atol_i + rtol_i · max(|y_i|, |y*_i|))`). By default this metric is informational; set `max_departure_tolerance` to a finite value to enforce a hard cap on the snap distance.

6. **Outcome handling**
   * If detailed balance, timescale, and (optional) departure checks all pass, overwrite the integrator state with `y*` and resume the hydro advance.
   * Otherwise return `FallbackToVode` so the caller can continue with the stiff integrator. When VODE already failed on this state, call the helper with `allow_fallback_to_vode = false` to receive a `Failure` result instead.

## Implementation Notes

The header `include/integrators/steady_state_departure.hpp` exposes:

```cpp
struct SteadyStateSnapConfig {
    SteadyStateGthConfig steady_state;
    Real balance_tolerance{1.0e-3};
    Real balance_floor{1.0e-30};
    Real timescale_safety{0.3};
    Real min_diagonal{1.0e-30};
    Real max_departure_tolerance{std::numeric_limits<Real>::infinity()};
    Real departure_floor{1.0e-30};
};

enum class SteadyStateSnapResult { Snapped, FallbackToVode, Failure };

template<typename Problem>
SteadyStateSnapOutcome attempt_steady_state_snap(Real time,
                                                 typename ProblemTraits<Problem>::state_type& y,
                                                 const typename ProblemTraits<Problem>::state_type& atol,
                                                 const typename ProblemTraits<Problem>::state_type& rtol,
                                                 Real hydro_dt,
                                                 bool allow_fallback_to_vode,
                                                 const SteadyStateSnapConfig& cfg = {});
```

A typical integration loop wraps VODE as follows:

```cpp
using Problem = MyNetwork;
using State = typename ProblemTraits<Problem>::state_type;

State y = /* current composition */;
State atol = chemistry_atol();
State rtol = chemistry_rtol();
const Real hydro_dt = current_hydro_dt();

SteadyStateSnapConfig snap_cfg{};
snap_cfg.timescale_safety = 0.3;
snap_cfg.balance_tolerance = 1.0e-2;

auto outcome = attempt_steady_state_snap<Problem>(
    current_time, y, atol, rtol, hydro_dt,
    /*allow_fallback_to_vode=*/true, snap_cfg);

switch (outcome.result) {
case SteadyStateSnapResult::Snapped:
    // y already replaced with y*, resume hydro update
    break;
case SteadyStateSnapResult::FallbackToVode:
    // Run VODE normally; consider caching outcome diagnostics for logging
    break;
case SteadyStateSnapResult::Failure:
    // Neither snap nor VODE succeeded; escalate to caller
    throw std::runtime_error("steady_state_snap failed");
}
```

## Generator Synthesis

For automatically generated chemistry problems, reuse the symbolic Jacobian machinery to emit `steady_state_generator`. With stoichiometric matrices `ν⁻` and `ν⁺` and reaction rates `k_r(y)`:

1. Build off-diagonal entries with `G[i][j] = -Σ_r ν⁻_{r,i} ν⁺_{r,j} k_r(y) y^{ν⁻_r}`.
2. Set diagonals to the negative row sum so each row sums to zero with positive diagonals (total outflow).
3. Optionally provide `steady_state_order` to permute species into a GTH-friendly elimination order.

These functions live alongside the RHS and Jacobian in generated headers so both the steady-state solver and snap helper can operate without extra runtime plumbing.

## Transition Criteria Summary

* **Attempt snap** when VODE reports repeated failures or diagnostics flag near-LTE behavior.
* **Snap accepted** when detailed balance mismatches drop below `balance_tolerance`, maximum relaxation time is under `timescale_safety · dt_hydro`, and (if configured) the weighted departure stays within `max_departure_tolerance`.
* **Fallback** when the snap gate rejects but VODE still has a chance to succeed.
* **Failure** when both VODE and the snap path are exhausted, signalling the caller to reduce the hydro timestep or inspect the chemistry network.

The helper keeps the LTE shortcut aligned with the robust GTH steady-state solve while avoiding the maintenance overhead of a dedicated departure integrator.
