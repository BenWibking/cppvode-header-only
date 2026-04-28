# Primordial Chemistry Port Summary

This note summarizes the work done to port the Microphysics
`primordial_chem` network and its one-zone collapse test into this
header-only VODE codebase, and the debugging needed to make the local
integrator agree with the Microphysics implementation.

## Ported Pieces

- Added a standalone primordial chemistry network implementation in
  `include/integrators/primordial_chem.hpp`.
- Ported the generated RHS and analytic Jacobian from
  `../Microphysics/networks/primordial_chem/actual_rhs.H`.
- Added the primordial chemistry species metadata, number-density helpers,
  EOS helpers, charge balancing, density normalization, and final state
  cleanup needed by the burn-cell problem.
- Added `examples/primordial_chem.cpp`, a port of
  `../Microphysics/unit_test/burn_cell_primordial_chem`.
- Registered the example in `examples/CMakeLists.txt` and added
  `primordial_chem_test` to CTest.

## Microphysics Control Settings Matched Locally

The local example configures VODE to match the Microphysics input file:

- Species tolerances: `rtol_spec = 1.0e-4`, `atol_spec = 1.0e-4`.
- Energy tolerances: `rtol_enuc = 1.0e-6`, `atol_enuc = 1.0e-6`.
- Analytic Jacobian enabled.
- `HMXI = 1.0e-30`, matching Microphysics' effective max-step behavior.
- Number-density species cleanup uses `SMALL_X_SAFE = 1.0e-100`.
- Species change rejection uses `X_reject_buffer = 1.0e100`.
- Species failure tolerance is `1.0e-2`.

This required adding vector tolerances and species constraint controls to
`IntegratorState`.

## VODE Agreement Work

Several VODE details had to be aligned with Microphysics:

- Error weights now support separate species and energy tolerances.
- The energy component is weighted differently from species for error,
  stepsize, and order control, matching Microphysics.
- RHS evaluation now applies the same state cleaning used by Microphysics
  before evaluating the primordial chemistry RHS.
- Jacobian update state, retry flags, nonlinear solver failure paths, and
  step/order retry control were brought closer to the Microphysics VODE
  flow.
- The local `math::powi` helper was changed to match AMReX's recursive
  `powi` behavior, avoiding small arithmetic differences in generated
  rate expressions.
- The unconditional `-O3` interface compile option was removed from the
  top-level CMake target so build type controls optimization as expected.

## EOS Arithmetic Fix

An early divergence at collapse step 151 was traced to a tiny difference
in the first burn's energy RHS. The root cause was EOS arithmetic order.

Microphysics computes the number-density EOS sums in a specific two-pass
order:

1. Accumulate total mass density from number densities and species masses.
2. Accumulate `sum_Abarinv` and `sum_gammasinv`.
3. Scale `sum_Abarinv` by `protonmass / rhotot`.
4. Divide `sum_gammasinv` by `sum_Abarinv`.

The local EOS now follows that same ordering. This changed the first-burn
energy RHS from a value that differed enough to change `dvhin` to one that
matches Microphysics to printed precision in the GNU debug-style run.

## Instrumentation Used

Temporary trace instrumentation was added to both the local VODE and the
Microphysics VODE implementation. The Microphysics edits were restored
after diagnosis.

The traces logged:

- Initial RHS values.
- `dvhin` bounds, iterations, and dominant weighted component.
- Step attempts and accepted steps.
- Jacobian refreshes.
- Nonlinear corrector iterations and failures.
- Error test failures and retry decisions.
- Order and stepsize decisions.

This made it possible to compare collapse steps and find the first
control-flow divergence instead of only comparing final abundances.

## Collapse Step 392 Root Cause

After the EOS fix, the first remaining GNU debug-style mismatch moved to
collapse step 392.

Both implementations agreed through the same repeated error and constraint
failures, then both hit a nonlinear corrector failure at the same step size.
The split occurred in the repeated-failure, order-1 reset path.

Microphysics recomputes `YH(:,2)` from the corrected `vstate.y` left by the
failed attempt. The local port was first overwriting `y` with `YH(:,1)`,
so the retry derivative was evaluated at a different state. That produced
a much larger local error norm at the next retry and sent the adaptive
history down a different path.

The fix was to remove that local reset before recomputing `savf` and
`YH(:,2)` in the order-1 repeated-failure branch.

## Apple Clang FMA Contraction

After the algorithmic mismatch was fixed, Apple Clang still failed the
Microphysics reference comparison unless FMA contraction was disabled.

With Apple Clang's default contraction, the integration completes all 433
collapse steps but diverges from the Microphysics/GNU adaptive trajectory
starting at collapse step 151. The final state then misses the reference
tolerance.

Adding `-ffp-contract=off` makes Apple Clang follow the Microphysics/GNU
trajectory and pass. This flag is scoped only to the `primordial_chem`
example target to avoid changing behavior of unrelated examples and tests.

## Final Verification

The current local build passes the ported primordial chemistry regression:

```text
build/examples/primordial_chem
completed collapse steps: 433
T final:   3032.9924787970513
Eint final:   272183716262.20801
max species relative error vs Microphysics reference: 1.8078479081230499e-10
reference comparison: PASS
```

The full CTest suite also passes with the primordial chemistry example
registered as a test:

```text
ctest --test-dir build --output-on-failure
100% tests passed, 0 tests failed out of 16
```

The temporary Microphysics instrumentation was restored after the
comparison, leaving the sibling Microphysics checkout unchanged.
