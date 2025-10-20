# Implementation Plan: Drop-In GTH Factorization

This plan outlines how to introduce the Grassmann–Taksar–Heyman (GTH) factorization as a drop-in replacement for the current LU routines (`include/integrators/linear_algebra.hpp:20`, `include/integrators/linear_algebra.hpp:70`) while preserving existing call sites (`include/integrators/backward_euler.hpp:79`, `include/integrators/vode.hpp:255`, `include/integrators/dvodpk.hpp:1510`). Algorithmic details are based on Zhao (2021, arXiv:2101.11657) and Cloudy’s implementation (`../cloudy/source/atom_leveln.cpp:43`).

## 1. Assess Current Usage
- Enumerate every invocation of `linalg::lu_decomposition` and `linalg::lu_solve` in the integrators and tests to catalog expected matrix properties, pivot usage, and error handling.
- Classify matrices as general Jacobians, Markov generators, or row-stochastic forms to know where GTH can safely substitute LU.

## 2. Design API Surface
- Introduce templated entry points `gth_factorization<N>` and `gth_solve<N>` that mirror the LU signatures: in-place matrix reference, working arrays, and integer status code (`0` success, positive pivot index on failure).
- Provide a convenience wrapper `solve_system_gth` analogous to the existing LU helper so callers can swap implementations with minimal code churn.

## 3. Internal Data Structures
- Define a lightweight container (`GthFactor<N>`) holding:
  - The modified matrix (reusing `std::array<std::array<Real, N>, N>`).
  - Reciprocals of the elimination denominators (`std::array<Real, N>`).
  - Optional metadata (e.g., flags for fallback to LU).
- Maintain compatibility with constexpr contexts and stack allocation (matching current LU design).

## 4. Algorithm Implementation
- Implement the subtraction-free forward sweep described in Zhao (2021, Section 2): descend `n = N … 1`, compute `den_n = ∑_{k<n} a_{n,k}`, guard against `den_n ≤ math::UROUND`, store `1 / den_n`, and apply the Schur-complement update `a_{i,j} += (a_{i,n} / den_n) * a_{n,j}` for `i, j < n`.
- Implement back substitution using the stored reciprocals to reconstruct the solution vector; normalize according to caller requirements (e.g., probability sum or conserved abundance).
- Return a non-zero status if any denominator is non-positive, signalling the caller to fall back to LU.

## 5. Integration Strategy
- Extend `linalg::solve_system` (or equivalent) with an overload or policy flag to attempt GTH first when a matrix satisfies `gth_traits::applicable(A)`.
- Provide detection utilities (static or runtime) to verify structure (non-negative off-diagonal, zero row sums) before opting into GTH; otherwise route to LU.
- Maintain API backward compatibility so existing callers continue to compile without modifications.

## 6. Validation Suite
- Augment `tests/test_linear_algebra.cpp` with:
  - Row-stochastic matrices (probability transitions).
  - Continuous-time generators modeled after Cloudy (ensuring positive populations).
  - Nearly singular systems where LU yields negative entries, demonstrating GTH robustness.
- Compare results from `gth_solve` and `lu_solve` on overlapping cases to ensure equivalence where both apply.

## 7. Documentation and Examples
- Update `docs/gth_factorization.md` with API usage, constraints, and illustrative examples.
- Add a dedicated example program (e.g., `examples/stationary_distribution.cpp`) showing stationary distribution computation via the new interface.

## 8. Performance and Safety
- Benchmark GTH versus LU on representative matrix sizes to quantify overhead.
- Audit numerical safeguards (thresholds, underflow/overflow checks) and ensure error codes propagate to callers.

## 9. Rollout Plan
- Gate the new solver behind an optional CMake switch (`-DINTEGRATORS_ENABLE_GTH=ON`) during initial testing.
- Communicate migration guidance in `README.md`, highlighting when developers should opt into GTH and how to interpret its error codes.

Following this plan will deliver a GTH implementation that integrates cleanly with the existing API, provides a numerically stable alternative for Markov-style systems, and remains safe to use as a drop-in replacement where applicable.
