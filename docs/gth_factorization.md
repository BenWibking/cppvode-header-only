# Research Note: GTH Factorization for Non-LTE Populations and Homogeneous Kinetics

This research note documents how the Grassmann–Taksar–Heyman (GTH) rearrangement of Gaussian elimination can be deployed both for non-LTE level population solvers (as implemented in Cloudy) and for homogeneous chemical kinetics with no external source terms. The intent is to provide a unified algebraic and practical reference for applying the algorithm on sparse, nearly singular systems that demand positivity-preserving solutions.

## Background

The GTH algorithm evaluates stationary distributions of finite Markov chains by reorganizing Gaussian elimination to avoid subtractive cancellation and pivoting. The method operates directly on the row-stochastic transition matrix `P`, progressing from the highest-indexed state to the lowest and preserving stochastic structure at every stage (`2101.11657v2.md:63-95`).

Key ideas:
- **Forward eliminations** recursively "censor" the chain, eliminating state `n` while updating the remaining transition coefficients with a rank-one correction (`2101.11657v2.md:79`).
- **Back substitutions** rebuild the stationary vector using stored ratios and the censored transition coefficients, guaranteeing non-negative entries when the inputs are non-negative (`2101.11657v2.md:99-108`).
- **Numerical stability** is achieved by replacing terms like `1 - p_{n,n}` with the positive row sum `∑_{k < n} p_{n,k}`, suppressing the round-off amplification that plagues conventional LU factorization on nearly singular stochastic systems (`2101.11657v2.md:95`).

## Linear Algebra Perspective

Viewed purely as linear algebra, the GTH algorithm factorizes a matrix `A ∈ ℝ^{N×N}` whose rows have unit sum and non-negative off-diagonal entries. The goal is to solve the underdetermined system

```
x^T A = x^T,  1^T x = 1,
```

which can be recast as a homogeneous system with a normalization constraint. The standard approach would form `B = A^T - I` and apply LU factorization with partial pivoting. GTH instead performs the LU steps in reverse row order while exploiting two structural properties of `A`:

1. Each diagonal entry satisfies `a_{n,n} = 1 - ∑_{k < n} a_{n,k}`. Direct elimination would use `a_{n,n}` as the pivot, but it is prone to catastrophic cancellation when `a_{n,n}` is close to 1. GTH replaces the pivot with the positive quantity `s_n = ∑_{k < n} a_{n,k}`.
2. Subtractions never appear: all multiplier updates take the form `m_{i,n} = a_{i,n} / s_n`, and trailing-matrix updates become `a_{i,j} ← a_{i,j} + m_{i,n} a_{n,j}`. This carries out the same Schur-complement update as Gaussian elimination but with addition only.

As a result, the factorization produces implicit lower and upper triangular matrices:

```
A = L U,  with   L_{i,n} = m_{i,n}, U_{n,j} = a_{n,j},
```

where `L` has unit diagonal and non-negative subdiagonal entries, while `U` preserves the remaining upper-triangular structure. Solving `U^T y = 0` and `L^T x = y`, followed by enforcing the normalization constraint, yields a non-negative solution vector without recourse to pivoting. The RG-factorization described in the paper shows that the factors derived in this manner coincide with applying a canonical LU decomposition to `I - A` once the rows have been permuted to match the GTH elimination order (`2101.11657v2.md:169-188`).

## Forward Elimination Mechanics

For each state index `n = N, …, 2`, the GTH step computes

```
p_{i,j}^{n-1} = p_{i,j}^n + (p_{i,n}^n * p_{n,j}^n) / ∑_{k=1}^{n-1} p_{n,k}^n
```

for all `i, j < n` (`2101.11657v2.md:65-83`). The denominator is strictly positive for irreducible chains because it equals the probability of leaving state `n` toward already retained states. The update can be viewed as a Schur complement applied with non-negative weights:

- The numerator `p_{i,n}^n * p_{n,j}^n` captures indirect transitions `i → n → j`.
- Division by the row sum redistributes the outgoing probability mass of state `n` among the remaining states without invoking subtraction.

After each elimination:
- Rows remain non-negative and continue to sum to one, so the reduced matrix `P^{n-1}` is also stochastic (`2101.11657v2.md:72-88`).
- The process only requires storing the reciprocals `1 / ∑_{k < n} p_{n,k}^n`, which later serve as multipliers during back substitution.

## Back Substitution and Normalization

Define the ratios `r_j = π_j / π_1` with `r_1 = 1`. Working upward from the smallest subsystem (`n = 2, …, N`), the algorithm recovers

```
r_j = ∑_{i=1}^{j-1} r_i * p_{i,j}^j / ∑_{k=1}^{j-1} p_{j,k}^j
```

(`2101.11657v2.md:99-108`). Every denominator reuses the positive row sums computed earlier, so the reconstruction again avoids subtraction. Once all ratios are known, the vector is normalized to satisfy `∑_j π_j = 1`.

The RG-factorization described in the paper (`2101.11657v2.md:169-188`) formalizes this procedure as a product of upper- and lower-triangular factors with positive diagonal scaling, drawing a direct analogy to LU decomposition while preserving probabilistic semantics.

## Cloudy’s Adaptation

Cloudy models level populations by solving a homogeneous linear system derived from radiative and collisional transition rates. The routine `gthsolve` in `source/atom_leveln.cpp` implements a modified GTH sweep tailored to the generator matrix of a continuous-time Markov process:

- **Matrix structure**: For every level, the diagonal entry holds the total depopulation rate, and off-diagonal entries store the negatives of transition rates into the target level (`source/atom_leveln.cpp:420-436`). With no external sources, each row sums to zero, i.e., the matrix is a rate generator rather than a row-stochastic probability matrix.
- **Forward sweep**: The solver iterates from the highest level downward. For each state `n`, it accumulates the total outflow toward lower-index states, stores the reciprocal in the population array `pi[n]`, and updates the remaining rows via a rank-one correction (`source/atom_leveln.cpp:59-74`). The algebra is identical to the discrete-time GTH step but applied to the generator entries.
- **Back reconstruction**: After elimination, `pi[0]` is set to unity and the populations are rebuilt by multiplying each stored reciprocal with the cumulative inflow from lower states (`source/atom_leveln.cpp:78-87`). The vector is finally scaled so that the total population equals the supplied species abundance (`source/atom_leveln.cpp:89-93`).
- **Usage guard**: Cloudy only dispatches to `gthsolve` when the net source term is non-positive (`totsrc ≤ 0`), i.e., the equations are homogeneous and physically represent a closed population system (`source/atom_leveln.cpp:592-606`). Otherwise, it falls back to a general LU solver.

Although the Cloudy matrix is not row-stochastic, the continuous-time generator shares the key property that each eliminated state has strictly positive coupling to already retained levels, ensuring the denominators in the forward sweep remain finite. This allows the GTH rearrangement to deliver non-negative level populations even when standard Gaussian elimination produces spurious negative values in nearly singular regimes.

## Takeaways for Chemical Kinetics

Chemical kinetics Jacobians resemble Markov generators—diagonals are negative and rows usually sum to the net depletion rate—yet they often include external source terms or reactions that violate the strict connectivity assumptions needed for a direct GTH application. Cloudy succeeds because the level network guarantees downward or lateral coupling into every previously retained state in the sweep, preserving positive denominators during elimination. Any attempt to reuse this approach for general kinetics must therefore:

1. Reorder species to maintain positive outflow toward retained states at each step.
2. Strip or treat source terms separately so the solver works on a homogeneous generator.
3. Reintroduce the physical normalization (e.g., total abundance) after computing the stationary distribution.

These constraints explain why the GTH factorization is powerful for Markov-style population balances but cannot be dropped into arbitrary LU-based solvers without careful restructuring of the underlying linear system.

## Applying GTH to Source-Free Chemical Kinetics

When a chemical kinetics Jacobian represents a closed network with no external source terms, it shares the essential structure of a continuous-time Markov generator: diagonal entries are non-positive, off-diagonal entries are non-negative reaction rates, and each row sums to zero. In that setting the GTH sweep can be adapted as follows:

1. **Reorder species** so that every eliminated species `n` has some reaction pathway into species with lower indices. Graph algorithms (topological sort on a reduced reaction graph) can enforce this ordering by ensuring the forward sweep never encounters a zero denominator.
2. **Form the generator** `Q` where `q_{i,j} ≥ 0` for `i ≠ j` holds the rate from species `i` into `j`, and `q_{i,i} = -∑_{j≠i} q_{i,j}`. This matches the Cloudy construction (`source/atom_leveln.cpp:420-436`).
3. **Execute the GTH forward sweep** using the continuous-time variant: for each `n`, compute `s_n = -q_{n,n} = ∑_{k < n} q_{n,k}` and update `q_{i,j} ← q_{i,j} + (q_{i,n}/s_n) q_{n,j}` for all `i, j < n`. Store `π_n = 1 / s_n` as the reciprocal pivot.
4. **Perform back substitution** mirroring Cloudy’s reconstruction (`source/atom_leveln.cpp:78-93`): start with `π_0 = 1`, accumulate inflow from lower-index species, and then normalize the final vector to match the conserved total quantity (e.g., total concentration or mole fraction).
5. **Recover the steady state** by applying the normalization to obtain the steady-state concentrations. Because the algorithm never introduces subtraction of nearly equal numbers, it is robust even when the Jacobian is nearly singular due to slow reactions.

This workflow preserves non-negativity and bypasses pivoting, making it attractive for stiff, closed chemical systems where traditional LU solvers struggle to maintain positivity. However, any external source or sink terms must be removed (or handled separately) before applying the GTH factorization, otherwise the denominators in the forward sweep may vanish or become negative.

## Library Usage

The header `include/integrators/linear_algebra.hpp` exposes a subtraction-free implementation ready to integrate with the rest of the library:

```cpp
#include <integrators/linear_algebra.hpp>

using namespace integrators;

constexpr std::size_t N = 4;
using Matrix = std::array<std::array<Real, N>, N>;
using Vector = std::array<Real, N>;

Matrix P = {{
    {{0.85, 0.10, 0.05, 0.0}},
    {{0.05, 0.90, 0.05, 0.0}},
    {{0.10, 0.10, 0.75, 0.05}},
    {{0.00, 0.10, 0.40, 0.50}}
}};

Matrix Pfact = P;
Vector inv_piv{};
Vector pi{};

int info = linalg::gth_factorization<N>(Pfact, inv_piv, linalg::GthMatrixKind::RowStochastic);
if (info != 0) { /* handle breakdown */ }

info = linalg::gth_solve<N>(Pfact, inv_piv, pi, 1.0, linalg::GthMatrixKind::RowStochastic);
if (info != 0) { /* handle breakdown */ }
```

Internally the factorization overwrites the matrix with censored coefficients and stores the reciprocal pivots in `inv_piv`. The solver reconstructs the stationary vector scaled so `sum(pi) == 1` (or the supplied target). For generator matrices, pass `GthMatrixKind::Generator` and set `norm_target` to the conserved quantity (e.g., species abundance).

The repository contains a runnable example at `examples/gth_stationary.cpp` that computes and verifies a stationary distribution for a four-state Markov chain. Build it with CMake (`cmake --build build --target gth_stationary`) and run the executable to see the factorization in action.

For nonlinear chemistry, the header `include/integrators/steady_state_gth.hpp` provides a Picard-style steady-state solver (`steady_state_gth`) that repeatedly rebuilds the generator with user-supplied rates and applies the GTH sweep at every iteration. Problems that expose a `steady_state_generator(state, matrix)` function can call this utility to converge homogeneous kinetics without touching the time-dependent integrators.
