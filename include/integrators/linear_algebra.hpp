// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Linear algebra utilities for integrator library
// ABOUTME: Simplified LINPACK-style LU decomposition and solve routines
#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <array>
#include <cmath>
#include <numeric>
#include "integrator_types.hpp"

namespace integrators {
namespace linalg {

// System LAPACK support removed; always use built-in LU/solve

// LU decomposition with partial pivoting (built-in implementation)
// Stores ipvt[k] = index of pivot row chosen at column k (0-based).
// The matrix A is overwritten with L (unit diagonal implied) and U.
template<size_type N, bool AllowPivoting = true>
int lu_decomposition(std::array<std::array<Real, N>, N>& A,
                     std::array<int, N>& ipvt) {
    for (size_type k = 0; k < N - 1; ++k) {
        // Find pivot index in column k (rows k..N-1)
        size_type pivot_row = k;
        if constexpr (AllowPivoting) {
            Real max_val = std::abs(A[k][k]);
            for (size_type i = k + 1; i < N; ++i) {
                Real val = std::abs(A[i][k]);
                if (val > max_val) { max_val = val; pivot_row = i; }
            }
        }
        ipvt[k] = static_cast<int>(pivot_row);
        // Bring pivot into A[k][k] by swapping only the pivot column entry here
        if (pivot_row != k) {
            std::swap(A[k][k], A[pivot_row][k]);
        }
        // Check for zero pivot (after swap)
        const Real akk = A[k][k];
        if (std::abs(akk) < math::UROUND) {
            return static_cast<int>(k + 1);
        }
        // Compute multipliers: store negative multipliers in A[i][k]
        const Real tscale = -1.0 / akk;
        for (size_type i = k + 1; i < N; ++i) {
            A[i][k] *= tscale;
        }
        // Rank-1 updates across columns with in-loop row interchange (LINPACK style)
        for (size_type j = k + 1; j < N; ++j) {
            Real t = A[pivot_row][j];
            if (pivot_row != k) {
                // swap A(k,j) and A(l,j)
                std::swap(A[k][j], A[pivot_row][j]);
            } else {
                t = A[k][j];
            }
            if (t != 0.0) {
                for (size_type i = k + 1; i < N; ++i) {
                    A[i][j] += A[i][k] * t;
                }
            }
        }
    }
    ipvt[N-1] = static_cast<int>(N-1);
    if (std::abs(A[N-1][N-1]) < math::UROUND) {
        return static_cast<int>(N);
    }
    return 0;
}

enum class GthMatrixKind {
    RowStochastic,
    Generator
};

// GTH factorization for stochastic/generator matrices.
// Stores inverse pivots inv_piv[n] = 1 / sum_{k<n} A[n][k].
// On success returns 0 and leaves A overwritten with censored coefficients.
// On failure returns 1-based index of the row that produced a zero pivot.
template<size_type N>
int gth_factorization(std::array<std::array<Real, N>, N>& A,
                      std::array<Real, N>& inv_piv,
                      GthMatrixKind kind = GthMatrixKind::RowStochastic) {
    static_cast<void>(kind);
    inv_piv.fill(0.0);
    if constexpr (N == 0) {
        return 0;
    }
    const Real eps = math::UROUND;
    for (size_type n = N; n-- > 0;) {
        if (n == 0) {
            break;
        }
        Real sum = 0.0;
        for (size_type k = 0; k < n; ++k) {
            sum += A[n][k];
        }
        if (std::abs(sum) <= eps) {
            return static_cast<int>(n + 1);
        }
        const Real inv = 1.0 / sum;
        inv_piv[n] = inv;
        for (size_type i = 0; i < n; ++i) {
            Real factor = A[i][n] * inv;
            if (factor == 0.0) {
                continue;
            }
            for (size_type j = 0; j < n; ++j) {
                A[i][j] += factor * A[n][j];
            }
        }
    }
    inv_piv[0] = 1.0;
    return 0;
}

// Solve for the stationary vector using GTH factors.
// On success returns 0 and writes solution into x normalized so sum(x) == norm_target.
// For generator matrices, norm_target represents the conserved total (e.g. abundance).
// Returns N if normalization fails due to near-zero sum.
template<size_type N>
int gth_solve(const std::array<std::array<Real, N>, N>& A,
              const std::array<Real, N>& inv_piv,
              std::array<Real, N>& x,
              Real norm_target = 1.0,
              GthMatrixKind kind = GthMatrixKind::RowStochastic) {
    static_cast<void>(kind);
    x.fill(0.0);
    if constexpr (N == 0) {
        return 0;
    }
    x[0] = 1.0;
    for (size_type j = 1; j < N; ++j) {
        Real sum = 0.0;
        for (size_type i = 0; i < j; ++i) {
            sum += x[i] * A[i][j];
        }
        x[j] = inv_piv[j] * sum;
    }
    Real total = std::accumulate(x.begin(), x.end(), 0.0);
    if (std::abs(total) <= math::UROUND) {
        return static_cast<int>(N);
    }
    const Real scale = norm_target / total;
    for (size_type j = 0; j < N; ++j) {
        x[j] *= scale;
    }
    return 0;
}

// Solve Ax = b given LU and ipvt from lu_decomposition (LINPACK-style)
template<size_type N, bool AllowPivoting = true>
void lu_solve(const std::array<std::array<Real, N>, N>& LU,
              const std::array<int, N>& ipvt,
              std::array<Real, N>& x) {
    // Apply row interchanges to x as recorded in ipvt
    if constexpr (AllowPivoting) {
        for (size_type k = 0; k < N - 1; ++k) {
            const int pk = ipvt[k];
            if (pk != static_cast<int>(k)) {
                std::swap(x[k], x[static_cast<size_type>(pk)]);
            }
        }
    }
    // Forward substitution in LINPACK form: for k=0..N-2, x[k+1:] += x[k] * Lcol(k)
    for (size_type k = 0; k + 1 < N; ++k) {
        const Real t = x[k];
        if (t != 0.0) {
            for (size_type i = k + 1; i < N; ++i) {
                x[i] += t * LU[i][k];
            }
        }
    }
    // Backward substitution in LINPACK form
    for (int k = static_cast<int>(N) - 1; k >= 0; --k) {
        x[static_cast<size_type>(k)] /= LU[static_cast<size_type>(k)][static_cast<size_type>(k)];
        const Real t = -x[static_cast<size_type>(k)];
        for (int i = 0; i < k; ++i) {
            x[static_cast<size_type>(i)] += t * LU[static_cast<size_type>(i)][static_cast<size_type>(k)];
        }
    }
}

// Matrix-vector multiplication
template<size_type N>
void matvec(const std::array<std::array<Real, N>, N>& A,
            const std::array<Real, N>& x,
            std::array<Real, N>& y) {
    for (size_type i = 0; i < N; ++i) {
        y[i] = 0.0;
        for (size_type j = 0; j < N; ++j) {
            y[i] += A[i][j] * x[j];
        }
    }
}

// Vector norms
template<size_type N>
Real norm2(const std::array<Real, N>& x) {
    Real sum = 0.0;
    for (size_type i = 0; i < N; ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum / N);
}

template<size_type N>
Real norm_inf(const std::array<Real, N>& x) {
    Real max_val = 0.0;
    for (size_type i = 0; i < N; ++i) {
        max_val = std::max(max_val, std::abs(x[i]));
    }
    return max_val;
}

} // namespace linalg
} // namespace integrators

#endif // LINEAR_ALGEBRA_HPP
