// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Linear algebra utilities for integrator library
// ABOUTME: Simplified LINPACK-style LU decomposition and solve routines
#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <array>
#include <cmath>
#include "integrator_types.hpp"

namespace integrators {
namespace linalg {

#ifdef INTEGRATORS_USE_LAPACK
extern "C" {
void dgetrf_(int* m, int* n, double* a, int* lda, int* ipiv, int* info);
void dgetrs_(const char* trans, int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info);
}
#endif

// LU decomposition with partial pivoting (prefer LAPACK if available)
// Stores ipvt[k] = index of pivot row chosen at column k (0-based).
// The matrix A is overwritten with L (unit diagonal implied) and U.
template<size_type N, bool AllowPivoting = true>
int lu_decomposition(std::array<std::array<Real, N>, N>& A,
                     std::array<int, N>& ipvt) {
#ifdef INTEGRATORS_USE_LAPACK
    // Use LAPACK dgetrf for highest parity with DVODE (column-major)
    static_assert(std::is_same_v<Real, double>, "LAPACK path requires Real=double");
    constexpr int n = static_cast<int>(N);
    constexpr int lda = n;
    double a[lda * n];
    // Copy to column-major buffer
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) a[i + lda * j] = A[static_cast<size_type>(i)][static_cast<size_type>(j)];
    }
    int ipiv[n];
    int info = 0;
    dgetrf_(const_cast<int*>(&n), const_cast<int*>(&n), a, const_cast<int*>(&lda), ipiv, &info);
    if (info != 0) return info;
    // Copy LU back into A (still row/col positions correspond to (i,j))
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) A[static_cast<size_type>(i)][static_cast<size_type>(j)] = a[i + lda * j];
    }
    // Store pivots (convert to 0-based)
    for (int k = 0; k < n; ++k) ipvt[static_cast<size_type>(k)] = ipiv[k] - 1;
    return 0;
#else
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
#endif
}

// Solve Ax = b given LU and ipvt from lu_decomposition (LINPACK-style)
template<size_type N, bool AllowPivoting = true>
void lu_solve(const std::array<std::array<Real, N>, N>& LU,
              const std::array<int, N>& ipvt,
              std::array<Real, N>& x) {
#ifdef INTEGRATORS_USE_LAPACK
    static_assert(std::is_same_v<Real, double>, "LAPACK path requires Real=double");
    constexpr int n = static_cast<int>(N);
    constexpr int lda = n;
    double a[lda * n];
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) a[i + lda * j] = LU[static_cast<size_type>(i)][static_cast<size_type>(j)];
    }
    int ipiv[n];
    for (int k = 0; k < n; ++k) ipiv[k] = ipvt[static_cast<size_type>(k)] + 1; // back to 1-based
    int info = 0;
    int nrhs = 1;
    int ldb = n;
    double b[n];
    for (int i = 0; i < n; ++i) b[i] = x[static_cast<size_type>(i)];
    const char trans = 'N';
    dgetrs_(&trans, const_cast<int*>(&n), &nrhs, a, const_cast<int*>(&lda), ipiv, b, &ldb, &info);
    for (int i = 0; i < n; ++i) x[static_cast<size_type>(i)] = b[i];
    (void)info;
    return;
#else
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
#endif
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
