// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Linear algebra utilities for integrator library
// ABOUTME: Simplified LINPACK-style LU decomposition and solve routines
#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
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

// Matrix helpers used by exponential propagator
template<size_type N>
std::array<std::array<Real, N>, N> identity_matrix() {
    std::array<std::array<Real, N>, N> I{};
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            I[i][j] = (i == j) ? Real{1.0} : Real{0.0};
        }
    }
    return I;
}

template<size_type N>
std::array<std::array<Real, N>, N>
matrix_add(const std::array<std::array<Real, N>, N>& A,
           const std::array<std::array<Real, N>, N>& B) {
    std::array<std::array<Real, N>, N> C{};
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

template<size_type N>
std::array<std::array<Real, N>, N>
matrix_sub(const std::array<std::array<Real, N>, N>& A,
           const std::array<std::array<Real, N>, N>& B) {
    std::array<std::array<Real, N>, N> C{};
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

template<size_type N>
std::array<std::array<Real, N>, N>
matrix_scale(const std::array<std::array<Real, N>, N>& A, Real alpha) {
    std::array<std::array<Real, N>, N> C{};
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            C[i][j] = alpha * A[i][j];
        }
    }
    return C;
}

template<size_type N>
void matrix_scale_inplace(std::array<std::array<Real, N>, N>& A, Real alpha) {
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            A[i][j] *= alpha;
        }
    }
}

template<size_type N>
std::array<std::array<Real, N>, N>
matrix_multiply(const std::array<std::array<Real, N>, N>& A,
                const std::array<std::array<Real, N>, N>& B) {
    std::array<std::array<Real, N>, N> C{};
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            Real sum = 0.0;
            for (size_type k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}

template<size_type N>
Real matrix_norm1(const std::array<std::array<Real, N>, N>& A) {
    Real max_col_sum = 0.0;
    for (size_type j = 0; j < N; ++j) {
        Real col_sum = 0.0;
        for (size_type i = 0; i < N; ++i) {
            col_sum += std::abs(A[i][j]);
        }
        max_col_sum = std::max(max_col_sum, col_sum);
    }
    return max_col_sum;
}

namespace detail {

template<size_type N>
void matrix_pade13(const std::array<std::array<Real, N>, N>& A,
                   std::array<std::array<Real, N>, N>& U,
                   std::array<std::array<Real, N>, N>& V) {
    constexpr Real b[] = {
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0
    };

    const auto A2 = matrix_multiply(A, A);
    const auto A4 = matrix_multiply(A2, A2);
    const auto A6 = matrix_multiply(A4, A2);
    const auto I = identity_matrix<N>();

    auto tmp = matrix_add(matrix_scale(A6, b[13]), matrix_scale(A4, b[11]));
    tmp = matrix_add(tmp, matrix_scale(A2, b[9]));
    auto R = matrix_multiply(A6, tmp);

    auto tmp2 = matrix_add(matrix_scale(A6, b[7]), matrix_scale(A4, b[5]));
    tmp2 = matrix_add(tmp2, matrix_scale(A2, b[3]));
    tmp2 = matrix_add(tmp2, matrix_scale(I, b[1]));
    auto inner = matrix_add(R, tmp2);
    U = matrix_multiply(A, inner);

    auto tmp3 = matrix_add(matrix_scale(A6, b[12]), matrix_scale(A4, b[10]));
    tmp3 = matrix_add(tmp3, matrix_scale(A2, b[8]));
    auto term1 = matrix_multiply(A6, tmp3);

    auto tmp4 = matrix_add(matrix_scale(A6, b[6]), matrix_scale(A4, b[4]));
    tmp4 = matrix_add(tmp4, matrix_scale(A2, b[2]));
    tmp4 = matrix_add(tmp4, matrix_scale(I, b[0]));
    V = matrix_add(term1, tmp4);
}

template<size_type N>
bool matrix_solve_inplace(std::array<std::array<Real, N>, N>& A,
                          std::array<std::array<Real, N>, N>& B) {
    std::array<int, N> ipvt{};
    int info = lu_decomposition<N>(A, ipvt);
    if (info != 0) {
        return false;
    }
    for (size_type j = 0; j < N; ++j) {
        std::array<Real, N> column{};
        for (size_type i = 0; i < N; ++i) {
            column[i] = B[i][j];
        }
        lu_solve<N>(A, ipvt, column);
        for (size_type i = 0; i < N; ++i) {
            B[i][j] = column[i];
        }
    }
    return true;
}

} // namespace detail

template<size_type N>
bool matrix_exponential(const std::array<std::array<Real, N>, N>& A,
                        std::array<std::array<Real, N>, N>& expA) {
    if constexpr (N == 0) {
        return true;
    }

    const Real theta13 = 4.25;
    auto A_scaled = A;
    const Real normA = matrix_norm1(A);
    int s = 0;
    if (normA > theta13 && std::isfinite(normA)) {
        s = static_cast<int>(std::ceil(std::log2(normA / theta13)));
        s = std::max(s, 0);
        const Real scale = std::ldexp(1.0, -s);
        matrix_scale_inplace(A_scaled, scale);
    }

    std::array<std::array<Real, N>, N> U{};
    std::array<std::array<Real, N>, N> V{};
    detail::matrix_pade13(A_scaled, U, V);

    auto P = matrix_add(V, U);
    auto Q = matrix_sub(V, U);
    auto Q_factor = Q;
    if (!detail::matrix_solve_inplace(Q_factor, P)) {
        return false;
    }

    auto result = P;
    for (int k = 0; k < s; ++k) {
        result = matrix_multiply(result, result);
    }
    expA = result;
    return true;
}

template<size_type N>
bool matrix_phi1(const std::array<std::array<Real, N>, N>& A,
                 std::array<std::array<Real, N>, N>& phi1A) {
    if constexpr (N == 0) {
        return true;
    }
    constexpr size_type M = 2 * N;
    std::array<std::array<Real, M>, M> block{};
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            block[i][j] = A[i][j];
        }
        block[i][N + i] = Real{1.0};
    }
    std::array<std::array<Real, M>, M> exp_block{};
    if (!matrix_exponential<M>(block, exp_block)) {
        return false;
    }
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = 0; j < N; ++j) {
            phi1A[i][j] = exp_block[i][N + j];
        }
    }
    return true;
}

} // namespace linalg
} // namespace integrators

#endif // LINEAR_ALGEBRA_HPP
