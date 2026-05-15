// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Linear algebra utilities for integrator library
// ABOUTME: Simplified LINPACK-style LU decomposition and solve routines
#ifndef LINEAR_ALGEBRA_HPP
#define LINEAR_ALGEBRA_HPP

#include <array>
#include <cmath>
#include "integrator_types.hpp"

#if defined(INTEGRATORS_USE_CUSOLVERDX) && defined(__CUDACC__)
#include <cusolverdx.hpp>
#endif

namespace integrators {
namespace linalg {

// System LAPACK support removed; always use built-in LU/solve

#if defined(INTEGRATORS_USE_CUSOLVERDX) && defined(__CUDA_ARCH__)
namespace detail {

template<size_type N, bool AllowPivoting>
__device__ int cusolverdx_lu_decomposition(std::array<std::array<Real, N>, N>& A,
                                           std::array<int, N>& ipvt) {
    if constexpr (N == 0) {
        return 0;
    } else {
        static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                      "cuSolverDx LU support requires integrators::Real to be float or double");

        if constexpr (AllowPivoting) {
            using Solver = decltype(cusolverdx::Size<static_cast<unsigned int>(N),
                                                     static_cast<unsigned int>(N)>()
                                    + cusolverdx::Precision<Real>()
                                    + cusolverdx::Type<cusolverdx::type::real>()
                                    + cusolverdx::Function<cusolverdx::function::getrf_partial_pivot>()
                                    + cusolverdx::Arrangement<cusolverdx::row_major>()
                                    + cusolverdx::SM<__CUDA_ARCH__>()
                                    + cusolverdx::Thread());
            typename Solver::status_type info[1]{};
            Solver().execute(&A[0][0], static_cast<unsigned int>(N), &ipvt[0], info);
            return static_cast<int>(info[0]);
        } else {
            using Solver = decltype(cusolverdx::Size<static_cast<unsigned int>(N),
                                                     static_cast<unsigned int>(N)>()
                                    + cusolverdx::Precision<Real>()
                                    + cusolverdx::Type<cusolverdx::type::real>()
                                    + cusolverdx::Function<cusolverdx::function::getrf_no_pivot>()
                                    + cusolverdx::Arrangement<cusolverdx::row_major>()
                                    + cusolverdx::SM<__CUDA_ARCH__>()
                                    + cusolverdx::Thread());
            typename Solver::status_type info[1]{};
            Solver().execute(&A[0][0], static_cast<unsigned int>(N), info);
            for (size_type i = 0; i < N; ++i) {
                ipvt[i] = static_cast<int>(i);
            }
            return static_cast<int>(info[0]);
        }
    }
}

template<size_type N, bool AllowPivoting>
__device__ void cusolverdx_lu_solve(const std::array<std::array<Real, N>, N>& LU,
                                    const std::array<int, N>& ipvt,
                                    std::array<Real, N>& x) {
    if constexpr (N > 0) {
        static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                      "cuSolverDx LU support requires integrators::Real to be float or double");

        if constexpr (AllowPivoting) {
            using Solver = decltype(cusolverdx::Size<static_cast<unsigned int>(N),
                                                     static_cast<unsigned int>(N), 1>()
                                    + cusolverdx::Precision<Real>()
                                    + cusolverdx::Type<cusolverdx::type::real>()
                                    + cusolverdx::Function<cusolverdx::function::getrs_partial_pivot>()
                                    + cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::row_major>()
                                    + cusolverdx::SM<__CUDA_ARCH__>()
                                    + cusolverdx::Thread());
            Solver().execute(&LU[0][0], static_cast<unsigned int>(N), &ipvt[0],
                             &x[0], 1);
        } else {
            using Solver = decltype(cusolverdx::Size<static_cast<unsigned int>(N),
                                                     static_cast<unsigned int>(N), 1>()
                                    + cusolverdx::Precision<Real>()
                                    + cusolverdx::Type<cusolverdx::type::real>()
                                    + cusolverdx::Function<cusolverdx::function::getrs_no_pivot>()
                                    + cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::row_major>()
                                    + cusolverdx::SM<__CUDA_ARCH__>()
                                    + cusolverdx::Thread());
            Solver().execute(&LU[0][0], static_cast<unsigned int>(N), &x[0], 1);
        }
    }
}

template<size_type N, bool AllowPivoting>
__device__ int cusolverdx_lu_factor_solve(std::array<std::array<Real, N>, N>& A,
                                          std::array<int, N>& ipvt,
                                          std::array<Real, N>& x) {
    if constexpr (N == 0) {
        return 0;
    } else {
        static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                      "cuSolverDx LU support requires integrators::Real to be float or double");

        if constexpr (AllowPivoting) {
            using Solver = decltype(cusolverdx::Size<static_cast<unsigned int>(N),
                                                     static_cast<unsigned int>(N), 1>()
                                    + cusolverdx::Precision<Real>()
                                    + cusolverdx::Type<cusolverdx::type::real>()
                                    + cusolverdx::Function<cusolverdx::function::gesv_partial_pivot>()
                                    + cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::row_major>()
                                    + cusolverdx::SM<__CUDA_ARCH__>()
                                    + cusolverdx::Thread());
            typename Solver::status_type info[1]{};
            Solver().execute(&A[0][0], static_cast<unsigned int>(N), &ipvt[0],
                             &x[0], 1, info);
            return static_cast<int>(info[0]);
        } else {
            using Solver = decltype(cusolverdx::Size<static_cast<unsigned int>(N),
                                                     static_cast<unsigned int>(N), 1>()
                                    + cusolverdx::Precision<Real>()
                                    + cusolverdx::Type<cusolverdx::type::real>()
                                    + cusolverdx::Function<cusolverdx::function::gesv_no_pivot>()
                                    + cusolverdx::Arrangement<cusolverdx::row_major, cusolverdx::row_major>()
                                    + cusolverdx::SM<__CUDA_ARCH__>()
                                    + cusolverdx::Thread());
            typename Solver::status_type info[1]{};
            Solver().execute(&A[0][0], static_cast<unsigned int>(N), &x[0], 1, info);
            for (size_type i = 0; i < N; ++i) {
                ipvt[i] = static_cast<int>(i);
            }
            return static_cast<int>(info[0]);
        }
    }
}

} // namespace detail
#endif

// LU decomposition with partial pivoting (built-in implementation)
// Stores ipvt[k] = index of pivot row chosen at column k (0-based).
// The matrix A is overwritten with L (unit diagonal implied) and U.
template<size_type N, bool AllowPivoting = true>
INTEGRATORS_HOST_DEVICE int lu_decomposition(std::array<std::array<Real, N>, N>& A,
                                             std::array<int, N>& ipvt) {
#if defined(INTEGRATORS_USE_CUSOLVERDX) && defined(__CUDA_ARCH__)
    return detail::cusolverdx_lu_decomposition<N, AllowPivoting>(A, ipvt);
#else
    int info = 0;

    if constexpr (N > 1) {
        for (size_type k = 0; k < N - 1; ++k) {
            size_type pivot_row = k;
            if constexpr (AllowPivoting) {
                Real max_val = std::abs(A[k][k]);
                for (size_type i = k + 1; i < N; ++i) {
                    const Real val = std::abs(A[i][k]);
                    if (val > max_val) {
                        max_val = val;
                        pivot_row = i;
                    }
                }
                ipvt[k] = static_cast<int>(pivot_row);
            } else {
                ipvt[k] = static_cast<int>(k);
            }

            if (A[pivot_row][k] != 0.0) {
                if constexpr (AllowPivoting) {
                    if (pivot_row != k) {
                        const Real t = A[pivot_row][k];
                        A[pivot_row][k] = A[k][k];
                        A[k][k] = t;
                        for (size_type j = k + 1; j < N; ++j) {
                            const Real trailing = A[pivot_row][j];
                            A[pivot_row][j] = A[k][j];
                            A[k][j] = trailing;
                        }
                    }
                }

                const Real multiplier = -1.0 / A[k][k];
                for (size_type i = k + 1; i < N; ++i) {
                    A[i][k] *= multiplier;
                }

                for (size_type j = k + 1; j < N; ++j) {
                    const Real t = A[k][j];
                    for (size_type i = k + 1; i < N; ++i) {
                        A[i][j] += t * A[i][k];
                    }
                }
            } else {
                info = static_cast<int>(k + 1);
            }
        }
    }

    if constexpr (N > 0) {
        ipvt[N - 1] = static_cast<int>(N - 1);
        if (A[N - 1][N - 1] == 0.0) {
            info = static_cast<int>(N);
        }
    }

    return info;
#endif
}

// Solve Ax = b given LU and ipvt from lu_decomposition.
template<size_type N, bool AllowPivoting = true>
INTEGRATORS_HOST_DEVICE void lu_solve(const std::array<std::array<Real, N>, N>& LU,
                                      const std::array<int, N>& ipvt,
                                      std::array<Real, N>& x) {
#if defined(INTEGRATORS_USE_CUSOLVERDX) && defined(__CUDA_ARCH__)
    detail::cusolverdx_lu_solve<N, AllowPivoting>(LU, ipvt, x);
#else
    if constexpr (N > 1) {
        for (size_type k = 0; k < N - 1; ++k) {
            Real t{};
            if constexpr (AllowPivoting) {
                const auto pivot_row = static_cast<size_type>(ipvt[k]);
                t = x[pivot_row];
                if (pivot_row != k) {
                    x[pivot_row] = x[k];
                    x[k] = t;
                }
            } else {
                t = x[k];
            }

            for (size_type j = k + 1; j < N; ++j) {
                x[j] += t * LU[j][k];
            }
        }
    }

    for (size_type kb = 0; kb < N; ++kb) {
        const size_type k = N - 1 - kb;
        x[k] /= LU[k][k];
        const Real t = -x[k];
        for (size_type j = 0; j < k; ++j) {
            x[j] += t * LU[j][k];
        }
    }
#endif
}

// One-shot solve. On CUDA devices with cuSolverDx enabled, this maps to GESV.
template<size_type N, bool AllowPivoting = true>
INTEGRATORS_HOST_DEVICE int lu_factor_solve(std::array<std::array<Real, N>, N>& A,
                                            std::array<int, N>& ipvt,
                                            std::array<Real, N>& x) {
#if defined(INTEGRATORS_USE_CUSOLVERDX) && defined(__CUDA_ARCH__)
    return detail::cusolverdx_lu_factor_solve<N, AllowPivoting>(A, ipvt, x);
#else
    const int info = lu_decomposition<N, AllowPivoting>(A, ipvt);
    if (info == 0) {
        lu_solve<N, AllowPivoting>(A, ipvt, x);
    }
    return info;
#endif
}

// Matrix-vector multiplication
template<size_type N>
INTEGRATORS_HOST_DEVICE void matvec(const std::array<std::array<Real, N>, N>& A,
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
INTEGRATORS_HOST_DEVICE Real norm2(const std::array<Real, N>& x) {
    Real sum = 0.0;
    for (size_type i = 0; i < N; ++i) {
        sum += x[i] * x[i];
    }
    return std::sqrt(sum / N);
}

template<size_type N>
INTEGRATORS_HOST_DEVICE Real norm_inf(const std::array<Real, N>& x) {
    Real max_val = 0.0;
    for (size_type i = 0; i < N; ++i) {
        max_val = std::max(max_val, std::abs(x[i]));
    }
    return max_val;
}

} // namespace linalg
} // namespace integrators

#endif // LINEAR_ALGEBRA_HPP
