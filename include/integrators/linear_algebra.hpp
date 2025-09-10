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

// LU decomposition with partial pivoting
template<size_type N, bool AllowPivoting = true>
int lu_decomposition(std::array<std::array<Real, N>, N>& A, 
                     std::array<int, N>& pivot) {
    
    // Initialize pivot array
    for (size_type i = 0; i < N; ++i) {
        pivot[i] = static_cast<int>(i);
    }
    
    for (size_type k = 0; k < N - 1; ++k) {
        
        if constexpr (AllowPivoting) {
            // Find pivot
            size_type pivot_row = k;
            Real max_val = std::abs(A[k][k]);
            
            for (size_type i = k + 1; i < N; ++i) {
                Real val = std::abs(A[i][k]);
                if (val > max_val) {
                    max_val = val;
                    pivot_row = i;
                }
            }
            
            // Swap rows if needed
            if (pivot_row != k) {
                std::swap(A[k], A[pivot_row]);
                std::swap(pivot[k], pivot[pivot_row]);
            }
        }
        
        // Check for zero pivot
        if (std::abs(A[k][k]) < math::UROUND) {
            return static_cast<int>(k + 1); // Return 1-based index of zero pivot
        }
        
        // Eliminate below pivot
        for (size_type i = k + 1; i < N; ++i) {
            A[i][k] /= A[k][k];
            for (size_type j = k + 1; j < N; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
    
    // Check final pivot
    if (std::abs(A[N-1][N-1]) < math::UROUND) {
        return static_cast<int>(N);
    }
    
    return 0; // Success
}

// Solve Ax = b using LU decomposition
template<size_type N, bool AllowPivoting = true>
void lu_solve(const std::array<std::array<Real, N>, N>& LU,
              const std::array<int, N>& pivot,
              std::array<Real, N>& x) {
    
    if constexpr (AllowPivoting) {
        // Apply pivoting to x
        std::array<Real, N> temp = x;
        for (size_type i = 0; i < N; ++i) {
            x[i] = temp[pivot[i]];
        }
    }
    
    // Forward substitution (solve Ly = Px)
    for (size_type i = 1; i < N; ++i) {
        for (size_type j = 0; j < i; ++j) {
            x[i] -= LU[i][j] * x[j];
        }
    }
    
    // Backward substitution (solve Ux = y)
    for (int i = static_cast<int>(N) - 1; i >= 0; --i) {
        for (size_type j = static_cast<size_type>(i) + 1; j < N; ++j) {
            x[i] -= LU[i][j] * x[j];
        }
        x[i] /= LU[i][i];
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
