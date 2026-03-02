// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Unit tests for linear algebra utilities
// ABOUTME: Tests LU decomposition, matrix solve, and vector operations
#include <iostream>
#include <cassert>
#include <cmath>
#include <integrators/linear_algebra.hpp>

using namespace integrators;

void test_lu_decomposition() {
    std::cout << "Testing LU decomposition...\n";
    
    // Test 3x3 matrix
    std::array<std::array<Real, 3>, 3> A = {{
        {{2.0, 1.0, 1.0}},
        {{1.0, 3.0, 2.0}},
        {{1.0, 0.0, 0.0}}
    }};
    
    std::array<int, 3> pivot;
    int info = linalg::lu_decomposition<3, true>(A, pivot);
    assert(info == 0);
    (void)info; // ensure used in Release builds
    std::cout << "  LU decomposition: PASSED\n";
}

void test_matrix_solve() {
    std::cout << "Testing matrix solve...\n";
    
    // Solve Ax = b where A = [[2,1],[1,2]], b = [3,3], x should be [1,1]
    std::array<std::array<Real, 2>, 2> A = {{
        {{2.0, 1.0}},
        {{1.0, 2.0}}
    }};
    
    std::array<Real, 2> b = {3.0, 3.0};
    std::array<int, 2> pivot;
    
    int info = linalg::lu_decomposition<2, true>(A, pivot);
    assert(info == 0);
    (void)info; // ensure used in Release builds
    
    linalg::lu_solve<2, true>(A, pivot, b);
    
    // Check solution
    Real tol = 1.e-12;
    assert(std::abs(b[0] - 1.0) < tol);
    assert(std::abs(b[1] - 1.0) < tol);
    (void)tol; // silence unused when NDEBUG
    
    std::cout << "  Matrix solve: PASSED\n";
}

void test_matrix_solve_regression() {
    std::cout << "Testing LU solve regression case...\n";

    std::array<std::array<Real, 3>, 3> A = {{
        {{0.96181280249950907, 0.95268950339467517, -0.74900493855137884}},
        {{0.53100510298151371, 1.0717186887085923, -0.85952813616124646}},
        {{-0.57964266413365473, 0.32320129774529782, 2.4022762414188135}}
    }};
    std::array<Real, 3> b = {
        -0.040791658492035136,
        -0.59652843561031799,
        1.2119314898967246
    };
    std::array<Real, 3> x_true = {
        0.97497543198967151,
        -0.40292336836364195,
        0.79395290575768396
    };

    std::array<int, 3> pivot;
    int info = linalg::lu_decomposition<3, true>(A, pivot);
    assert(info == 0);
    (void)info;

    linalg::lu_solve<3, true>(A, pivot, b);

    Real max_abs_error = 0.0;
    for (size_type i = 0; i < 3; ++i) {
        max_abs_error = std::max(max_abs_error, std::abs(b[i] - x_true[i]));
    }
    assert(max_abs_error < 1.e-12);
    (void)max_abs_error;

    std::cout << "  LU solve regression: PASSED\n";
}

void test_vector_norms() {
    std::cout << "Testing vector norms...\n";
    
    std::array<Real, 3> v = {3.0, 4.0, 0.0};
    
    Real norm2 = linalg::norm2(v);
    Real expected_norm2 = std::sqrt((9.0 + 16.0) / 3.0); // RMS norm
    assert(std::abs(norm2 - expected_norm2) < 1.e-12);
    (void)norm2; (void)expected_norm2;
    
    Real norm_inf = linalg::norm_inf(v);
    assert(std::abs(norm_inf - 4.0) < 1.e-12);
    (void)norm_inf;
    
    std::cout << "  Vector norms: PASSED\n";
}

void test_matrix_vector() {
    std::cout << "Testing matrix-vector multiplication...\n";
    
    std::array<std::array<Real, 2>, 2> A = {{
        {{1.0, 2.0}},
        {{3.0, 4.0}}
    }};
    
    std::array<Real, 2> x = {1.0, 2.0};
    std::array<Real, 2> y;
    
    linalg::matvec(A, x, y);
    
    assert(std::abs(y[0] - 5.0) < 1.e-12);  // 1*1 + 2*2 = 5
    assert(std::abs(y[1] - 11.0) < 1.e-12); // 3*1 + 4*2 = 11
    
    std::cout << "  Matrix-vector multiplication: PASSED\n";
}

int main() {
    std::cout << "Linear Algebra Test Suite\n";
    std::cout << "=========================\n\n";
    
    test_lu_decomposition();
    test_matrix_solve();
    test_matrix_solve_regression();
    test_vector_norms();
    test_matrix_vector();
    
    std::cout << "\nAll linear algebra tests PASSED!\n";
    return 0;
}
