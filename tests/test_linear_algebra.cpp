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

void test_gth_row_stochastic() {
    std::cout << "Testing GTH factorization (row-stochastic)...\n";

    std::array<std::array<Real, 3>, 3> P = {{
        {{0.5, 0.4, 0.1}},
        {{0.3, 0.3, 0.4}},
        {{0.2, 0.4, 0.4}}
    }};

    std::array<std::array<Real, 3>, 3> Pfact = P;
    std::array<Real, 3> inv_piv;
    int info = linalg::gth_factorization<3>(Pfact, inv_piv, linalg::GthMatrixKind::RowStochastic);
    assert(info == 0);
    (void)info;

    std::array<Real, 3> pi;
    info = linalg::gth_solve<3>(Pfact, inv_piv, pi, 1.0, linalg::GthMatrixKind::RowStochastic);
    assert(info == 0);
    (void)info;

    // Solve reference via LU on (P^T - I)x = 0 with normalization constraint
    std::array<std::array<Real, 3>, 3> A = {{
        {{P[0][0] - 1.0, P[1][0], P[2][0]}},
        {{P[0][1], P[1][1] - 1.0, P[2][1]}},
        {{1.0, 1.0, 1.0}}
    }};
    std::array<Real, 3> b = {0.0, 0.0, 1.0};
    std::array<int, 3> pivot;
    info = linalg::lu_decomposition<3, true>(A, pivot);
    assert(info == 0);
    linalg::lu_solve<3, true>(A, pivot, b);

    Real tol = 1.e-10;
    for (size_type i = 0; i < 3; ++i) {
        assert(std::abs(pi[i] - b[i]) < tol);
    }
    (void)tol;

    std::cout << "  GTH row-stochastic: PASSED\n";
}

void test_gth_generator() {
    std::cout << "Testing GTH factorization (generator)...\n";

    // Rates: from state i to j
    std::array<std::array<Real, 3>, 3> rates = {{
        {{0.0, 0.5, 0.2}},
        {{0.1, 0.0, 0.3}},
        {{0.2, 0.4, 0.0}}
    }};

    std::array<std::array<Real, 3>, 3> Q{};
    for (size_type i = 0; i < 3; ++i) {
        Real diag = 0.0;
        for (size_type j = 0; j < 3; ++j) {
            if (i == j) {
                continue;
            }
            Q[i][j] = -rates[i][j];
            diag += rates[i][j];
        }
        Q[i][i] = diag;
    }

    std::array<std::array<Real, 3>, 3> Qfact = Q;
    std::array<Real, 3> inv_piv;
    int info = linalg::gth_factorization<3>(Qfact, inv_piv, linalg::GthMatrixKind::Generator);
    assert(info == 0);
    (void)info;

    std::array<Real, 3> x;
    info = linalg::gth_solve<3>(Qfact, inv_piv, x, 1.0, linalg::GthMatrixKind::Generator);
    assert(info == 0);
    (void)info;

    // Reference solution via LU on transpose with conservation row.
    std::array<std::array<Real, 3>, 3> A = {{
        {{Q[0][0], Q[1][0], Q[2][0]}},
        {{Q[0][1], Q[1][1], Q[2][1]}},
        {{1.0, 1.0, 1.0}}
    }};
    std::array<Real, 3> b = {0.0, 0.0, 1.0};
    std::array<int, 3> pivot;
    info = linalg::lu_decomposition<3, true>(A, pivot);
    assert(info == 0);
    linalg::lu_solve<3, true>(A, pivot, b);

    Real tol = 1.e-10;
    for (size_type i = 0; i < 3; ++i) {
        assert(std::abs(x[i] - b[i]) < tol);
    }
    (void)tol;

    std::cout << "  GTH generator: PASSED\n";
}

int main() {
    std::cout << "Linear Algebra Test Suite\n";
    std::cout << "=========================\n\n";
    
    test_lu_decomposition();
    test_matrix_solve();
    test_vector_norms();
    test_matrix_vector();
    test_gth_row_stochastic();
    test_gth_generator();
    
    std::cout << "\nAll linear algebra tests PASSED!\n";
    return 0;
}
