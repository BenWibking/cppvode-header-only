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
    
    linalg::lu_solve<2, true>(A, pivot, b);
    
    // Check solution
    Real tol = 1.e-12;
    assert(std::abs(b[0] - 1.0) < tol);
    assert(std::abs(b[1] - 1.0) < tol);
    
    std::cout << "  Matrix solve: PASSED\n";
}

void test_vector_norms() {
    std::cout << "Testing vector norms...\n";
    
    std::array<Real, 3> v = {3.0, 4.0, 0.0};
    
    Real norm2 = linalg::norm2(v);
    Real expected_norm2 = std::sqrt((9.0 + 16.0) / 3.0); // RMS norm
    assert(std::abs(norm2 - expected_norm2) < 1.e-12);
    
    Real norm_inf = linalg::norm_inf(v);
    assert(std::abs(norm_inf - 4.0) < 1.e-12);
    
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
    test_vector_norms();
    test_matrix_vector();
    
    std::cout << "\nAll linear algebra tests PASSED!\n";
    return 0;
}