// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Demonstrate GTH factorization on a Markov transition matrix
#include <array>
#include <iostream>
#include <integrators/linear_algebra.hpp>

using namespace integrators;

int main() {
    constexpr std::size_t N = 4;
    using Matrix = std::array<std::array<Real, N>, N>;
    using Vector = std::array<Real, N>;

    // Row-stochastic transition matrix with a nearly absorbing state.
    Matrix P = {{
        {{0.85, 0.10, 0.05, 0.0}},
        {{0.05, 0.90, 0.05, 0.0}},
        {{0.10, 0.10, 0.75, 0.05}},
        {{0.00, 0.10, 0.40, 0.50}}
    }};

    Matrix Pfact = P;
    Vector inv_piv{};
    int info = linalg::gth_factorization<N>(Pfact, inv_piv, linalg::GthMatrixKind::RowStochastic);
    if (info != 0) {
        std::cerr << "GTH factorization failed with info = " << info << "\n";
        return 1;
    }

    Vector pi{};
    info = linalg::gth_solve<N>(Pfact, inv_piv, pi, 1.0, linalg::GthMatrixKind::RowStochastic);
    if (info != 0) {
        std::cerr << "GTH solver failed with info = " << info << "\n";
        return 1;
    }

    std::cout.setf(std::ios::scientific);
    std::cout << "Stationary distribution:\n";
    for (std::size_t i = 0; i < N; ++i) {
        std::cout << "  pi[" << i << "] = " << pi[i] << "\n";
    }

    // Verify normalization and residual.
    Real sum = 0.0;
    for (Real val : pi) sum += val;

    std::cout << "Sum(pi) = " << sum << "\n";
    return 0;
}
