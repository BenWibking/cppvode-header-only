// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Validate GTH factorization on Robertson kinetics generator
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <integrators/integrators.hpp>
#include <integrators/linear_algebra.hpp>

using namespace integrators;

namespace {

struct Robertson {
    static constexpr size_type neqs = 3;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs(Real /*t*/, const state_type& y, rhs_type& dydt) {
        constexpr Real k1 = 0.04;
        constexpr Real k2 = 1.0e4;
        constexpr Real k3 = 3.0e7;
        dydt[0] = -k1 * y[0] + k2 * y[1] * y[2];
        dydt[1] = k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] * y[1];
        dydt[2] = k3 * y[1] * y[1];
    }

    static void jacobian(Real /*t*/, const state_type& y, jacobian_type& jac) {
        constexpr Real k1 = 0.04;
        constexpr Real k2 = 1.0e4;
        constexpr Real k3 = 3.0e7;
        jac[0][0] = -k1;
        jac[0][1] = k2 * y[2];
        jac[0][2] = k2 * y[1];
        jac[1][0] = k1;
        jac[1][1] = -k2 * y[2] - 2.0 * k3 * y[1];
        jac[1][2] = -k2 * y[1];
        jac[2][0] = 0.0;
        jac[2][1] = 2.0 * k3 * y[1];
        jac[2][2] = 0.0;
    }
};

std::array<std::array<Real, 3>, 3> build_generator(const Robertson::state_type& y) {
    constexpr Real k1 = 0.04;
    constexpr Real k2 = 1.0e4;
    constexpr Real k3 = 3.0e7;

    // Order states as [C, B, A] to satisfy GTH elimination prerequisites.
    const Real rate_b_to_c = k3 * y[1];
    const Real rate_b_to_a = k2 * y[1] * y[2];
    const Real rate_a_to_b = k1;

    std::array<std::array<Real, 3>, 3> Q{};
    // Row 0 : C (absorbing in this reduced model)
    Q[0][0] = 0.0;
    Q[0][1] = 0.0;
    Q[0][2] = 0.0;

    // Row 1 : B
    Q[1][0] = rate_b_to_c;
    Q[1][1] = -(rate_b_to_c + rate_b_to_a);
    Q[1][2] = rate_b_to_a;

    // Row 2 : A
    Q[2][0] = 0.0;
    Q[2][1] = rate_a_to_b;
    Q[2][2] = -rate_a_to_b;

    return Q;
}

} // namespace

int main() {
    // Integrate the Robertson system briefly to obtain a physically consistent state.
    constexpr Real final_time = 1.0e-6;

    VODE<Robertson> vode;
    VODEState<Robertson::neqs> state;
    state.t = 0.0;
    state.tout = final_time;
    state.y = {1.0, 0.0, 0.0};
    state.rtol = 1.0e-8;
    state.atol = 1.0e-12;
    state.jacobian_analytic = true;
    state.max_steps = 500000;

    Robertson::state_type y0 = {1.0, 0.0, 0.0};
    const auto status = vode.integrate(y0, state);
    if (status != IntegratorResult::SUCCESS) {
        std::cerr << "VODE integration failed with status "
                  << static_cast<int>(status) << "\n";
        return 1;
    }

    const auto Q = build_generator(state.y);

    // Run GTH factorization on the generator.
    auto Qfact = Q;
    std::array<Real, 3> inv_piv{};
    const int gth_info = linalg::gth_factorization<3>(Qfact, inv_piv, linalg::GthMatrixKind::Generator);
    if (gth_info != 0) {
        std::cerr << "GTH factorization failed with info = " << gth_info << "\n";
        return 1;
    }

    std::array<Real, 3> pi_cba{};
    const int solve_info = linalg::gth_solve<3>(Qfact, inv_piv, pi_cba, 1.0, linalg::GthMatrixKind::Generator);
    if (solve_info != 0) {
        std::cerr << "GTH solver failed with info = " << solve_info << "\n";
        return 1;
    }

    // Reference solution via standard LU on the same generator (transpose form).
    std::array<std::array<Real, 3>, 3> A{};
    for (size_type row = 0; row < 3; ++row) {
        for (size_type col = 0; col < 3; ++col) {
            A[row][col] = Q[col][row];
        }
    }
    A[2][0] = 1.0;
    A[2][1] = 1.0;
    A[2][2] = 1.0;

    std::array<Real, 3> b = {0.0, 0.0, 1.0};
    std::array<int, 3> piv{};
    const int lu_info = linalg::lu_decomposition<3, true>(A, piv);
    if (lu_info != 0) {
        std::cerr << "LU factorization failed with info = " << lu_info << "\n";
        return 1;
    }
    linalg::lu_solve<3, true>(A, piv, b);

    // Compare GTH and LU results (in [C,B,A] order).
    const Real tol = 1.0e-10;
    Real max_diff = 0.0;
    for (size_type i = 0; i < 3; ++i) {
        const Real diff = std::abs(pi_cba[i] - b[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > tol) {
            std::cerr << "Mismatch between GTH and LU at index " << i
                      << ": GTH=" << pi_cba[i] << " LU=" << b[i] << "\n";
            return 1;
        }
    }

    // Confirm that the stationary distribution concentrates on species C in original ordering.
    const std::array<Real, 3> pi_original = {pi_cba[2], pi_cba[1], pi_cba[0]};
    if (std::abs(pi_original[2] - 1.0) > 1.0e-10 ||
        pi_original[0] < -1.0e-12 || pi_original[1] < -1.0e-12) {
        std::cerr << "Unexpected stationary distribution: "
                  << pi_original[0] << ", "
                  << pi_original[1] << ", "
                  << pi_original[2] << "\n";
        return 1;
    }

    std::cout << std::scientific << std::setprecision(6);
    std::cout << "GTH vs LU max difference: " << max_diff << "\n";
    std::cout << "Stationary distribution (A,B,C): "
              << pi_original[0] << ", "
              << pi_original[1] << ", "
              << pi_original[2] << "\n";
    std::cout.unsetf(std::ios::floatfield);

    return 0;
}
