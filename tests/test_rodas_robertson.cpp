// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: RODAS regression test on the stiff Robertson problem
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

struct RobertsonRODAS {
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

int main() {
    auto integrator = RODAS<RobertsonRODAS>{};
    auto state = RODASState<3>{};
    state.jacobian_analytic = true;
    state.autonomous = true;
    state.t = 0.0;
    state.tout = 1.0;
    state.dt = 1.e-6;
    state.rtol = 1.e-10;
    state.atol = 1.e-12;
    state.y = {1.0, 0.0, 0.0};

    auto problem_state = RobertsonRODAS::state_type{1.0, 0.0, 0.0};
    const auto result = integrator.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        std::cerr << "RODAS failed with code " << static_cast<int>(result) << "\n";
        return 1;
    }

    const Real total = state.y[0] + state.y[1] + state.y[2];
    if (std::abs(total - 1.0) >= 1.e-12) {
        std::cerr << "conservation failed: total=" << total << "\n";
        return 1;
    }
    for (size_type i = 0; i < 3; ++i) {
        if (problem_state[i] != state.y[i]) {
            std::cerr << "problem_state not synchronized at component " << i << "\n";
            return 1;
        }
    }

    // Dense identity-mass RODAS regression for the autonomous Robertson
    // problem with the same tolerances.
    constexpr std::array<Real, 3> ref{
        0.9664597373325807,
        3.0746265766362e-5,
        0.03350951640165291,
    };
    for (size_type i = 0; i < 3; ++i) {
        const Real scale = std::max<Real>(1.0, std::abs(ref[i]));
        if (std::abs(state.y[i] - ref[i]) > 5.e-13 * scale) {
            std::cerr << std::setprecision(17)
                      << "component " << i << " differs: got " << state.y[i]
                      << " expected " << ref[i] << "\n";
            return 1;
        }
    }

    std::cout << "RODAS Robertson: PASSED steps=" << state.n_step
              << " accepted=" << state.n_accept << " rejected=" << state.n_reject << "\n";
    return 0;
}
