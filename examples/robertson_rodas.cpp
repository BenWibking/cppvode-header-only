// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Robertson chemical kinetics problem example using the C++ RODAS integrator
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <integrators/integrators.hpp>

struct RobertsonRODAS {
    static constexpr integrators::size_type neqs = 3;

    using state_type = std::array<integrators::Real, neqs>;
    using rhs_type = std::array<integrators::Real, neqs>;
    using jacobian_type = std::array<std::array<integrators::Real, neqs>, neqs>;

    static void rhs([[maybe_unused]] integrators::Real t, const state_type& y, rhs_type& dydt) {
        constexpr integrators::Real k1 = 0.04;
        constexpr integrators::Real k2 = 1.0e4;
        constexpr integrators::Real k3 = 3.0e7;

        dydt[0] = -k1 * y[0] + k2 * y[1] * y[2];
        dydt[1] = k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] * y[1];
        dydt[2] = k3 * y[1] * y[1];
    }

    static void jacobian([[maybe_unused]] integrators::Real t, const state_type& y, jacobian_type& jac) {
        constexpr integrators::Real k1 = 0.04;
        constexpr integrators::Real k2 = 1.0e4;
        constexpr integrators::Real k3 = 3.0e7;

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
    using namespace integrators;

    auto integrator = RODAS<RobertsonRODAS>{};
    auto state = RODASState<3>{};
    state.jacobian_analytic = true;
    state.autonomous = true;
    state.t = 0.0;
    state.tout = 40.0;
    state.dt = 1.e-6;
    state.rtol = 1.e-6;
    state.atol = 1.e-10;
    state.y = {1.0, 0.0, 0.0};

    auto problem_state = RobertsonRODAS::state_type{1.0, 0.0, 0.0};
    const auto result = integrator.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        std::cerr << "RODAS failed with code " << static_cast<int>(result) << "\n";
        return 1;
    }

    const auto total = state.y[0] + state.y[1] + state.y[2];
    std::cout << std::setprecision(17);
    std::cout << "C++ RODAS Robertson final state at t=40\n";
    std::cout << "y = " << state.y[0] << " " << state.y[1] << " " << state.y[2] << "\n";
    std::cout << "conservation_error = " << std::abs(total - 1.0) << "\n";
    std::cout << "steps = " << state.n_step << "\n";
    std::cout << "accepted = " << state.n_accept << "\n";
    std::cout << "rejected = " << state.n_reject << "\n";
    std::cout << "rhs_evals = " << state.n_rhs << "\n";
    std::cout << "jacobian_evals = " << state.n_jac << "\n";

    return std::abs(total - 1.0) < 1.e-10 ? 0 : 1;
}
