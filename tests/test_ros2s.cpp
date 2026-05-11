// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: ROS2S Rosenbrock tableau regression tests
#include <array>
#include <cmath>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

struct ScalarDecay {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs(Real /*t*/, const state_type& y, rhs_type& dydt) {
        dydt[0] = -100.0 * y[0];
    }

    static void jacobian(Real /*t*/, const state_type& /*y*/, jacobian_type& jac) {
        jac[0][0] = -100.0;
    }
};

int main() {
    auto integrator = RODAS<ScalarDecay>{};
    auto state = RODASState<1>{};
    state.jacobian_analytic = true;
    state.autonomous = true;
    state.t = 0.0;
    state.tout = 0.1;
    state.dt = 1.e-3;
    state.rtol = 1.e-8;
    state.atol = 1.e-12;
    state.y = {1.0};

    auto problem_state = ScalarDecay::state_type{1.0};
    const auto result = integrator.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        std::cerr << "ROS2S failed with code " << static_cast<int>(result) << "\n";
        return 1;
    }

    const Real exact = std::exp(-10.0);
    const Real err = std::abs(state.y[0] - exact);
    if (err > 5.e-7) {
        std::cerr << "ROS2S scalar decay error too large: got " << state.y[0]
                  << " expected " << exact << " err=" << err << "\n";
        return 1;
    }
    if (problem_state[0] != state.y[0]) {
        std::cerr << "problem_state not synchronized\n";
        return 1;
    }

    std::cout << "ROS2S scalar decay: PASSED steps=" << state.n_step
              << " accepted=" << state.n_accept << " rejected=" << state.n_reject << "\n";
    return 0;
}
