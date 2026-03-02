// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Regression test for optional Jacobian API contract
#include <cassert>
#include <cmath>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

struct RhsOnlyDecay {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;

    static void rhs([[maybe_unused]] Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = -2.0 * y[0];
    }
};

int main() {
    {
        auto integrator = BackwardEuler<RhsOnlyDecay>{};
        auto state = BackwardEulerState<1>{};
        RhsOnlyDecay::state_type problem_state = {1.0};

        state.t = 0.0;
        state.tout = 0.1;
        state.dt = 0.05;
        state.y = {1.0};
        state.jacobian_analytic = false;

        const auto result = integrator.integrate(problem_state, state);
        assert(result == IntegratorResult::SUCCESS);
        assert(state.y[0] < 1.0);
        (void)result;
    }

    {
        auto integrator = VODE<RhsOnlyDecay>{};
        auto state = VODEState<1>{};
        RhsOnlyDecay::state_type problem_state = {1.0};

        state.t = 0.0;
        state.tout = 0.1;
        state.y = {1.0};
        state.jacobian_analytic = false;
        state.rtol = 1.e-8;
        state.atol = 1.e-12;

        const auto result = integrator.integrate(problem_state, state);
        assert(result == IntegratorResult::SUCCESS);
        assert(std::abs(state.t - state.tout) < 1.e-15);
        assert(state.y[0] < 1.0);
        (void)result;
    }

    std::cout << "Optional Jacobian regression: PASSED\n";
    return 0;
}
