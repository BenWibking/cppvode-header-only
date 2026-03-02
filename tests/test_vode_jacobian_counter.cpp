// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Regression test for VODE Jacobian evaluation accounting
#include <cassert>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

struct StiffDecayWithJac {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static inline int jacobian_calls = 0;

    static void rhs([[maybe_unused]] Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = -1000.0 * y[0];
    }

    static void jacobian([[maybe_unused]] Real t, [[maybe_unused]] const state_type& y, jacobian_type& jac) {
        jacobian_calls += 1;
        jac[0][0] = -1000.0;
    }
};

int main() {
    StiffDecayWithJac::jacobian_calls = 0;

    auto integrator = VODE<StiffDecayWithJac>{};
    auto state = VODEState<1>{};
    StiffDecayWithJac::state_type problem_state = {1.0};

    state.t = 0.0;
    state.tout = 1.0;
    state.y = {1.0};
    state.jacobian_analytic = true;
    state.rtol = 1.e-8;
    state.atol = 1.e-12;

    const auto result = integrator.integrate(problem_state, state);
    assert(result == IntegratorResult::SUCCESS);
    assert(state.n_jac > 0);
    assert(state.n_jac == StiffDecayWithJac::jacobian_calls);
    (void)result;

    std::cout << "VODE Jacobian counter regression: PASSED\n";
    return 0;
}
