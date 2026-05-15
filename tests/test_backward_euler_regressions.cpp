// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Regression tests for Backward Euler numerical correctness bugs
#include <cassert>
#include <cmath>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

struct Growth {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs([[maybe_unused]] Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = y[0];
    }

    static void jacobian([[maybe_unused]] Real t, [[maybe_unused]] const state_type& y, jacobian_type& jac) {
        jac[0][0] = 1.0;
    }
};

struct Decay {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs([[maybe_unused]] Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = -y[0];
    }

    static void jacobian([[maybe_unused]] Real t, [[maybe_unused]] const state_type& y, jacobian_type& jac) {
        jac[0][0] = -1.0;
    }
};

void test_failed_single_step_preserves_state() {
    auto integrator = BackwardEuler<Growth>{};
    auto state = BackwardEulerState<1>{};
    Growth::state_type problem_state = {1.0};

    state.t = 0.0;
    state.tout = 1.0;
    state.dt = 1.0;
    state.y = {1.0};
    state.jacobian_analytic = true;

    const Real t_old = state.t;
    const Real y_old = state.y[0];

    const auto result = integrator.integrate(problem_state, state);
    assert(result == IntegratorResult::LU_DECOMPOSITION_ERROR);
    assert(state.n_step == 0);
    assert(state.t == t_old);
    assert(state.y[0] == y_old);
    (void)t_old;
    (void)y_old;
    (void)result;
}

void test_reverse_time_integration() {
    auto integrator = BackwardEuler<Decay>{};
    auto state = BackwardEulerState<1>{};
    Decay::state_type problem_state = {std::exp(-1.0)};

    state.t = 1.0;
    state.tout = 0.0;
    state.dt = 0.0;
    state.y = {std::exp(-1.0)};
    state.jacobian_analytic = true;
    state.rtol = 1.e-6;
    state.atol = 1.e-12;

    const auto result = integrator.integrate(problem_state, state);
    assert(result == IntegratorResult::SUCCESS);
    assert(state.t == state.tout);
    assert(std::abs(state.y[0] - 1.0) < 5.e-3);
    (void)result;
}

void test_zero_solution_converges() {
    auto integrator = BackwardEuler<Decay>{};
    auto state = BackwardEulerState<1>{};
    Decay::state_type problem_state = {0.0};

    state.t = 0.0;
    state.tout = 1.0;
    state.dt = 1.0;
    state.y = {0.0};
    state.jacobian_analytic = true;
    state.tolerance = 1.e-12;

    const auto result = integrator.integrate(problem_state, state);
    assert(result == IntegratorResult::SUCCESS);
    assert(state.y[0] == 0.0);
    (void)result;
}

void test_numerical_jacobian_rhs_count() {
    auto integrator = BackwardEuler<Decay>{};
    auto state = BackwardEulerState<1>{};
    Decay::state_type problem_state = {1.0};

    state.t = 0.0;
    state.tout = 0.25;
    state.dt = 0.25;
    state.y = {1.0};
    state.jacobian_analytic = false;
    state.max_iter = 1;
    state.tolerance = 1.e6;

    const auto result = integrator.integrate(problem_state, state);
    assert(result == IntegratorResult::SUCCESS);
    // single_step predictor + Newton RHS + (base + perturbed) in numerical_jacobian for N=1
    assert(state.n_rhs == 4);
    (void)result;
}

int main() {
    std::cout << "Backward Euler regression tests\n";
    test_failed_single_step_preserves_state();
    test_reverse_time_integration();
    test_zero_solution_converges();
    test_numerical_jacobian_rhs_count();
    std::cout << "Backward Euler regressions: PASSED\n";
    return 0;
}
