// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Tests for Khokhlov YASS integrator
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

void require(bool condition) {
    if (!condition) {
        std::cerr << "YASS test assertion failed\n";
        std::abort();
    }
}

struct StiffDecay {
    static constexpr size_type neqs = 1;
    static constexpr Real rate = 4.0;

    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs([[maybe_unused]] Real t, const state_type &y,
                    rhs_type &dydt) {
        dydt[0] = -rate * y[0];
    }

    static void jacobian([[maybe_unused]] Real t,
                         [[maybe_unused]] const state_type &y,
                         jacobian_type &jac) {
        jac[0][0] = -rate;
    }
};

struct RhsOnlyDecay {
    static constexpr size_type neqs = 1;
    static constexpr Real rate = 2.0;

    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;

    static void rhs([[maybe_unused]] Real t, const state_type &y,
                    rhs_type &dydt) {
        dydt[0] = -rate * y[0];
    }
};

struct ReversiblePair {
    static constexpr size_type neqs = 2;
    static constexpr Real forward = 1000.0;
    static constexpr Real reverse = 1.0;

    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs([[maybe_unused]] Real t, const state_type &y,
                    rhs_type &dydt) {
        const Real flux = forward * y[0] - reverse * y[1];
        dydt[0] = -flux;
        dydt[1] = flux;
    }

    static void jacobian([[maybe_unused]] Real t,
                         [[maybe_unused]] const state_type &y,
                         jacobian_type &jac) {
        jac[0][0] = -forward;
        jac[0][1] = reverse;
        jac[1][0] = forward;
        jac[1][1] = -reverse;
    }
};

Real reverse_equilibrium_fraction() {
    return ReversiblePair::reverse /
           (ReversiblePair::forward + ReversiblePair::reverse);
}

void test_scalar_yass_step() {
    auto integrator = YASS<StiffDecay>{};
    auto state = YASSState<1>{};
    StiffDecay::state_type problem_state = {2.0};

    state.t = 0.0;
    state.tout = 0.25;
    state.dt = 0.25;
    state.y = {2.0};
    state.jacobian_analytic = true;

    const auto result = integrator.integrate(problem_state, state);
    const Real expected = 2.0 / (1.0 + StiffDecay::rate * state.dt);

    require(result == IntegratorResult::SUCCESS);
    require(state.t == state.tout);
    require(state.n_step == 1);
    require(state.n_rhs == 1);
    require(state.n_jac == 1);
    require(std::abs(state.y[0] - expected) < 1.e-14);
    (void)result;
}

void test_linear_invariant_is_preserved() {
    auto integrator = YASS<ReversiblePair>{};
    auto state = YASSState<2>{};
    ReversiblePair::state_type problem_state = {1.0, 0.0};

    state.t = 0.0;
    state.tout = 10.0;
    state.dt = 10.0;
    state.y = {1.0, 0.0};
    state.jacobian_analytic = true;

    const Real sum0 = state.y[0] + state.y[1];
    const auto result = integrator.integrate(problem_state, state);
    const Real sum1 = state.y[0] + state.y[1];

    require(result == IntegratorResult::SUCCESS);
    require(std::abs(sum1 - sum0) < 1.e-12);
    require(std::abs(state.y[0] - reverse_equilibrium_fraction()) < 2.e-4);
    (void)result;
}

void test_numerical_jacobian_fallback() {
    auto integrator = YASS<RhsOnlyDecay>{};
    auto state = YASSState<1>{};
    RhsOnlyDecay::state_type problem_state = {1.0};

    state.t = 0.0;
    state.tout = 0.1;
    state.dt = 0.1;
    state.y = {1.0};
    state.jacobian_analytic = false;

    const auto result = integrator.integrate(problem_state, state);
    const Real expected = 1.0 / (1.0 + RhsOnlyDecay::rate * state.dt);

    require(result == IntegratorResult::SUCCESS);
    require(std::abs(state.y[0] - expected) < 1.e-12);
    require(state.n_rhs == 3);
    require(state.n_jac == 1);
    (void)result;
}

void test_factory_support() {
    using Factory = IntegratorFactory<StiffDecay>;
    auto integrator = Factory::create<Factory::Type::YASS>();
    auto state = Factory::state_type<Factory::Type::YASS>{};
    StiffDecay::state_type problem_state = {1.0};

    state.t = 0.0;
    state.tout = 0.1;
    state.dt = 0.1;
    state.y = {1.0};
    state.jacobian_analytic = true;

    const auto result = integrator.integrate(problem_state, state);
    require(result == IntegratorResult::SUCCESS);
    require(state.y[0] < 1.0);
    (void)result;
}

int main() {
    std::cout << "YASS integrator tests\n";
    test_scalar_yass_step();
    test_linear_invariant_is_preserved();
    test_numerical_jacobian_fallback();
    test_factory_support();
    std::cout << "YASS tests: PASSED\n";
    return 0;
}
