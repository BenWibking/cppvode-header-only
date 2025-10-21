// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Tests for the steady-state departure exponential propagator
#include <cmath>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

namespace {

struct LinearDepartureProblem {
    static constexpr size_type neqs = 2;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs([[maybe_unused]] Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = -1.0 * y[0] + 0.1 * y[1] + 0.1;
        dydt[1] = 0.05 * y[0] - 0.5 * y[1] + 0.05;
    }

    static void jacobian([[maybe_unused]] Real t, [[maybe_unused]] const state_type& y, jacobian_type& jac) {
        jac[0][0] = -1.0;
        jac[0][1] = 0.1;
        jac[1][0] = 0.05;
        jac[1][1] = -0.5;
    }
};

constexpr LinearDepartureProblem::state_type steady_state() {
    return LinearDepartureProblem::state_type{Real{0.1111111111111111}, Real{0.1111111111111111}};
}

bool almost_equal(const LinearDepartureProblem::state_type& a,
                  const LinearDepartureProblem::state_type& b,
                  Real tol) {
    for (size_type i = 0; i < LinearDepartureProblem::neqs; ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool test_homogeneous_propagation() {
    std::cout << "[departure] homogeneous propagation test\n";
    DepartureState<LinearDepartureProblem> state{};
    state.current_time = 0.0;
    state.y_star = steady_state();
    state.y = LinearDepartureProblem::state_type{0.12, 0.10};
    for (size_type i = 0; i < LinearDepartureProblem::neqs; ++i) {
        state.delta[i] = state.y[i] - state.y_star[i];
        state.atol[i] = 1.0e-8;
        state.rtol[i] = 1.0e-6;
    }
    LinearDepartureProblem::jacobian(0.0, state.y_star, state.jacobian);
    LinearDepartureProblem::rhs(0.0, state.y_star, state.rhs_star);

    DepartureWorkspace<LinearDepartureProblem> workspace{};
    DepartureIntegrationConfig config{};
    config.defect_tolerance = 1.0e-3;
    config.rebase_defect = 1.0;
    config.max_departure_norm = 1.0;
    config.max_step = 1.0;

    const Real h_request = 0.2;
    auto initial_delta = state.delta;
    auto result = integrate_departure_step<LinearDepartureProblem>(state, workspace, config, h_request);
    if (result.status == DepartureStepStatus::Failure) {
        std::cout << "  step failed (actual_step=" << result.actual_step
                  << ", suggested=" << result.suggested_step
                  << ", defect=" << result.defect << ")\n";
        return false;
    }
    std::cout << "  status=" << static_cast<int>(result.status)
              << " actual_step=" << result.actual_step
              << " suggested=" << result.suggested_step
              << " defect=" << result.defect << "\n";

    const auto scaled = linalg::matrix_scale(state.jacobian, result.actual_step);
    std::array<std::array<Real, LinearDepartureProblem::neqs>, LinearDepartureProblem::neqs> exp_hA{};
    if (!linalg::matrix_exponential(scaled, exp_hA)) {
        std::cout << "  matrix exponential failed for reference\n";
        return false;
    }

    LinearDepartureProblem::state_type expected_delta{};
    linalg::matvec(exp_hA, initial_delta, expected_delta);
    LinearDepartureProblem::state_type expected_y{};
    for (size_type i = 0; i < LinearDepartureProblem::neqs; ++i) {
        expected_y[i] = state.y_star[i] + expected_delta[i];
    }

    const bool match = almost_equal(state.y, expected_y, 1.0e-11);
    std::cout << "  expected y = [" << expected_y[0] << ", " << expected_y[1] << "]\n";
    std::cout << "  actual y   = [" << state.y[0] << ", " << state.y[1] << "]\n";
    return match;
}

bool test_inhomogeneous_propagation() {
    std::cout << "[departure] inhomogeneous propagation test\n";
    DepartureState<LinearDepartureProblem> state{};
    state.current_time = 0.0;
    state.y_star = steady_state();
    state.y_star[0] += 5.0e-6;
    state.y_star[1] -= 7.5e-6;
    state.y = LinearDepartureProblem::state_type{state.y_star[0] + 2.5e-6,
                                                 state.y_star[1] - 3.0e-6};
    for (size_type i = 0; i < LinearDepartureProblem::neqs; ++i) {
        state.delta[i] = state.y[i] - state.y_star[i];
        state.atol[i] = 1.0e-8;
        state.rtol[i] = 1.0e-6;
    }
    LinearDepartureProblem::jacobian(0.0, state.y_star, state.jacobian);
    LinearDepartureProblem::rhs(0.0, state.y_star, state.rhs_star);

    DepartureWorkspace<LinearDepartureProblem> workspace{};
    DepartureIntegrationConfig config{};
    config.defect_tolerance = 1.0e-3;
    config.rebase_defect = 1.0;
    config.max_departure_norm = 1.0;
    config.max_step = 1.0;

    const Real h_request = 0.05;
    auto initial_delta = state.delta;
    auto rhs_star = state.rhs_star;
    auto result = integrate_departure_step<LinearDepartureProblem>(state, workspace, config, h_request);
    if (result.status == DepartureStepStatus::Failure) {
        std::cout << "  step failed (actual_step=" << result.actual_step
                  << ", suggested=" << result.suggested_step
                  << ", defect=" << result.defect << ")\n";
        return false;
    }
    std::cout << "  status=" << static_cast<int>(result.status)
              << " actual_step=" << result.actual_step
              << " suggested=" << result.suggested_step
              << " defect=" << result.defect << "\n";

    const auto scaled = linalg::matrix_scale(state.jacobian, result.actual_step);
    std::array<std::array<Real, LinearDepartureProblem::neqs>, LinearDepartureProblem::neqs> exp_hA{};
    std::array<std::array<Real, LinearDepartureProblem::neqs>, LinearDepartureProblem::neqs> phi1_hA{};
    if (!linalg::matrix_exponential(scaled, exp_hA) || !linalg::matrix_phi1(scaled, phi1_hA)) {
        std::cout << "  matrix functions failed for reference\n";
        return false;
    }

    LinearDepartureProblem::state_type expected_delta{};
    LinearDepartureProblem::state_type correction{};
    linalg::matvec(exp_hA, initial_delta, expected_delta);
    linalg::matvec(phi1_hA, rhs_star, correction);
    for (size_type i = 0; i < LinearDepartureProblem::neqs; ++i) {
        expected_delta[i] += result.actual_step * correction[i];
    }

    LinearDepartureProblem::state_type expected_y{};
    for (size_type i = 0; i < LinearDepartureProblem::neqs; ++i) {
        expected_y[i] = state.y_star[i] + expected_delta[i];
    }

    const bool match = almost_equal(state.y, expected_y, 1.0e-11);
    std::cout << "  expected y = [" << expected_y[0] << ", " << expected_y[1] << "]\n";
    std::cout << "  actual y   = [" << state.y[0] << ", " << state.y[1] << "]\n";
    return match;
}

bool test_rebase_trigger() {
    std::cout << "[departure] rebase trigger test\n";
    DepartureState<LinearDepartureProblem> state{};
    state.current_time = 0.0;
    state.y_star = steady_state();
    state.y = LinearDepartureProblem::state_type{0.25, 0.05};
    for (size_type i = 0; i < LinearDepartureProblem::neqs; ++i) {
        state.delta[i] = state.y[i] - state.y_star[i];
        state.atol[i] = 1.0e-12;
        state.rtol[i] = 1.0e-10;
    }
    LinearDepartureProblem::jacobian(0.0, state.y_star, state.jacobian);
    LinearDepartureProblem::rhs(0.0, state.y_star, state.rhs_star);

    DepartureWorkspace<LinearDepartureProblem> workspace{};
    DepartureIntegrationConfig config{};
    config.defect_tolerance = 1.0e-10;
    config.rebase_defect = 1.0e-9;
    config.max_departure_norm = 5.0e-2;
    config.max_step = 0.5;

    auto result = integrate_departure_step<LinearDepartureProblem>(state, workspace, config, 0.2);
    const bool rebase = (result.status == DepartureStepStatus::RebaseRequested);
    const bool cache_reset = !workspace.cache_valid;
    std::cout << "  status=" << (rebase ? "rebase" : "other") << ", cache_valid=" << workspace.cache_valid << "\n";
    return rebase && cache_reset;
}

} // namespace

int main() {
    int failures = 0;
    if (!test_homogeneous_propagation()) {
        std::cout << "homogeneous propagation: FAILED\n";
        failures++;
    } else {
        std::cout << "homogeneous propagation: PASSED\n";
    }
    if (!test_inhomogeneous_propagation()) {
        std::cout << "inhomogeneous propagation: FAILED\n";
        failures++;
    } else {
        std::cout << "inhomogeneous propagation: PASSED\n";
    }
    if (!test_rebase_trigger()) {
        std::cout << "rebase trigger: FAILED\n";
        failures++;
    } else {
        std::cout << "rebase trigger: PASSED\n";
    }
    if (failures > 0) {
        std::cout << failures << " steady-state departure test(s) failed\n";
        return 1;
    }
    std::cout << "All steady-state departure tests passed\n";
    return 0;
}
