// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Verify that VODE snaps to steady state before stepping
#include <cmath>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

namespace {

struct SnapProblem {
    static constexpr size_type neqs = 2;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static constexpr Real k01 = 2.0;
    static constexpr Real k10 = 3.0;

    static void rhs([[maybe_unused]] Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = -k01 * y[0] + k10 * y[1];
        dydt[1] = k01 * y[0] - k10 * y[1];
    }

    static void jacobian([[maybe_unused]] Real t, [[maybe_unused]] const state_type& y, jacobian_type& jac) {
        jac[0][0] = -k01;
        jac[0][1] = k10;
        jac[1][0] = k01;
        jac[1][1] = -k10;
    }

    static void steady_state_generator(const state_type&, jacobian_type& gen) {
        gen[0][0] = k01;
        gen[0][1] = -k01;
        gen[1][0] = -k10;
        gen[1][1] = k10;
    }
};

SnapProblem::state_type steady_state() {
    return {
        SnapProblem::k10 / (SnapProblem::k01 + SnapProblem::k10),
        SnapProblem::k01 / (SnapProblem::k01 + SnapProblem::k10)};
}

template<typename State>
bool near(const State& a, const State& b, Real tol) {
    for (size_type i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool test_initial_snap_success() {
    std::cout << "[vode_snap] initial snap attempt\n";

    VODE<SnapProblem> solver;
    VODEState<SnapProblem::neqs> state;
    state.t = 0.0;
    state.tout = 2.0; // hydro dt large enough to satisfy timescale gate
    state.y = {0.2, 0.8};
    state.rtol = 1.0e-6;
    state.atol = 1.0e-12;
    state.jacobian_analytic = true;

    auto problem_state = state.y;
    const auto result = solver.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        std::cout << "  integrate returned status " << static_cast<int>(result) << "\n";
        return false;
    }

    const auto y_star = steady_state();
    const bool snapped = near(state.y, y_star, 1.0e-12);
    std::cout << "  n_step=" << state.n_step
              << " y=[" << state.y[0] << ", " << state.y[1] << "]\n";

    return snapped && state.n_step == 0 && std::abs(state.t - state.tout) < 1.0e-12;
}

bool test_vode_fallback_after_snap_reject() {
    std::cout << "[vode_snap] fallback to VODE when snap gate fails\n";

    VODE<SnapProblem> solver;
    VODEState<SnapProblem::neqs> state;
    state.t = 0.0;
    state.tout = 0.1; // hydro dt too small for the timescale gate
    state.y = {0.2, 0.8};
    state.rtol = 1.0e-6;
    state.atol = 1.0e-12;
    state.jacobian_analytic = true;
    state.max_steps = 1000;

    auto problem_state = state.y;
    const auto result = solver.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        std::cout << "  integrate returned status " << static_cast<int>(result) << "\n";
        return false;
    }

    const auto y_star = steady_state();
    const bool not_snapped = !near(state.y, y_star, 1.0e-6);
    std::cout << "  n_step=" << state.n_step
              << " y=[" << state.y[0] << ", " << state.y[1] << "]\n";

    return state.n_step > 0 && not_snapped;
}

bool test_snap_disabled() {
    std::cout << "[vode_snap] disabled snap falls back to VODE\n";

    VODE<SnapProblem> solver;
    VODEState<SnapProblem::neqs> state;
    state.t = 0.0;
    state.tout = 0.1;
    state.y = {0.2, 0.8};
    state.rtol = 1.0e-6;
    state.atol = 1.0e-12;
    state.jacobian_analytic = true;
    state.steady_state_snap_enabled = false;

    auto problem_state = state.y;
    const auto result = solver.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        std::cout << "  integrate returned status " << static_cast<int>(result) << "\n";
        return false;
    }

    return state.n_step > 0;
}

} // namespace

int main() {
    int failures = 0;
    if (!test_initial_snap_success()) {
        std::cout << "initial snap test: FAILED\n";
        failures++;
    }
    if (!test_vode_fallback_after_snap_reject()) {
        std::cout << "fallback test: FAILED\n";
        failures++;
    }
    if (!test_snap_disabled()) {
        std::cout << "disabled snap test: FAILED\n";
        failures++;
    }

    if (failures > 0) {
        std::cout << failures << " tests FAILED\n";
        return 1;
    }

    std::cout << "All VODE snap tests passed.\n";
    return 0;
}
