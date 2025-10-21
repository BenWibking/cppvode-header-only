// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Tests for the steady-state snap helper
#include <cmath>
#include <iostream>
#include <limits>
#include <integrators/integrators.hpp>

using namespace integrators;

namespace {

struct TwoStateProblem {
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

struct CycleProblem {
    static constexpr size_type neqs = 3;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static constexpr Real k12 = 4.0;
    static constexpr Real k23 = 2.0;
    static constexpr Real k31 = 1.0;

    static void rhs([[maybe_unused]] Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = -k12 * y[0] + k31 * y[2];
        dydt[1] = k12 * y[0] - k23 * y[1];
        dydt[2] = k23 * y[1] - k31 * y[2];
    }

    static void jacobian([[maybe_unused]] Real t, [[maybe_unused]] const state_type& y, jacobian_type& jac) {
        jac[0][0] = -k12;
        jac[0][1] = 0.0;
        jac[0][2] = k31;
        jac[1][0] = k12;
        jac[1][1] = -k23;
        jac[1][2] = 0.0;
        jac[2][0] = 0.0;
        jac[2][1] = k23;
        jac[2][2] = -k31;
    }

    static void steady_state_generator(const state_type&, jacobian_type& gen) {
        gen[0][0] = k12;
        gen[0][1] = -k12;
        gen[0][2] = 0.0;
        gen[1][0] = 0.0;
        gen[1][1] = k23;
        gen[1][2] = -k23;
        gen[2][0] = -k31;
        gen[2][1] = 0.0;
        gen[2][2] = k31;
    }
};

template<typename Problem>
bool is_close(const typename Problem::state_type& a,
              const typename Problem::state_type& b,
              Real tol) {
    for (size_type i = 0; i < Problem::neqs; ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool test_snap_success() {
    std::cout << "[snap] success path\n";

    TwoStateProblem::state_type y = {0.2, 0.8};
    TwoStateProblem::state_type atol = {1.0e-10, 1.0e-10};
    TwoStateProblem::state_type rtol = {1.0e-8, 1.0e-8};

    SteadyStateSnapConfig config{};
    config.balance_tolerance = 1.0e-6;
    config.timescale_safety = 0.5;
    config.max_departure_tolerance = std::numeric_limits<Real>::infinity();

    const auto outcome = attempt_steady_state_snap<TwoStateProblem>(
        0.0, y, atol, rtol, 10.0, true, config);

    const TwoStateProblem::state_type y_star = {
        TwoStateProblem::k10 / (TwoStateProblem::k01 + TwoStateProblem::k10),
        TwoStateProblem::k01 / (TwoStateProblem::k01 + TwoStateProblem::k10)};

    std::cout << "  result=" << static_cast<int>(outcome.result)
              << " max_balance=" << outcome.max_balance_mismatch
              << " max_tau=" << outcome.max_timescale << "\n";

    return outcome.result == SteadyStateSnapResult::Snapped &&
           is_close<TwoStateProblem>(y, y_star, 1.0e-12);
}

bool test_fallback_due_to_balance() {
    std::cout << "[snap] detailed balance rejection\n";

    CycleProblem::state_type y = {0.5, 0.3, 0.2};
    CycleProblem::state_type atol = {1.0e-10, 1.0e-10, 1.0e-10};
    CycleProblem::state_type rtol = {1.0e-8, 1.0e-8, 1.0e-8};

    SteadyStateSnapConfig config{};
    config.balance_tolerance = 1.0e-6;
    config.timescale_safety = 1.0;

    const auto outcome = attempt_steady_state_snap<CycleProblem>(
        0.0, y, atol, rtol, 1.0, true, config);

    std::cout << "  result=" << static_cast<int>(outcome.result)
              << " max_balance=" << outcome.max_balance_mismatch << "\n";

    return outcome.result == SteadyStateSnapResult::FallbackToVode &&
           outcome.max_balance_mismatch > config.balance_tolerance;
}

bool test_failure_without_fallback() {
    std::cout << "[snap] abort when VODE already failed\n";

    CycleProblem::state_type y = {0.5, 0.3, 0.2};
    CycleProblem::state_type atol = {1.0e-10, 1.0e-10, 1.0e-10};
    CycleProblem::state_type rtol = {1.0e-8, 1.0e-8, 1.0e-8};

    SteadyStateSnapConfig config{};
    config.balance_tolerance = 1.0e-6;
    config.timescale_safety = 1.0;

    const auto outcome = attempt_steady_state_snap<CycleProblem>(
        0.0, y, atol, rtol, 1.0, false, config);

    return outcome.result == SteadyStateSnapResult::Failure;
}

bool test_fallback_due_to_timescale() {
    std::cout << "[snap] timescale rejection\n";

    TwoStateProblem::state_type y = {0.9, 0.1};
    TwoStateProblem::state_type atol = {1.0e-10, 1.0e-10};
    TwoStateProblem::state_type rtol = {1.0e-8, 1.0e-8};

    SteadyStateSnapConfig config{};
    config.balance_tolerance = 1.0e-6;
    config.timescale_safety = 0.1;

    const auto outcome = attempt_steady_state_snap<TwoStateProblem>(
        0.0, y, atol, rtol, 0.01, true, config);

    std::cout << "  result=" << static_cast<int>(outcome.result)
              << " max_tau=" << outcome.max_timescale << "\n";

    const Real threshold = config.timescale_safety * 0.01;
    return outcome.result == SteadyStateSnapResult::FallbackToVode &&
           outcome.max_timescale > threshold;
}

} // namespace

int main() {
    int failures = 0;

    if (!test_snap_success()) {
        std::cout << "snap success test: FAILED\n";
        failures++;
    }
    if (!test_fallback_due_to_balance()) {
        std::cout << "balance rejection test: FAILED\n";
        failures++;
    }
    if (!test_failure_without_fallback()) {
        std::cout << "failure without fallback test: FAILED\n";
        failures++;
    }
    if (!test_fallback_due_to_timescale()) {
        std::cout << "timescale rejection test: FAILED\n";
        failures++;
    }

    if (failures > 0) {
        std::cout << failures << " tests FAILED\n";
        return 1;
    }

    std::cout << "All steady-state snap tests passed.\n";
    return 0;
}
