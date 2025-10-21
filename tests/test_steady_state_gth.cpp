// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Tests steady-state solver using GTH factorization
#include <cmath>
#include <iostream>
#include <integrators/steady_state_gth.hpp>

using namespace integrators;

struct SimpleSteadyProblem {
    static constexpr size_type neqs = 3;
    using state_type = std::array<Real, neqs>;
    using rhs_type = state_type;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void steady_state_generator(const state_type&, jacobian_type& G) {
        // Generator with diagonals summing to zero rows (diag positive, off-diagonals negative)
        G = {{
            {{1.5, -0.5, -1.0}},
            {{-1.0, 1.2, -0.2}},
            {{-0.5, -0.7, 1.2}}
        }};
    }
};

struct ConstrainedSteadyProblem {
    static constexpr size_type neqs = 3;
    using state_type = std::array<Real, neqs>;
    using rhs_type = state_type;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void steady_state_generator(const state_type&, jacobian_type& G) {
        G = {{
            {{2.0, -1.0, -1.0}},
            {{-0.5, 1.4, -0.9}},
            {{-0.5, -0.4, 0.9}}
        }};
    }

    static size_type steady_state_constraint_count() {
        return 2; // total abundance + effective charge
    }

    static void steady_state_constraints(const state_type&, std::array<std::array<Real, neqs>, neqs>& A) {
        // Row 0 (mass) is supplied by the solver; append charge weights here
        A[1] = {{2.0, 1.0, 0.0}};
    }
};

int test_steady_state_solver() {
    SimpleSteadyProblem::state_type y = {0.2, 0.5, 0.3};

    SteadyStateGthConfig config;
    config.iteration_tol = 1.0e-12;
    config.max_iters = 5;
    config.verbose = false;

    const auto result = steady_state_gth<SimpleSteadyProblem>(y, config);

    if (result.status != SteadyStateStatus::Success) {
        std::cerr << "steady_state_gth failed with status "
                  << static_cast<int>(result.status) << "\n";
        return 1;
    }
    if (!(result.residual <= config.iteration_tol)) {
        std::cerr << "Residual too large: " << result.residual << "\n";
        return 1;
    }

    const Real total = std::accumulate(y.begin(), y.end(), Real{0});
    if (std::abs(total - 1.0) >= 1.0e-12) {
        std::cerr << "Normalization error: sum=" << total << "\n";
        return 1;
    }

    SimpleSteadyProblem::jacobian_type G{};
    SimpleSteadyProblem::steady_state_generator(y, G);

    // Verify that G^T * y is approximately zero (stationary condition)
    std::array<Real, SimpleSteadyProblem::neqs> residual{};
    for (size_type i = 0; i < SimpleSteadyProblem::neqs; ++i) {
        Real sum = 0.0;
        for (size_type j = 0; j < SimpleSteadyProblem::neqs; ++j) {
            sum += G[j][i] * y[j];
        }
        residual[i] = sum;
        if (std::abs(sum) >= 1.0e-11) {
            std::cerr << "Residual component " << i << " = " << sum << "\n";
            return 1;
        }
    }
    return 0;
}

int test_constrained_solver() {
    ConstrainedSteadyProblem::state_type y = {0.3, 0.4, 0.3};
    const Real total_initial = std::accumulate(y.begin(), y.end(), Real{0});
    const Real charge_initial = 2.0 * y[0] + 1.0 * y[1] + 0.0 * y[2];

    SteadyStateGthConfig config;
    config.iteration_tol = 1.0e-12;

    const auto result = steady_state_gth<ConstrainedSteadyProblem>(y, config);
    if (result.status != SteadyStateStatus::Success) {
        std::cerr << "steady_state_gth (constrained) failed with status "
                  << static_cast<int>(result.status) << "\n";
        return 1;
    }

    const Real total_final = std::accumulate(y.begin(), y.end(), Real{0});
    const Real charge_final = 2.0 * y[0] + 1.0 * y[1] + 0.0 * y[2];

    if (std::abs(total_final - total_initial) > 1.0e-12) {
        std::cerr << "Total abundance drifted: " << total_final << "\n";
        return 1;
    }
    if (std::abs(charge_final - charge_initial) > 1.0e-12) {
        std::cerr << "Charge invariant drifted: " << charge_final << "\n";
        return 1;
    }
    return 0;
}

int main() {
    int failures = 0;
    failures += test_steady_state_solver();
    failures += test_constrained_solver();
    if (failures != 0) {
        return 1;
    }
    return 0;
}
