// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Convergence tests for integrator algorithms
// ABOUTME: Tests order of accuracy and convergence rates on simple test problems
#include <iostream>
#include <cmath>
#include <vector>
#include <integrators/integrators.hpp>

using namespace integrators;

// Test problem: y' = y, y(0) = 1, exact solution = e^t
struct ExponentialGrowth {
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

template<typename Integrator, typename State>
Real compute_error(Integrator& integrator, State& state, Real dt) {
    ExponentialGrowth::state_type problem_state = {1.0};
    
    state.t = 0.0;
    state.tout = dt;
    state.y[0] = 1.0;
    state.rtol = 1.e-12;
    state.atol = 1.e-16;
    
    auto result = integrator.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        return -1.0; // Signal failure
    }
    
    Real exact = std::exp(dt);
    return std::abs(state.y[0] - exact);
}

bool test_backward_euler_convergence() {
    std::cout << "Testing Backward Euler convergence (expected order 1)...\n";
    
    auto integrator = BackwardEuler<ExponentialGrowth>{};
    
    std::vector<Real> dts = {0.1, 0.05, 0.025, 0.0125};
    std::vector<Real> errors;
    
    for (Real dt : dts) {
        auto state = BackwardEulerState<1>{};
        state.jacobian_analytic = true;
        Real error = compute_error(integrator, state, dt);
        if (error < 0.0) {
            std::cout << "  dt = " << dt << ", INTEGRATOR FAILED\n";
            std::cout << "  Backward Euler convergence: FAILED (integration failure)\n\n";
            return false;
        }
        errors.push_back(error);
        std::cout << "  dt = " << dt << ", error = " << error << "\n";
    }
    
    // Compute convergence rate
    Real rate = std::log(errors[0] / errors[1]) / std::log(dts[0] / dts[1]);
    std::cout << "  Convergence rate: " << rate << " (expected ~1.0)\n";
    
    // Accept rates >= 0.8 (including super-convergence cases where rate > 1.0)
    if (rate >= 0.8) {
        std::cout << "  Backward Euler convergence: PASSED\n\n";
        return true;
    } else {
        std::cout << "  Backward Euler convergence: FAILED (rate below acceptable minimum)\n\n";
        return false;
    }
}

bool test_vode_convergence() {
    std::cout << "Testing VODE convergence (expected high order)...\n";
    
    auto integrator = VODE<ExponentialGrowth>{};
    
    std::vector<Real> tolerances = {1.e-4, 1.e-6, 1.e-8, 1.e-10};
    std::vector<Real> errors;
    int failed_integrations = 0;
    
    for (Real tol : tolerances) {
        auto state = VODEState<1>{};
        state.jacobian_analytic = true;
        state.rtol = tol;
        state.atol = 1.e-10;
        // Increase step limit for very tight tolerances
        if (tol <= 1.e-6) {
	  state.max_steps = 2e5;
        }
        
        ExponentialGrowth::state_type problem_state = {1.0};
        
        state.t = 0.0;
        state.tout = 1.0;
        state.y[0] = 1.0;
        
        auto result = integrator.integrate(problem_state, state);
        if (result == IntegratorResult::SUCCESS) {
            Real exact = std::exp(1.0);
            Real error = std::abs(state.y[0] - exact);
            errors.push_back(error);
            std::cout << "  tol = " << tol << ", error = " << error << ", steps = " << state.n_step << "\n";
        } else {
	    std::cout << "  tol = " << tol << ", steps = " << state.n_step << ", FAILED\n";
            failed_integrations++;
        }
    }
    
    if (failed_integrations > 0) {
        std::cout << "  VODE error control: FAILED (" << failed_integrations << " integrations failed)\n\n";
        return false;
    }
    
    if (errors.size() >= 2) {
        bool decreasing = true;
        for (size_t i = 1; i < errors.size(); ++i) {
            if (errors[i] >= errors[i-1]) {
                decreasing = false;
                break;
            }
        }
        
        if (decreasing) {
            std::cout << "  VODE error control: PASSED\n\n";
            return true;
        } else {
            std::cout << "  VODE error control: FAILED (errors not decreasing with tighter tolerance)\n\n";
            return false;
        }
    } else {
        std::cout << "  VODE error control: FAILED (insufficient successful integrations)\n\n";
        return false;
    }
}

int main() {
    std::cout << "Integrator Convergence Test Suite\n";
    std::cout << "=================================\n\n";
    std::cout << "Test problem: y' = y, y(0) = 1\n";
    std::cout << "Exact solution: y(t) = exp(t)\n\n";
    
    int failed_tests = 0;
    
    if (!test_backward_euler_convergence()) {
        failed_tests++;
    }
    
    if (!test_vode_convergence()) {
        failed_tests++;
    }
    
    if (failed_tests > 0) {
        std::cout << "OVERALL RESULT: " << failed_tests << " test(s) failed\n";
        return 1;
    } else {
        std::cout << "OVERALL RESULT: All convergence tests passed\n";
        return 0;
    }
}
