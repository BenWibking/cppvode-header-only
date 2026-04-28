// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Simple ODE example demonstrating header-only integrator usage
// ABOUTME: Solves dy/dt = -y with y(0) = 1, exact solution y(t) = exp(-t)
#include <iostream>
#include <cmath>
#include <integrators/integrators.hpp>

// Define a simple ODE problem
struct SimpleDecay {
    static constexpr integrators::size_type neqs = 1;
    
    using state_type = std::array<integrators::Real, neqs>;
    using rhs_type = std::array<integrators::Real, neqs>;
    using jacobian_type = std::array<std::array<integrators::Real, neqs>, neqs>;
    
    // RHS: dy/dt = -y
    static void rhs([[maybe_unused]] integrators::Real t, const state_type& state, rhs_type& dydt) {
        dydt[0] = -state[0];
    }
    
    // Jacobian: J = -1
    static void jacobian([[maybe_unused]] integrators::Real t, [[maybe_unused]] const state_type& state, jacobian_type& jac) {
        jac[0][0] = -1.0;
    }
};

bool test_integrator(const std::string& name, auto& integrator, auto& state) {
    using namespace integrators;
    
    // Set up problem: y(0) = 1, integrate to t = 1
    SimpleDecay::state_type problem_state = {1.0};
    
    state.t = 0.0;
    state.tout = 1.0;
    state.y[0] = 1.0;
    state.rtol = 1.e-8;
    state.atol = 1.e-12;
    
    auto result = integrator.integrate(problem_state, state);
    
    Real exact = std::exp(-1.0); // Exact solution at t=1
    Real error = std::abs(state.y[0] - exact);
    Real relative_error = error / exact;
    
    // Consider test passed if integrator succeeded and relative error < 1%
    bool test_passed = (result == IntegratorResult::SUCCESS) && (relative_error < 0.01);
    
    std::cout << name << ":\n";
    std::cout << "  Result: " << (result == IntegratorResult::SUCCESS ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Steps: " << state.n_step << ", RHS evals: " << state.n_rhs;
    if (state.n_jac > 0) std::cout << ", Jacobian evals: " << state.n_jac;
    std::cout << "\n";
    std::cout << "  Final value: " << state.y[0] << "\n";
    std::cout << "  Exact value: " << exact << "\n";
    std::cout << "  Error: " << error << "\n";
    std::cout << "  Relative error: " << relative_error << "\n";
    std::cout << "  Test: " << (test_passed ? "PASSED" : "FAILED") << "\n\n";
    
    return test_passed;
}

int main() {
    using namespace integrators;
    
    std::cout << "Header-Only Integrator Library Test\n";
    std::cout << "====================================\n\n";
    std::cout << "Problem: dy/dt = -y, y(0) = 1\n";
    std::cout << "Exact solution: y(t) = exp(-t)\n";
    std::cout << "Integration interval: [0, 1]\n\n";
    
    int failed_tests = 0;
    
    // Test Backward Euler
    {
        auto integrator = BackwardEuler<SimpleDecay>{};
        auto state = BackwardEulerState<1>{};
        state.jacobian_analytic = true;
        state.dt = 0.1;  // Set initial timestep to enable adaptive stepping
        if (!test_integrator("Backward Euler", integrator, state)) {
            failed_tests++;
        }
    }
    
    // Test YASS
    {
        auto integrator = YASS<SimpleDecay>{};
        auto state = YASSState<1>{};
        state.jacobian_analytic = true;
        state.dt = 0.1;  // Set initial timestep to enable adaptive stepping
        if (!test_integrator("YASS", integrator, state)) {
            failed_tests++;
        }
    }
    
    // Test VODE
    {
        auto integrator = VODE<SimpleDecay>{};
        auto state = VODEState<1>{};
        state.jacobian_analytic = true;
        state.max_steps = 50000;  // Allow more steps for tight tolerances
        if (!test_integrator("VODE", integrator, state)) {
            failed_tests++;
        }
    }
    
    if (failed_tests > 0) {
        std::cout << "OVERALL RESULT: " << failed_tests << " test(s) failed\n";
        return 1;
    } else {
        std::cout << "OVERALL RESULT: All tests passed\n";
        return 0;
    }
}
