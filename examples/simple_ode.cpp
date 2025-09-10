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
    static void rhs(integrators::Real t, const state_type& state, rhs_type& dydt) {
        dydt[0] = -state[0];
    }
    
    // Jacobian: J = -1
    static void jacobian(integrators::Real t, const state_type& state, jacobian_type& jac) {
        jac[0][0] = -1.0;
    }
};

void test_integrator(const std::string& name, auto& integrator, auto& state) {
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
    
    std::cout << name << ":\n";
    std::cout << "  Result: " << (result == IntegratorResult::SUCCESS ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Steps: " << state.n_step << ", RHS evals: " << state.n_rhs;
    if (state.n_jac > 0) std::cout << ", Jacobian evals: " << state.n_jac;
    std::cout << "\n";
    std::cout << "  Final value: " << state.y[0] << "\n";
    std::cout << "  Exact value: " << exact << "\n";
    std::cout << "  Error: " << error << "\n";
    std::cout << "  Relative error: " << error/exact << "\n\n";
}

int main() {
    using namespace integrators;
    
    std::cout << "Header-Only Integrator Library Test\n";
    std::cout << "====================================\n\n";
    std::cout << "Problem: dy/dt = -y, y(0) = 1\n";
    std::cout << "Exact solution: y(t) = exp(-t)\n";
    std::cout << "Integration interval: [0, 1]\n\n";
    
    // Test Backward Euler
    {
        auto integrator = BackwardEuler<SimpleDecay>{};
        auto state = BackwardEulerState<1>{};
        state.jacobian_analytic = true;
        test_integrator("Backward Euler", integrator, state);
    }
    
    // Test VODE
    {
        auto integrator = VODE<SimpleDecay>{};
        auto state = VODEState<1>{};
        state.jacobian_analytic = true;
        test_integrator("VODE", integrator, state);
    }
    
    return 0;
}
