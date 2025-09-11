// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Robertson chemical kinetics problem example
// ABOUTME: Classic stiff ODE test case: A -> B -> C with very different time scales
#include <iostream>
#include <cmath>
#include <integrators/integrators.hpp>

// Robertson problem: 3 species chemical kinetics
// dy1/dt = -0.04*y1 + 1e4*y2*y3
// dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2  
// dy3/dt = 3e7*y2^2
struct Robertson {
    static constexpr integrators::size_type neqs = 3;
    
    using state_type = std::array<integrators::Real, neqs>;
    using rhs_type = std::array<integrators::Real, neqs>;
    using jacobian_type = std::array<std::array<integrators::Real, neqs>, neqs>;
    
    static void rhs([[maybe_unused]] integrators::Real t, const state_type& y, rhs_type& dydt) {
        constexpr integrators::Real k1 = 0.04;
        constexpr integrators::Real k2 = 1.0e4;  
        constexpr integrators::Real k3 = 3.0e7;
        
        dydt[0] = -k1 * y[0] + k2 * y[1] * y[2];
        dydt[1] = k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] * y[1];
        dydt[2] = k3 * y[1] * y[1];
    }
    
    static void jacobian([[maybe_unused]] integrators::Real t, const state_type& y, jacobian_type& jac) {
        constexpr integrators::Real k1 = 0.04;
        constexpr integrators::Real k2 = 1.0e4;
        constexpr integrators::Real k3 = 3.0e7;
        
        jac[0][0] = -k1;
        jac[0][1] = k2 * y[2];
        jac[0][2] = k2 * y[1];
        
        jac[1][0] = k1;
        jac[1][1] = -k2 * y[2] - 2.0 * k3 * y[1];
        jac[1][2] = -k2 * y[1];
        
        jac[2][0] = 0.0;
        jac[2][1] = 2.0 * k3 * y[1];
        jac[2][2] = 0.0;
    }
};

void test_robertson(const std::string& name, auto& integrator, auto& state) {
    using namespace integrators;
    
    // Initial conditions: y1=1, y2=0, y3=0
    Robertson::state_type problem_state = {1.0, 0.0, 0.0};
    
    state.t = 0.0;
    state.tout = 40.0;  // Integrate to t=40 (quite stiff)
    state.y[0] = 1.0;
    state.y[1] = 0.0; 
    state.y[2] = 0.0;
    state.rtol = 1.e-6;
    state.atol = 1.e-10;
    
    auto result = integrator.integrate(problem_state, state);
    
    // Check conservation: y1 + y2 + y3 should equal 1
    Real total = state.y[0] + state.y[1] + state.y[2];
    Real conservation_error = std::abs(total - 1.0);
    
    std::cout << name << ":\n";
    std::cout << "  Result: " << (result == IntegratorResult::SUCCESS ? "SUCCESS" : "FAILED") << "\n";
    std::cout << "  Steps: " << state.n_step << ", RHS evals: " << state.n_rhs;
    if (state.n_jac > 0) std::cout << ", Jacobian evals: " << state.n_jac;
    std::cout << "\n";
    std::cout << "  Final state: [" << state.y[0] << ", " << state.y[1] << ", " << state.y[2] << "]\n";
    std::cout << "  Conservation error: " << conservation_error << "\n\n";
}

int main() {
    using namespace integrators;
    
    std::cout << "Robertson Chemical Kinetics Problem\n";
    std::cout << "===================================\n\n";
    std::cout << "Stiff ODE system with time scales from 1e-5 to 1e5\n";
    std::cout << "Integration interval: [0, 40]\n";
    std::cout << "Initial conditions: y1=1, y2=0, y3=0\n\n";
    
    // Test VODE - should handle stiffness well
    {
        auto integrator = VODE<Robertson>{};
        auto state = VODEState<3>{};
        state.jacobian_analytic = true;
        // Integrate to 40 to collect many steps with VODE
        state.tout = 40.0;
        test_robertson("VODE", integrator, state);
    }
    
    return 0;
}
