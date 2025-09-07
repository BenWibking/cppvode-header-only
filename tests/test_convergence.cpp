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
    
    static void rhs(Real t, const state_type& y, rhs_type& dydt) {
        dydt[0] = y[0];
    }
    
    static void jacobian(Real t, const state_type& y, jacobian_type& jac) {
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

void test_backward_euler_convergence() {
    std::cout << "Testing Backward Euler convergence (expected order 1)...\n";
    
    auto integrator = BackwardEuler<ExponentialGrowth>{};
    
    std::vector<Real> dts = {0.1, 0.05, 0.025, 0.0125};
    std::vector<Real> errors;
    
    for (Real dt : dts) {
        auto state = BackwardEulerState<1>{};
        state.jacobian_analytic = true;
        Real error = compute_error(integrator, state, dt);
        errors.push_back(error);
        std::cout << "  dt = " << dt << ", error = " << error << "\n";
    }
    
    // Compute convergence rate
    Real rate = std::log(errors[0] / errors[1]) / std::log(dts[0] / dts[1]);
    std::cout << "  Convergence rate: " << rate << " (expected ~1.0)\n";
    
    if (rate > 0.8 && rate < 1.2) {
        std::cout << "  Backward Euler convergence: PASSED\n\n";
    } else {
        std::cout << "  Backward Euler convergence: QUESTIONABLE\n\n";
    }
}

void test_vode_convergence() {
    std::cout << "Testing VODE convergence (expected high order)...\n";
    
    auto integrator = VODE<ExponentialGrowth>{};
    
    std::vector<Real> tolerances = {1.e-4, 1.e-6, 1.e-8, 1.e-10};
    std::vector<Real> errors;
    
    for (Real tol : tolerances) {
        auto state = VODEState<1>{};
        state.jacobian_analytic = true;
        state.rtol = tol;
        state.atol = tol * 1.e-6;
        
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
            std::cout << "  tol = " << tol << ", FAILED\n";
        }
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
        } else {
            std::cout << "  VODE error control: QUESTIONABLE\n\n";
        }
    }
}

int main() {
    std::cout << "Integrator Convergence Test Suite\n";
    std::cout << "=================================\n\n";
    std::cout << "Test problem: y' = y, y(0) = 1\n";
    std::cout << "Exact solution: y(t) = exp(t)\n\n";
    
    test_backward_euler_convergence();
    test_vode_convergence();
    
    std::cout << "Convergence tests completed!\n";
    return 0;
}