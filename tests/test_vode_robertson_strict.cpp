// ABOUTME: Strict convergence test for VODE on the stiff Robertson problem
#include <iostream>
#include <vector>
#include <cmath>
#include <integrators/integrators.hpp>

using namespace integrators;

// Robertson problem definition (same as example)
struct Robertson {
    static constexpr size_type neqs = 3;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;
    static void rhs(Real /*t*/, const state_type& y, rhs_type& dydt) {
        constexpr Real k1 = 0.04;
        constexpr Real k2 = 1.0e4;
        constexpr Real k3 = 3.0e7;
        dydt[0] = -k1 * y[0] + k2 * y[1] * y[2];
        dydt[1] = k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] * y[1];
        dydt[2] = k3 * y[1] * y[1];
    }
    static void jacobian(Real /*t*/, const state_type& y, jacobian_type& jac) {
        constexpr Real k1 = 0.04;
        constexpr Real k2 = 1.0e4;
        constexpr Real k3 = 3.0e7;
        jac[0][0] = -k1;          jac[0][1] = k2 * y[2];               jac[0][2] = k2 * y[1];
        jac[1][0] = k1;           jac[1][1] = -k2 * y[2] - 2.0 * k3 * y[1]; jac[1][2] = -k2 * y[1];
        jac[2][0] = 0.0;          jac[2][1] = 2.0 * k3 * y[1];         jac[2][2] = 0.0;
    }
};

int main() {
    std::cout << "Strict VODE convergence test on Robertson problem\n";
    std::cout << "=================================================\n";

    // Compute solutions at decreasing tolerances and check self-consistency
    std::vector<Real> tols = {1.e-6, 1.e-8, 1.e-10};
    std::vector<Robertson::state_type> ys;
    std::vector<Real> diffs;
    for (auto tol : tols) {
        auto integrator = VODE<Robertson>{};
        auto state = VODEState<3>{};
        state.jacobian_analytic = true;
        state.t = 0.0;
        state.tout = 1.0;
        state.y = {1.0, 0.0, 0.0};
        state.rtol = tol;
        state.atol = 1.e-10; // keep reasonable absolute tolerance
        auto problem_state2 = Robertson::state_type{1.0, 0.0, 0.0};
        auto res = integrator.integrate(problem_state2, state);
        if (res != IntegratorResult::SUCCESS) {
            std::cout << "tol=" << tol << ": FAILED (code=" << static_cast<int>(res) << ")\n";
            return 1;
        }
        ys.push_back(state.y);
        Real total = state.y[0] + state.y[1] + state.y[2];
        Real cons = std::abs(total - 1.0);
        std::cout << "tol=" << tol << ", steps=" << state.n_step << ", cons_err=" << cons << "\n";
        if (cons > 1.e-6) {
            std::cerr << "Conservation error too large" << std::endl;
            return 1;
        }
    }

    // Pairwise differences should decrease with tighter tolerances
    for (size_t k = 1; k < ys.size(); ++k) {
        Real d = 0.0;
        for (size_t i = 0; i < 3; ++i) d = std::max(d, std::abs(ys[k][i] - ys[k-1][i]));
        diffs.push_back(d);
    }
    bool monotone = true;
    for (size_t i = 1; i < diffs.size(); ++i) {
        if (!(diffs[i] < diffs[i-1])) { monotone = false; break; }
    }
    if (!monotone) {
        std::cerr << "Pairwise solution differences not decreasing with tol" << std::endl;
        return 1;
    }
    if (diffs.back() > 1.e-6) {
        std::cerr << "Final pairwise difference too large: " << diffs.back() << std::endl;
        return 1;
    }

    std::cout << "Strict Robertson self-consistency: PASSED\n";
    return 0;
}
