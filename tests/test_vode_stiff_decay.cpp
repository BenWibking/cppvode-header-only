// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Strict convergence test for VODE on a stiff linear decay
#include <iostream>
#include <vector>
#include <cmath>
#include <integrators/integrators.hpp>

using namespace integrators;

struct StiffDecay {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;
    static constexpr Real lambda = 1.0e6;
    static void rhs(Real /*t*/, const state_type& y, rhs_type& dydt) {
        dydt[0] = -lambda * y[0];
    }
    static void jacobian(Real /*t*/, const state_type& /*y*/, jacobian_type& J) {
        J[0][0] = -lambda;
    }
};

int main() {
    std::cout << "VODE stiff decay convergence test\n";
    std::cout << "================================\n";

    const Real T = 1.0e-3; // moderately stiff interval
    const Real y0 = 1.0;
    const Real exact = std::exp(-StiffDecay::lambda * T);

    std::vector<Real> tols = {1.e-4, 1.e-6, 1.e-8};
    std::vector<Real> errors;

    for (auto tol : tols) {
        auto integ = VODE<StiffDecay>{};
        auto s = VODEState<1>{};
        s.jacobian_analytic = true;
        s.t = 0.0;
        s.tout = T;
        s.y[0] = y0;
        s.rtol = tol;
        s.atol = 1.e-14;
        auto ps = StiffDecay::state_type{y0};
        auto res = integ.integrate(ps, s);
        if (res != IntegratorResult::SUCCESS) {
            std::cerr << "Integration failed with code=" << static_cast<int>(res) << std::endl;
            return 1;
        }
        Real err = std::abs(s.y[0] - exact);
        errors.push_back(err);
        std::cout << "tol=" << tol << ", error=" << err << ", steps=" << s.n_step << ", rhs=" << s.n_rhs << ", jac=" << s.n_jac << "\n";
        if (err > 1.e3 * tol) {
            std::cerr << "Error " << err << " exceeds 1e3*tol for tol=" << tol << std::endl;
            return 1;
        }
    }

    bool monotone = true;
    for (size_t i = 1; i < errors.size(); ++i) {
        if (!(errors[i] < errors[i-1])) { monotone = false; break; }
    }
    if (!monotone) {
        std::cerr << "Errors not monotonically decreasing with tol" << std::endl;
        return 1;
    }

    std::cout << "Stiff decay test: PASSED\n";
    return 0;
}
