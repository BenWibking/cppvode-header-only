// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Verify VODE conserves linear invariants to machine precision
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <integrators/integrators.hpp>

using namespace integrators;

// Linear 4x4 system with two independent conservation laws:
// - y1 + y2 is conserved (fast exchange)
// - y3 + y4 is conserved (slow exchange)
// This mixes very different time scales to stress the solver.
struct LinearExchange4 {
    static constexpr size_type neqs = 4;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static constexpr Real k12 = 1.0e3;  // 1 <-> 2 fast
    static constexpr Real k21 = 2.0e3;
    static constexpr Real k34 = 1.0e-2; // 3 <-> 4 slow
    static constexpr Real k43 = 3.0e-2;

    static void rhs(Real /*t*/, const state_type& y, rhs_type& f) {
        f[0] = -k12 * y[0] + k21 * y[1];
        f[1] =  k12 * y[0] - k21 * y[1];
        f[2] = -k34 * y[2] + k43 * y[3];
        f[3] =  k34 * y[2] - k43 * y[3];
    }

    static void jacobian(Real /*t*/, const state_type& /*y*/, jacobian_type& J) {
        for (size_type i = 0; i < neqs; ++i) {
            for (size_type j = 0; j < neqs; ++j) J[i][j] = 0.0;
        }
        J[0][0] = -k12; J[0][1] =  k21;
        J[1][0] =  k12; J[1][1] = -k21;
        J[2][2] = -k34; J[2][3] =  k43;
        J[3][2] =  k34; J[3][3] = -k43;
    }
};

int main() {
    // Initial conditions chosen to exercise both fast and slow blocks
    LinearExchange4::state_type y0 = {1.0, 0.0, 2.0, 0.0};
    const Real inv12_0 = y0[0] + y0[1];
    const Real inv34_0 = y0[2] + y0[3];
    const Real invtot_0 = inv12_0 + inv34_0;

    auto integ = VODE<LinearExchange4>{};
    auto s = VODEState<LinearExchange4::neqs>{};
    s.jacobian_analytic = true;  // exact constant Jacobian
    s.t = 0.0;
    s.tout = 1000.0;             // long enough for slow block to equilibrate
    s.y = y0;
    s.rtol = 1.e-9;              // tight tolerances to isolate roundoff
    s.atol = 1.e-12;
    s.max_steps = 200000;

    auto problem_state = y0;
    auto rc = integ.integrate(problem_state, s);
    if (rc != IntegratorResult::SUCCESS) {
        std::cerr << "Integration failed: code=" << static_cast<int>(rc)
                  << ", steps=" << s.n_step << ", rhs=" << s.n_rhs << ", jac=" << s.n_jac << "\n";
        return 1;
    }

    const Real inv12 = s.y[0] + s.y[1];
    const Real inv34 = s.y[2] + s.y[3];
    const Real invtot = inv12 + inv34;

    const Real eps = std::numeric_limits<Real>::epsilon();
    // Allow a modest multiple of machine epsilon to account for accumulation
    const Real tol = 1.0e3 * eps * std::max<Real>(1.0, std::max({std::abs(inv12_0), std::abs(inv34_0), std::abs(invtot_0)}));

    const Real e12 = std::abs(inv12 - inv12_0);
    const Real e34 = std::abs(inv34 - inv34_0);
    const Real etot = std::abs(invtot - invtot_0);

    std::cout.setf(std::ios::scientific, std::ios::floatfield);
    std::cout << std::setprecision(16)
              << "VODE linear invariants test\n"
              << "  steps=" << s.n_step << ", rhs=" << s.n_rhs << ", jac=" << s.n_jac << "\n"
              << "  inv12 error = " << e12 << " (tol ~ " << tol << ")\n"
              << "  inv34 error = " << e34 << " (tol ~ " << tol << ")\n"
              << "  invtot error = " << etot << " (tol ~ " << tol << ")\n";

    if (!(e12 <= 10*tol && e34 <= 10*tol && etot <= 10*tol)) {
        std::cerr << "Linear invariants not conserved to machine precision" << std::endl;
        return 1;
    }

    std::cout << "Linear invariants conservation: PASSED\n";
    return 0;
}

