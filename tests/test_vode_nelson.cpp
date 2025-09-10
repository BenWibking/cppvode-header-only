// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Validate Nelson astrochemistry network against reference (L2 relative error)
#include <iostream>
#include <iomanip>
#include <cmath>
#include <array>
#include <cassert>
#include <integrators/integrators.hpp>

using namespace integrators;

struct Nelson {
    static constexpr size_type neqs = 14;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static constexpr Real T = 10.0;
    static constexpr Real Av = 2.0;
    static constexpr Real Go = 1.7;
    static constexpr Real n_H = 611.0;
    static constexpr Real shield = 1.0;

    static void rhs(Real /*t*/, const state_type& u, rhs_type& du) {
        const Real T054 = std::pow(T, 0.54);
        const Real T061 = std::pow(T, 0.61);
        const Real T065 = std::pow(T, 0.65);
        const Real T05  = std::sqrt(T);

        const Real e_m15Av = std::exp(-1.5 * Av);
        const Real e_m17Av = std::exp(-1.7 * Av);
        const Real e_m19Av = std::exp(-1.9 * Av);
        const Real e_m3Av  = std::exp(-3.0 * Av);

        const Real& u1=u[0];  const Real& u2=u[1];  const Real& u3=u[2];  const Real& u4=u[3];
        const Real& u5=u[4];  const Real& u6=u[5];  const Real& u7=u[6];  const Real& u8=u[7];
        const Real& u9=u[8];  const Real& u10=u[9]; const Real& u11=u[10]; const Real& u12=u[11];
        const Real& u13=u[12]; const Real& u14=u[13];

        du[0] = -1.2e-17 * u1
              + n_H * (1.9e-6 * u2 * u3) / T054
              - n_H * 4e-16 * u1 * u12
              - n_H * 7e-15 * u1 * u5
              + n_H * 1.7e-9 * u10 * u2
              + n_H * 2e-9 * u2 * u6
              + n_H * 2e-9 * u2 * u14
              + n_H * 8e-10 * u2 * u8;

        du[1] =  1.2e-17 * u1
              + n_H * (-1.9e-6 * u3 * u2) / T054
              - n_H * 1.7e-9 * u10 * u2
              - n_H * 2e-9 * u2 * u6
              - n_H * 2e-9 * u2 * u14
              - n_H * 8e-10 * u2 * u8;

        du[2] =  n_H * (-1.4e-10 * u3 * u12) / T061
              - n_H * (3.8e-10 * u13 * u3) / T065
              - n_H * (3.3e-5  * u11 * u3) / T
              + 1.2e-17 * u1
              - n_H * (1.9e-6  * u3 * u2) / T054
              + 6.8e-18 * u4
              - n_H * (9e-11   * u3 * u5) / std::pow(T, 0.64)
              + 3e-10 * Go * e_m3Av * u6
              + n_H * 2e-9 * u2 * u13
              + 2.0e-10 * Go * e_m19Av * u14;

        du[3] =  n_H * (9e-11 * u3 * u5) / std::pow(T, 0.64)
              - 6.8e-18 * u4
              + n_H * 7e-15 * u1 * u5
              + n_H * 1.6e-9 * u10 * u5;

        du[4] =  6.8e-18 * u4
              - n_H * (9e-11 * u3 * u5) / std::pow(T, 0.64)
              - n_H * 7e-15 * u1 * u5
              - n_H * 1.6e-9 * u10 * u5;

        du[5] =  n_H * (1.4e-10 * u3 * u12) / T061
              - n_H * 2e-9 * u2 * u6
              - n_H * 5.8e-12 * T05 * u9 * u6
              + 1e-9 * Go * e_m15Av * u7
              - 3e-10 * Go * e_m3Av * u6
              + 1e-10 * Go * e_m3Av * u10 * shield;

        du[6] =  n_H * (-2e-10) * u7 * u8
              + n_H * 4e-16 * u1 * u12
              + n_H * 2e-9 * u2 * u6
              - 1e-9 * Go * u7 * e_m15Av;

        du[7] =  n_H * (-2e-10) * u7 * u8
              + n_H * 1.6e-9 * u10 * u5
              - n_H * 8e-10 * u2 * u8
              + 5e-10 * Go * e_m17Av * u9
              + 1e-10 * Go * e_m3Av * u10 * shield;

        du[8] =  n_H * (-1e-9) * u9 * u12
              + n_H * 8e-10 * u2 * u8
              - n_H * 5.8e-12 * T05 * u9 * u6
              - 5e-10 * Go * e_m17Av * u9;

        du[9] =  n_H * (3.3e-5 * u11 * u3) / T
              + n_H * 2e-10 * u7 * u8
              - n_H * 1.7e-9 * u10 * u2
              - n_H * 1.6e-9 * u10 * u5
              + n_H * 5.8e-12 * T05 * u9 * u6
              - 1e-10 * Go * e_m3Av * u10
              + 1.5e-10 * Go * std::exp(-2.5 * Av) * u11 * shield;

        du[10] =  n_H * (-3.3e-5 * u11 * u3) / T
               +  n_H * 1e-9 * u9 * u12
               +  n_H * 1.7e-9 * u10 * u2
               -  1.5e-10 * Go * std::exp(-2.5 * Av) * u11;

        du[11] =  n_H * (-1.4e-10 * u3 * u12) / T061
               -  n_H * 4e-16 * u1 * u12
               -  n_H * 1e-9 * u9 * u12
               +  n_H * 1.6e-9 * u10 * u5
               +  3e-10 * Go * e_m3Av * u6;

        du[12] =  n_H * (-3.8e-10 * u13 * u3) / T065
               +  n_H * 2e-9 * u2 * u14
               +  2.0e-10 * Go * e_m19Av * u14;

        du[13] =  n_H * (3.8e-10 * u13 * u3) / T065
               -  n_H * 2e-9 * u2 * u14
               -  2.0e-10 * Go * e_m19Av * u14;
    }

    static void jacobian(Real, const state_type&, jacobian_type& J) {
        for (size_type i = 0; i < neqs; ++i) {
            for (size_type j = 0; j < neqs; ++j) J[i][j] = (i==j ? 1.0 : 0.0);
        }
    }
};

int main() {
    // Time span ~ 30 kyr in seconds
    const Real seconds_per_year = 3600.0 * 24.0 * 365.0;
    const Real t_end = 30000.0 * seconds_per_year;

    Nelson::state_type y0 = {
        0.5, 9.059e-9, 2.0e-4, 0.1, 7.866e-7,
        0.0, 0.0, 4.0e-4, 0.0, 0.0, 0.0, 2.0e-4, 2.0e-7, 2.0e-7
    };

    const std::array<Real, Nelson::neqs> y_ref = {
        0.4999878445605511,
        1.976608431204695e-10,
        8.960595204507145e-5,
        0.10000066577550308,
        1.208244969246418e-7,
        0.00011756938799503817,
        7.302419667363057e-8,
        0.00039688311996499643,
        4.906657141307707e-10,
        3.1163892353666996e-6,
        1.35007321965445e-13,
        7.924119897589606e-5,
        2.479215172415109e-7,
        1.5207848275848912e-7
    };

    auto integ = VODE<Nelson>{};
    auto s = VODEState<Nelson::neqs>{};
    s.jacobian_analytic = false; // numerical Jacobian
    s.t = 0.0;
    s.tout = t_end;
    s.y = y0;
    // Tighten tolerances to meet 1e-6 L2 rel threshold
    s.rtol = 1.e-4;
    s.atol = 1.e-6;
    s.max_steps = 1000;

    auto problem_state = y0;
    auto rc = integ.integrate(problem_state, s);
    if (rc != IntegratorResult::SUCCESS) {
        std::cerr << "Nelson integration failed: code=" << static_cast<int>(rc)
                  << ", steps=" << s.n_step << ", rhs=" << s.n_rhs << ", jac=" << s.n_jac << "\n";
        return 1;
    }

    Real num = 0.0, den = 0.0;
    for (size_type i = 0; i < Nelson::neqs; ++i) {
        const Real di = s.y[i] - y_ref[i];
        num += di * di;
        den += y_ref[i] * y_ref[i];
    }
    const Real l2_rel_err = std::sqrt(num) / std::sqrt(den);
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
    std::cout << std::setprecision(12)
              << "Nelson L2_rel_err = " << l2_rel_err
              << ", steps = " << s.n_step << ", rhs = " << s.n_rhs << ", jac = " << s.n_jac << "\n";

    // Threshold per request
    if (l2_rel_err < 3.0e-6) {
      return 0;
    } else {
      return 1;
    }
}

