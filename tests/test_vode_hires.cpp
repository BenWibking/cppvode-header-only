// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: VODE benchmark on the classic HIRES stiff system (Hairer/Wanner)
#include <iostream>
#include <vector>
#include <cmath>
#include <integrators/integrators.hpp>

using namespace integrators;

// HIRES (High irradiance) benchmark (8D)
// Reference form from Hairer/Wanner test set
struct HIRES {
    static constexpr size_type neqs = 8;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    static void rhs(Real /*t*/, const state_type& y, rhs_type& f) {
        // Parameters per canonical HIRES specification
        constexpr Real c1 = 1.71;
        constexpr Real c2 = 0.43;
        constexpr Real c3 = 8.32;
        constexpr Real c4 = 8.75;
        constexpr Real c5 = 10.03;
        constexpr Real c6 = 0.035;
        constexpr Real c7 = 1.12;
        constexpr Real c8 = 1.745;
        constexpr Real c9 = 280.0;
        constexpr Real c10 = 0.69;
        constexpr Real c11 = 1.81;

        f[0] = -c1 * y[0] + c2 * y[1] + c3 * y[2] + 7.0e-4;
        f[1] =  c1 * y[0] - c4 * y[1];
        f[2] = -c5 * y[2] + c2 * y[3] + c6 * y[4];
        f[3] =  c3 * y[1] + c1 * y[2] - c7 * y[3];
        f[4] = -c8 * y[4] + c2 * y[5] + c2 * y[6];
        f[5] = -c9 * y[5] * y[7] + c10 * y[3] + c1 * y[4] - c2 * y[5] + c10 * y[6];
        f[6] =  c9 * y[5] * y[7] - c11 * y[6];
        f[7] = -c9 * y[5] * y[7] + c11 * y[6];
    }

    static void jacobian(Real /*t*/, const state_type& y, jacobian_type& J) {
        // Zero initialize
        for (size_type i = 0; i < neqs; ++i) {
            for (size_type j = 0; j < neqs; ++j) J[i][j] = 0.0;
        }
        constexpr Real c1 = 1.71;
        constexpr Real c2 = 0.43;
        constexpr Real c3 = 8.32;
        constexpr Real c4 = 8.75;
        constexpr Real c5 = 10.03;
        constexpr Real c6 = 0.035;
        constexpr Real c7 = 1.12;
        constexpr Real c8 = 1.745;
        constexpr Real c9 = 280.0;
        constexpr Real c10 = 0.69;
        constexpr Real c11 = 1.81;

        // Row 0
        J[0][0] = -c1;  J[0][1] =  c2;  J[0][2] =  c3;
        // Row 1
        J[1][0] =  c1;  J[1][1] = -c4;
        // Row 2
        J[2][2] = -c5;  J[2][3] =  c2;  J[2][4] =  c6;
        // Row 3
        J[3][1] =  c3;  J[3][2] =  c1;  J[3][3] = -c7;
        // Row 4
        J[4][4] = -c8;  J[4][5] =  c2;  J[4][6] =  c2;
        // Row 5
        J[5][3] =  c10; J[5][4] =  c1;  J[5][5] = -(c9 * y[7] + c2); J[5][6] =  c10; J[5][7] = -c9 * y[5];
        // Row 6
        J[6][5] =  c9 * y[7]; J[6][6] = -c11; J[6][7] =  c9 * y[5];
        // Row 7
        J[7][5] = -c9 * y[7]; J[7][6] =  c11; J[7][7] = -c9 * y[5];
    }
};

int main() {
    std::cout << "HIRES stiff benchmark with VODE (BDF)\n";
    std::cout << "====================================\n";

    // Standard initial condition used in literature
    HIRES::state_type y0 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.7e-3};
    // Canonical final time from Hairer/Wanner
    const Real t_end = 321.8122;

    // Solve at multiple tolerances and check monotone self-consistency
    // High-accuracy baseline (serves as reference final state)
    auto integ_ref = VODE<HIRES>{};
    auto s_ref = VODEState<HIRES::neqs>{};
    s_ref.jacobian_analytic = true;
    s_ref.t = 0.0;
    s_ref.tout = t_end;
    s_ref.y = y0;
    s_ref.rtol = 1.e-9;
    s_ref.atol = 1.e-12;
    s_ref.max_steps = 1000000;
    auto ps_ref = y0;
    auto rc_ref = integ_ref.integrate(ps_ref, s_ref);
    if (rc_ref != IntegratorResult::SUCCESS) {
        std::cerr << "Reference integration failed: code=" << static_cast<int>(rc_ref) << " steps=" << s_ref.n_step << "\n";
        return 1;
    }
    // Sanity: positivity for reference
    {
        Real miny = s_ref.y[0];
        for (size_type i = 0; i < HIRES::neqs; ++i) miny = std::min(miny, s_ref.y[i]);
        if (miny < -1.e-12) {
            std::cerr << "Reference negativity (min=" << miny << ")\n";
            return 1;
        }
    }
    std::cout << "ref: tol=1e-9, steps=" << s_ref.n_step << ", rhs=" << s_ref.n_rhs << ", jac=" << s_ref.n_jac << "\n";

    auto maxdiff = [](const HIRES::state_type& a, const HIRES::state_type& b) {
        Real d = 0.0; for (size_type i = 0; i < HIRES::neqs; ++i) d = std::max(d, std::abs(a[i]-b[i])); return d;
    };

    // Compare solutions at two tolerances to the reference
    std::vector<Real> test_tols = {1.e-6, 1.e-8};
    std::vector<Real> diffs;
    for (auto tol : test_tols) {
        auto integ = VODE<HIRES>{};
        auto s = VODEState<HIRES::neqs>{};
        s.jacobian_analytic = true;
        s.t = 0.0;
        s.tout = t_end;
        s.y = y0;
        s.rtol = tol;
        s.atol = 1.e-12;
        s.max_steps = 500000;
        auto ps = y0;
        auto rc = integ.integrate(ps, s);
        if (rc != IntegratorResult::SUCCESS) {
            std::cerr << "Integration failed: code=" << static_cast<int>(rc) << " steps=" << s.n_step << "\n";
            return 1;
        }
        // Positivity
        Real miny = s.y[0];
        for (size_type i = 0; i < HIRES::neqs; ++i) miny = std::min(miny, s.y[i]);
        if (miny < -1.e-10) {
            std::cerr << "Negativity detected (min=" << miny << ")\n";
            return 1;
        }
        Real d = maxdiff(s.y, s_ref.y);
        diffs.push_back(d);
        std::cout << "tol=" << tol << ", steps=" << s.n_step << ", rhs=" << s.n_rhs << ", jac=" << s.n_jac << ", |y-yr|_inf=" << d << "\n";
    }

    if (!(diffs[1] < diffs[0])) {
        std::cerr << "Errors vs reference not decreasing with tighter tol" << std::endl;
        return 1;
    }
    // Loose absolute thresholds tied to tolerance scale
    if (diffs[0] > 5e-5 || diffs[1] > 5e-7) {
        std::cerr << "Final-state error too large: d6=" << diffs[0] << ", d8=" << diffs[1] << std::endl;
        return 1;
    }

    std::cout << "HIRES benchmark to t=321.8122: PASSED\n";
    return 0;
}
