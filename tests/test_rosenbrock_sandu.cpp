// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Sandu Rosenbrock method transfer-function regression tests
#include <array>
#include <cmath>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

struct ScalarDecaySandu {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    inline static int rhs_calls = 0;
    inline static int jacobian_calls = 0;

    static void rhs(Real /*t*/, const state_type &y, rhs_type &dydt) {
        rhs_calls += 1;
        dydt[0] = -10.0 * y[0];
    }

    static void jacobian(Real /*t*/, const state_type & /*y*/, jacobian_type &jac) {
        jacobian_calls += 1;
        jac[0][0] = -10.0;
    }
};

template <typename Integrator>
bool check_one_step(Real h, Real expected, const char *name, int expected_rhs_calls) {
    ScalarDecaySandu::rhs_calls = 0;
    ScalarDecaySandu::jacobian_calls = 0;
    auto integrator = Integrator{};
    auto state = RODASState<1>{};
    state.jacobian_analytic = true;
    state.autonomous = true;
    state.t = 0.0;
    state.tout = h;
    state.dt = h;
    state.rtol = 1.e6;
    state.atol = 1.e6;
    state.y = {1.0};

    auto problem_state = ScalarDecaySandu::state_type{1.0};
    const auto result = integrator.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        std::cerr << name << " failed with code " << static_cast<int>(result) << "\n";
        return false;
    }
    if (state.n_accept != 1) {
        std::cerr << name << " did not take exactly one accepted step\n";
        return false;
    }
    if (state.n_rhs != expected_rhs_calls ||
        ScalarDecaySandu::rhs_calls != expected_rhs_calls) {
        std::cerr << name << " RHS count mismatch: stats=" << state.n_rhs
                  << " actual=" << ScalarDecaySandu::rhs_calls
                  << " expected=" << expected_rhs_calls << "\n";
        return false;
    }
    if (state.n_jac != 1 || ScalarDecaySandu::jacobian_calls != 1) {
        std::cerr << name << " Jacobian count mismatch: stats=" << state.n_jac
                  << " actual=" << ScalarDecaySandu::jacobian_calls << "\n";
        return false;
    }
    const Real err = std::abs(state.y[0] - expected);
    if (err > 2.e-14) {
        std::cerr << name << " transfer mismatch: got " << state.y[0] << " expected " << expected
                  << " err=" << err << "\n";
        return false;
    }
    return true;
}

Real method_ab_transfer(Real z) {
    constexpr Real sqrt3 = 1.7320508075688772;
    constexpr Real a = (1.0 + sqrt3) / 2.0;
    constexpr Real gamma = (3.0 + sqrt3) / 6.0;
    return (1.0 - a * z) / std::pow(1.0 - gamma * z, 3);
}

Real method_d_transfer(Real z) { return (1.0 - z) / std::pow(1.0 - 0.5 * z, 4); }

Real paper_ros2_transfer(Real z) {
    constexpr Real gamma = 1.7071067811865475;
    constexpr Real a21 = 0.5857864376269049;
    constexpr Real c21 = -1.1715728752538097;
    constexpr Real m1 = 0.8786796564403572;
    constexpr Real m2 = 0.2928932188134524;

    const Real solve_scale = gamma / (1.0 - gamma * z);
    const Real k1 = solve_scale * z;
    const Real k2 = solve_scale * (z * (1.0 + a21 * k1) + c21 * k1);
    return 1.0 + m1 * k1 + m2 * k2;
}

Real method_c_transfer(Real z) {
    constexpr Real gamma = 0.78867513459481275;
    constexpr Real a31 = -0.4335531486700170;
    constexpr Real a32 = 1.7222282832648295;
    constexpr Real c21 = -0.3264246859454366;
    constexpr Real c31 = 0.4138899169340995;
    constexpr Real c32 = 1.6076951545867360;
    constexpr Real m1 = -7.1285171234651825;
    constexpr Real m2 = 8.4030361763035106;
    constexpr Real m3 = 0.9509618943233421;

    const Real solve_scale = gamma / (1.0 - gamma * z);
    const Real k1 = solve_scale * z;
    const Real k2 = solve_scale * (z + c21 * k1);
    const Real y3 = 1.0 + a31 * k1 + a32 * k2;
    const Real k3 = solve_scale * (z * y3 + c31 * k1 + c32 * k2);
    return 1.0 + m1 * k1 + m2 * k2 + m3 * k3;
}

int main() {
    constexpr Real h = 0.1;
    constexpr Real z = -1.0;

    if (!check_one_step<Ros2<ScalarDecaySandu>>(h, paper_ros2_transfer(z),
                                                "PaperRos2", 2)) {
        return 1;
    }
    if (!check_one_step<RosenbrockSanduA<ScalarDecaySandu>>(h, method_ab_transfer(z),
                                                            "RosenbrockSanduA", 2)) {
        return 1;
    }
    if (!check_one_step<RosenbrockSanduB<ScalarDecaySandu>>(h, method_ab_transfer(z),
                                                            "RosenbrockSanduB", 2)) {
        return 1;
    }
    if (!check_one_step<RosenbrockSanduC<ScalarDecaySandu>>(h, method_c_transfer(z),
                                                            "RosenbrockSanduC", 2)) {
        return 1;
    }
    if (!check_one_step<RosenbrockSanduD<ScalarDecaySandu>>(h, method_d_transfer(z),
                                                            "RosenbrockSanduD", 2)) {
        return 1;
    }

    std::cout << "Rosenbrock Sandu transfer functions: PASSED\n";
    return 0;
}
