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

int main() {
    constexpr Real h = 0.1;
    constexpr Real z = -1.0;

    if (!check_one_step<RosenbrockSanduA<ScalarDecaySandu>>(h, method_ab_transfer(z),
                                                            "RosenbrockSanduA", 2)) {
        return 1;
    }
    if (!check_one_step<RosenbrockSanduB<ScalarDecaySandu>>(h, method_ab_transfer(z),
                                                            "RosenbrockSanduB", 2)) {
        return 1;
    }
    if (!check_one_step<RosenbrockSanduD<ScalarDecaySandu>>(h, method_d_transfer(z),
                                                            "RosenbrockSanduD", 2)) {
        return 1;
    }

    std::cout << "Rosenbrock Sandu transfer functions: PASSED\n";
    return 0;
}
