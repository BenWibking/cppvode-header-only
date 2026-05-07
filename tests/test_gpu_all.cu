// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: CUDA device coverage for all integrator and linear algebra tests
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <integrators/integrators.hpp>
#include <integrators/linear_algebra.hpp>

using namespace integrators;

#define TEST_DEVICE __device__
#define GPU_TEST __device__ __noinline__

struct GpuTestResult {
    int passed;
    int line;
    int code;
    Real value;
    Real expected;
};

TEST_DEVICE GpuTestResult gpu_pass() {
    return {1, 0, 0, 0.0, 0.0};
}

TEST_DEVICE GpuTestResult gpu_fail(int line, int code = 0,
                                  Real value = 0.0, Real expected = 0.0) {
    return {0, line, code, value, expected};
}

#define GPU_REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            return gpu_fail(__LINE__); \
        } \
    } while (false)

#define GPU_REQUIRE_CLOSE(value, expected, tolerance) \
    do { \
        const Real gpu_require_value = (value); \
        const Real gpu_require_expected = (expected); \
        if (std::abs(gpu_require_value - gpu_require_expected) >= (tolerance)) { \
            return gpu_fail(__LINE__, 0, gpu_require_value, gpu_require_expected); \
        } \
    } while (false)

TEST_DEVICE Real max_abs_diff(const std::array<Real, 8>& a,
                              const std::array<Real, 8>& b) {
    Real d = 0.0;
    for (size_type i = 0; i < 8; ++i) {
        d = std::max(d, std::abs(a[i] - b[i]));
    }
    return d;
}

namespace linalg_cases {

GPU_TEST GpuTestResult lu_decomposition() {
    std::array<std::array<Real, 3>, 3> A = {{
        {{2.0, 1.0, 1.0}},
        {{1.0, 3.0, 2.0}},
        {{1.0, 0.0, 0.0}}
    }};
    std::array<int, 3> pivot{};
    const int info = linalg::lu_decomposition<3, true>(A, pivot);
    GPU_REQUIRE(info == 0);
    return gpu_pass();
}

GPU_TEST GpuTestResult matrix_solve() {
    std::array<std::array<Real, 2>, 2> A = {{
        {{2.0, 1.0}},
        {{1.0, 2.0}}
    }};
    std::array<Real, 2> b = {3.0, 3.0};
    std::array<int, 2> pivot{};
    const int info = linalg::lu_decomposition<2, true>(A, pivot);
    GPU_REQUIRE(info == 0);
    linalg::lu_solve<2, true>(A, pivot, b);
    GPU_REQUIRE_CLOSE(b[0], 1.0, 1.e-12);
    GPU_REQUIRE_CLOSE(b[1], 1.0, 1.e-12);
    return gpu_pass();
}

GPU_TEST GpuTestResult matrix_solve_regression() {
    std::array<std::array<Real, 3>, 3> A = {{
        {{0.96181280249950907, 0.95268950339467517, -0.74900493855137884}},
        {{0.53100510298151371, 1.0717186887085923, -0.85952813616124646}},
        {{-0.57964266413365473, 0.32320129774529782, 2.4022762414188135}}
    }};
    std::array<Real, 3> b = {
        -0.040791658492035136,
        -0.59652843561031799,
        1.2119314898967246
    };
    const std::array<Real, 3> x_true = {
        0.97497543198967151,
        -0.40292336836364195,
        0.79395290575768396
    };
    std::array<int, 3> pivot{};
    const int info = linalg::lu_decomposition<3, true>(A, pivot);
    GPU_REQUIRE(info == 0);
    linalg::lu_solve<3, true>(A, pivot, b);

    Real max_error = 0.0;
    for (size_type i = 0; i < 3; ++i) {
        max_error = std::max(max_error, std::abs(b[i] - x_true[i]));
    }
    GPU_REQUIRE(max_error < 1.e-12);
    return gpu_pass();
}

GPU_TEST GpuTestResult factor_solve() {
    std::array<std::array<Real, 2>, 2> A = {{
        {{0.0, 2.0}},
        {{1.0, 1.0}}
    }};
    std::array<Real, 2> b = {4.0, 3.0};
    std::array<int, 2> pivot{};
    const int info = linalg::lu_factor_solve<2, true>(A, pivot, b);
    GPU_REQUIRE(info == 0);
    GPU_REQUIRE_CLOSE(b[0], 1.0, 1.e-12);
    GPU_REQUIRE_CLOSE(b[1], 2.0, 1.e-12);
    return gpu_pass();
}

GPU_TEST GpuTestResult vector_norms() {
    const std::array<Real, 3> v = {3.0, 4.0, 0.0};
    const Real norm2 = linalg::norm2(v);
    const Real expected_norm2 = std::sqrt((9.0 + 16.0) / 3.0);
    const Real norm_inf = linalg::norm_inf(v);
    GPU_REQUIRE_CLOSE(norm2, expected_norm2, 1.e-12);
    GPU_REQUIRE_CLOSE(norm_inf, 4.0, 1.e-12);
    return gpu_pass();
}

GPU_TEST GpuTestResult matrix_vector() {
    const std::array<std::array<Real, 2>, 2> A = {{
        {{1.0, 2.0}},
        {{3.0, 4.0}}
    }};
    const std::array<Real, 2> x = {1.0, 2.0};
    std::array<Real, 2> y{};
    linalg::matvec(A, x, y);
    GPU_REQUIRE_CLOSE(y[0], 5.0, 1.e-12);
    GPU_REQUIRE_CLOSE(y[1], 11.0, 1.e-12);
    return gpu_pass();
}

} // namespace linalg_cases

namespace convergence_cases {

struct ExponentialGrowth {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        dydt[0] = y[0];
    }

    TEST_DEVICE static void jacobian([[maybe_unused]] Real t,
                                     [[maybe_unused]] const state_type& y,
                                     jacobian_type& jac) {
        jac[0][0] = 1.0;
    }
};

template<typename Integrator, typename State>
TEST_DEVICE Real compute_error(Integrator& integrator, State& state, Real dt) {
    ExponentialGrowth::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = dt;
    state.y[0] = 1.0;
    state.rtol = 1.e-12;
    state.atol = 1.e-16;

    const auto result = integrator.integrate(problem_state, state);
    if (result != IntegratorResult::SUCCESS) {
        return -1.0;
    }
    return std::abs(state.y[0] - std::exp(dt));
}

GPU_TEST GpuTestResult backward_euler_convergence() {
    auto integrator = BackwardEuler<ExponentialGrowth>{};
    const Real dts[4] = {0.1, 0.05, 0.025, 0.0125};
    Real errors[4] = {};

    for (int i = 0; i < 4; ++i) {
        auto state = BackwardEulerState<1>{};
        state.jacobian_analytic = true;
        errors[i] = compute_error(integrator, state, dts[i]);
        GPU_REQUIRE(errors[i] >= 0.0);
    }

    const Real rate = std::log(errors[0] / errors[1]) /
                      std::log(dts[0] / dts[1]);
    GPU_REQUIRE(rate >= 0.8);
    return gpu_pass();
}

GPU_TEST GpuTestResult vode_convergence() {
    auto integrator = VODE<ExponentialGrowth>{};
    const Real tolerances[4] = {1.e-4, 1.e-6, 1.e-8, 1.e-10};
    Real errors[4] = {};

    for (int i = 0; i < 4; ++i) {
        auto state = VODEState<1>{};
        state.jacobian_analytic = true;
        state.rtol = tolerances[i];
        state.atol = 1.e-10;
        if (tolerances[i] <= 1.e-6) {
            state.max_steps = 200000;
        }
        ExponentialGrowth::state_type problem_state = {1.0};
        state.t = 0.0;
        state.tout = 1.0;
        state.y[0] = 1.0;

        const auto result = integrator.integrate(problem_state, state);
        GPU_REQUIRE(result == IntegratorResult::SUCCESS);
        errors[i] = std::abs(state.y[0] - std::exp(1.0));
    }

    for (int i = 1; i < 4; ++i) {
        GPU_REQUIRE(errors[i] < errors[i - 1]);
    }
    return gpu_pass();
}

} // namespace convergence_cases

namespace backward_euler_cases {

struct Growth {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        dydt[0] = y[0];
    }

    TEST_DEVICE static void jacobian([[maybe_unused]] Real t,
                                     [[maybe_unused]] const state_type& y,
                                     jacobian_type& jac) {
        jac[0][0] = 1.0;
    }
};

struct Decay {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        dydt[0] = -y[0];
    }

    TEST_DEVICE static void jacobian([[maybe_unused]] Real t,
                                     [[maybe_unused]] const state_type& y,
                                     jacobian_type& jac) {
        jac[0][0] = -1.0;
    }
};

GPU_TEST GpuTestResult failed_single_step_preserves_state() {
    auto integrator = BackwardEuler<Growth>{};
    auto state = BackwardEulerState<1>{};
    Growth::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = 1.0;
    state.dt = 1.0;
    state.y = {1.0};
    state.jacobian_analytic = true;

    const Real t_old = state.t;
    const Real y_old = state.y[0];
    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::LU_DECOMPOSITION_ERROR);
    GPU_REQUIRE(state.n_step == 0);
    GPU_REQUIRE(state.t == t_old);
    GPU_REQUIRE(state.y[0] == y_old);
    return gpu_pass();
}

GPU_TEST GpuTestResult reverse_time_integration() {
    auto integrator = BackwardEuler<Decay>{};
    auto state = BackwardEulerState<1>{};
    Decay::state_type problem_state = {std::exp(-1.0)};
    state.t = 1.0;
    state.tout = 0.0;
    state.dt = 0.0;
    state.y = {std::exp(-1.0)};
    state.jacobian_analytic = true;
    state.rtol = 1.e-9;
    state.atol = 1.e-12;

    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(state.t == state.tout);
    GPU_REQUIRE(std::abs(state.y[0] - 1.0) < 5.e-3);
    return gpu_pass();
}

GPU_TEST GpuTestResult zero_solution_converges() {
    auto integrator = BackwardEuler<Decay>{};
    auto state = BackwardEulerState<1>{};
    Decay::state_type problem_state = {0.0};
    state.t = 0.0;
    state.tout = 1.0;
    state.dt = 1.0;
    state.y = {0.0};
    state.jacobian_analytic = true;
    state.tolerance = 1.e-12;

    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(state.y[0] == 0.0);
    return gpu_pass();
}

GPU_TEST GpuTestResult numerical_jacobian_rhs_count() {
    auto integrator = BackwardEuler<Decay>{};
    auto state = BackwardEulerState<1>{};
    Decay::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = 0.25;
    state.dt = 0.25;
    state.y = {1.0};
    state.jacobian_analytic = false;
    state.max_iter = 1;
    state.tolerance = 1.e6;

    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(state.n_rhs == 4);
    return gpu_pass();
}

} // namespace backward_euler_cases

namespace yass_cases {

struct StiffDecay {
    static constexpr size_type neqs = 1;
    static constexpr Real rate = 4.0;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        dydt[0] = -rate * y[0];
    }

    TEST_DEVICE static void jacobian([[maybe_unused]] Real t,
                                     [[maybe_unused]] const state_type& y,
                                     jacobian_type& jac) {
        jac[0][0] = -rate;
    }
};

struct RhsOnlyDecay {
    static constexpr size_type neqs = 1;
    static constexpr Real rate = 2.0;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        dydt[0] = -rate * y[0];
    }
};

struct ReversiblePair {
    static constexpr size_type neqs = 2;
    static constexpr Real forward = 1000.0;
    static constexpr Real reverse = 1.0;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        const Real flux = forward * y[0] - reverse * y[1];
        dydt[0] = -flux;
        dydt[1] = flux;
    }

    TEST_DEVICE static void jacobian([[maybe_unused]] Real t,
                                     [[maybe_unused]] const state_type& y,
                                     jacobian_type& jac) {
        jac[0][0] = -forward;
        jac[0][1] = reverse;
        jac[1][0] = forward;
        jac[1][1] = -reverse;
    }
};

TEST_DEVICE Real reverse_equilibrium_fraction() {
    return ReversiblePair::reverse /
           (ReversiblePair::forward + ReversiblePair::reverse);
}

GPU_TEST GpuTestResult scalar_step() {
    auto integrator = YASS<StiffDecay>{};
    auto state = YASSState<1>{};
    StiffDecay::state_type problem_state = {2.0};
    state.t = 0.0;
    state.tout = 0.25;
    state.dt = 0.25;
    state.y = {2.0};
    state.jacobian_analytic = true;

    const auto result = integrator.integrate(problem_state, state);
    const Real expected = 2.0 / (1.0 + StiffDecay::rate * state.dt);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(state.t == state.tout);
    GPU_REQUIRE(state.n_step == 1);
    GPU_REQUIRE(state.n_rhs == 1);
    GPU_REQUIRE(state.n_jac == 1);
    GPU_REQUIRE_CLOSE(state.y[0], expected, 1.e-14);
    return gpu_pass();
}

GPU_TEST GpuTestResult linear_invariant() {
    auto integrator = YASS<ReversiblePair>{};
    auto state = YASSState<2>{};
    ReversiblePair::state_type problem_state = {1.0, 0.0};
    state.t = 0.0;
    state.tout = 10.0;
    state.dt = 10.0;
    state.y = {1.0, 0.0};
    state.jacobian_analytic = true;

    const Real sum0 = state.y[0] + state.y[1];
    const auto result = integrator.integrate(problem_state, state);
    const Real sum1 = state.y[0] + state.y[1];
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(std::abs(sum1 - sum0) < 1.e-12);
    GPU_REQUIRE(std::abs(state.y[0] - reverse_equilibrium_fraction()) < 2.e-4);
    return gpu_pass();
}

GPU_TEST GpuTestResult numerical_jacobian_fallback() {
    auto integrator = YASS<RhsOnlyDecay>{};
    auto state = YASSState<1>{};
    RhsOnlyDecay::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = 0.1;
    state.dt = 0.1;
    state.y = {1.0};
    state.jacobian_analytic = false;

    const auto result = integrator.integrate(problem_state, state);
    const Real expected = 1.0 / (1.0 + RhsOnlyDecay::rate * state.dt);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE_CLOSE(state.y[0], expected, 1.e-12);
    GPU_REQUIRE(state.n_rhs == 3);
    GPU_REQUIRE(state.n_jac == 1);
    return gpu_pass();
}

GPU_TEST GpuTestResult factory_support() {
    using Factory = IntegratorFactory<StiffDecay>;
    auto integrator = Factory::create<Factory::Type::YASS>();
    auto state = Factory::state_type<Factory::Type::YASS>{};
    StiffDecay::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = 0.1;
    state.dt = 0.1;
    state.y = {1.0};
    state.jacobian_analytic = true;

    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(state.y[0] < 1.0);
    return gpu_pass();
}

} // namespace yass_cases

namespace optional_jacobian_cases {

struct RhsOnlyDecay {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        dydt[0] = -2.0 * y[0];
    }
};

GPU_TEST GpuTestResult backward_euler_rhs_only() {
    auto integrator = BackwardEuler<RhsOnlyDecay>{};
    auto state = BackwardEulerState<1>{};
    RhsOnlyDecay::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = 0.1;
    state.dt = 0.05;
    state.y = {1.0};
    state.jacobian_analytic = false;

    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(state.y[0] < 1.0);
    return gpu_pass();
}

GPU_TEST GpuTestResult vode_rhs_only() {
    auto integrator = VODE<RhsOnlyDecay>{};
    auto state = VODEState<1>{};
    RhsOnlyDecay::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = 0.1;
    state.y = {1.0};
    state.jacobian_analytic = false;
    state.rtol = 1.e-8;
    state.atol = 1.e-12;

    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(std::abs(state.t - state.tout) < 1.e-15);
    GPU_REQUIRE(state.y[0] < 1.0);
    return gpu_pass();
}

} // namespace optional_jacobian_cases

namespace vode_cases {

struct ExpODE {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;
    static constexpr Real k = 2.5;

    TEST_DEVICE static void rhs([[maybe_unused]] Real t, const state_type& y,
                                rhs_type& dydt) {
        dydt[0] = k * y[0];
    }

    TEST_DEVICE static void jacobian([[maybe_unused]] Real t,
                                     [[maybe_unused]] const state_type& y,
                                     jacobian_type& J) {
        J[0][0] = k;
    }
};

GPU_TEST GpuTestResult expode_smoke() {
    constexpr int Ncases = 32;
    const Real dt = 0.1;
    const Real exact = std::exp(ExpODE::k * dt);
    const Real tol = 1.e-6;
    for (int i = 0; i < Ncases; ++i) {
        (void)i;
        auto integ = VODE<ExpODE>{};
        auto s = VODEState<ExpODE::neqs>{};
        s.jacobian_analytic = true;
        s.t = 0.0;
        s.tout = dt;
        s.y[0] = 1.0;
        s.rtol = tol;
        s.atol = 1.e-12;
        auto ps = ExpODE::state_type{1.0};
        const auto result = integ.integrate(ps, s);
        GPU_REQUIRE(result == IntegratorResult::SUCCESS);
        GPU_REQUIRE(std::abs(s.y[0] - exact) <= 1.e3 * tol);
    }
    return gpu_pass();
}

struct StiffDecay {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;
    static constexpr Real lambda = 1.0e6;

    TEST_DEVICE static void rhs(Real, const state_type& y, rhs_type& dydt) {
        dydt[0] = -lambda * y[0];
    }

    TEST_DEVICE static void jacobian(Real, const state_type&, jacobian_type& J) {
        J[0][0] = -lambda;
    }
};

GPU_TEST GpuTestResult stiff_decay_convergence() {
    const Real T = 1.0e-5;
    const Real exact = std::exp(-StiffDecay::lambda * T);
    const Real tols[3] = {1.e-4, 1.e-6, 1.e-8};
    Real errors[3] = {};

    for (int i = 0; i < 3; ++i) {
        auto integ = VODE<StiffDecay>{};
        auto s = VODEState<1>{};
        s.jacobian_analytic = true;
        s.t = 0.0;
        s.tout = T;
        s.y[0] = 1.0;
        s.rtol = tols[i];
        s.atol = 1.e-8;
        s.max_steps = 100000;
        auto ps = StiffDecay::state_type{1.0};
        const auto result = integ.integrate(ps, s);
        GPU_REQUIRE(result == IntegratorResult::SUCCESS);
        errors[i] = std::abs(s.y[0] - exact);
        GPU_REQUIRE(errors[i] <= 1.e3 * tols[i]);
    }

    GPU_REQUIRE(errors[1] < errors[0]);
    GPU_REQUIRE(errors[2] < errors[1]);
    return gpu_pass();
}

struct Robertson {
    static constexpr size_type neqs = 3;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs(Real, const state_type& y, rhs_type& dydt) {
        constexpr Real k1 = 0.04;
        constexpr Real k2 = 1.0e4;
        constexpr Real k3 = 3.0e7;
        dydt[0] = -k1 * y[0] + k2 * y[1] * y[2];
        dydt[1] = k1 * y[0] - k2 * y[1] * y[2] - k3 * y[1] * y[1];
        dydt[2] = k3 * y[1] * y[1];
    }

    TEST_DEVICE static void jacobian(Real, const state_type& y,
                                     jacobian_type& jac) {
        constexpr Real k1 = 0.04;
        constexpr Real k2 = 1.0e4;
        constexpr Real k3 = 3.0e7;
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

GPU_TEST GpuTestResult robertson_strict() {
    const Real tols[3] = {1.e-6, 1.e-8, 1.e-10};
    Robertson::state_type ys[3] = {};
    Real diffs[2] = {};

    for (int k = 0; k < 3; ++k) {
        auto integrator = VODE<Robertson>{};
        auto state = VODEState<3>{};
        state.jacobian_analytic = true;
        state.t = 0.0;
        state.tout = 1.0;
        state.y = {1.0, 0.0, 0.0};
        state.rtol = tols[k];
        state.atol = 1.e-10;
        auto problem_state = Robertson::state_type{1.0, 0.0, 0.0};
        const auto result = integrator.integrate(problem_state, state);
        GPU_REQUIRE(result == IntegratorResult::SUCCESS);
        ys[k] = state.y;
        const Real total = state.y[0] + state.y[1] + state.y[2];
        GPU_REQUIRE(std::abs(total - 1.0) <= 1.e-6);
    }

    for (int k = 1; k < 3; ++k) {
        Real d = 0.0;
        for (size_type i = 0; i < 3; ++i) {
            d = std::max(d, std::abs(ys[k][i] - ys[k - 1][i]));
        }
        diffs[k - 1] = d;
    }
    GPU_REQUIRE(diffs[1] < diffs[0]);
    GPU_REQUIRE(diffs[1] <= 1.e-6);
    return gpu_pass();
}

struct LinearExchange4 {
    static constexpr size_type neqs = 4;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;
    static constexpr Real k12 = 1.0e3;
    static constexpr Real k21 = 2.0e3;
    static constexpr Real k34 = 1.0e-2;
    static constexpr Real k43 = 3.0e-2;

    TEST_DEVICE static void rhs(Real, const state_type& y, rhs_type& f) {
        f[0] = -k12 * y[0] + k21 * y[1];
        f[1] = k12 * y[0] - k21 * y[1];
        f[2] = -k34 * y[2] + k43 * y[3];
        f[3] = k34 * y[2] - k43 * y[3];
    }

    TEST_DEVICE static void jacobian(Real, const state_type&, jacobian_type& J) {
        for (size_type i = 0; i < neqs; ++i) {
            for (size_type j = 0; j < neqs; ++j) {
                J[i][j] = 0.0;
            }
        }
        J[0][0] = -k12;
        J[0][1] = k21;
        J[1][0] = k12;
        J[1][1] = -k21;
        J[2][2] = -k34;
        J[2][3] = k43;
        J[3][2] = k34;
        J[3][3] = -k43;
    }
};

GPU_TEST GpuTestResult linear_invariants_analytic() {
    const LinearExchange4::state_type y0 = {1.0, 0.0, 2.0, 0.0};
    const Real inv12_0 = y0[0] + y0[1];
    const Real inv34_0 = y0[2] + y0[3];
    const Real invtot_0 = inv12_0 + inv34_0;

    auto integ = VODE<LinearExchange4>{};
    auto s = VODEState<LinearExchange4::neqs>{};
    s.jacobian_analytic = true;
    s.t = 0.0;
    s.tout = 10.0;
    s.y = y0;
    s.rtol = 1.e-6;
    s.atol = 1.e-10;
    s.max_steps = 200000;
    auto problem_state = y0;
    const auto result = integ.integrate(problem_state, s);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);

    const Real inv12 = s.y[0] + s.y[1];
    const Real inv34 = s.y[2] + s.y[3];
    const Real invtot = inv12 + inv34;
    Real scale = std::max(std::abs(inv12_0), std::abs(inv34_0));
    scale = std::max(scale, std::abs(invtot_0));
    scale = std::max(scale, Real(1.0));
    const Real tol = 1.0e3 * math::UROUND * scale;
    GPU_REQUIRE(std::abs(inv12 - inv12_0) <= 10.0 * tol);
    GPU_REQUIRE(std::abs(inv34 - inv34_0) <= 10.0 * tol);
    GPU_REQUIRE(std::abs(invtot - invtot_0) <= 10.0 * tol);
    return gpu_pass();
}

GPU_TEST GpuTestResult linear_invariants_numjac() {
    const LinearExchange4::state_type y0 = {1.0, 0.0, 2.0, 0.0};
    const Real inv12_0 = y0[0] + y0[1];
    const Real inv34_0 = y0[2] + y0[3];
    const Real invtot_0 = inv12_0 + inv34_0;

    auto integ = VODE<LinearExchange4>{};
    auto s = VODEState<LinearExchange4::neqs>{};
    s.jacobian_analytic = false;
    s.t = 0.0;
    s.tout = 1000.0;
    s.y = y0;
    s.rtol = 1.e-9;
    s.atol = 1.e-12;
    s.max_steps = 200000;
    auto problem_state = y0;
    const auto result = integ.integrate(problem_state, s);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);

    const Real inv12 = s.y[0] + s.y[1];
    const Real inv34 = s.y[2] + s.y[3];
    const Real invtot = inv12 + inv34;
    Real scale = std::max(std::abs(inv12_0), std::abs(inv34_0));
    scale = std::max(scale, std::abs(invtot_0));
    scale = std::max(scale, Real(1.0));
    const Real tol = 1.0e3 * math::UROUND * scale;
    GPU_REQUIRE(std::abs(inv12 - inv12_0) <= 10.0 * tol);
    GPU_REQUIRE(std::abs(inv34 - inv34_0) <= 10.0 * tol);
    GPU_REQUIRE(std::abs(invtot - invtot_0) <= 10.0 * tol);
    return gpu_pass();
}

struct StiffDecayWithJac {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs(Real, const state_type& y, rhs_type& dydt) {
        dydt[0] = -1000.0 * y[0];
    }

    TEST_DEVICE static void jacobian(Real, const state_type&, jacobian_type& jac) {
        jac[0][0] = -1000.0;
    }
};

GPU_TEST GpuTestResult jacobian_counter_path() {
    auto integrator = VODE<StiffDecayWithJac>{};
    auto state = VODEState<1>{};
    StiffDecayWithJac::state_type problem_state = {1.0};
    state.t = 0.0;
    state.tout = 1.0;
    state.y = {1.0};
    state.jacobian_analytic = true;
    state.rtol = 1.e-8;
    state.atol = 1.e-12;

    const auto result = integrator.integrate(problem_state, state);
    GPU_REQUIRE(result == IntegratorResult::SUCCESS);
    GPU_REQUIRE(state.n_jac > 0);
    return gpu_pass();
}

struct HIRES {
    static constexpr size_type neqs = 8;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;

    TEST_DEVICE static void rhs(Real, const state_type& y, rhs_type& f) {
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
        f[1] = c1 * y[0] - c4 * y[1];
        f[2] = -c5 * y[2] + c2 * y[3] + c6 * y[4];
        f[3] = c3 * y[1] + c1 * y[2] - c7 * y[3];
        f[4] = -c8 * y[4] + c2 * y[5] + c2 * y[6];
        f[5] = -c9 * y[5] * y[7] + c10 * y[3] + c1 * y[4] -
               c2 * y[5] + c10 * y[6];
        f[6] = c9 * y[5] * y[7] - c11 * y[6];
        f[7] = -c9 * y[5] * y[7] + c11 * y[6];
    }

    TEST_DEVICE static void jacobian(Real, const state_type& y, jacobian_type& J) {
        for (size_type i = 0; i < neqs; ++i) {
            for (size_type j = 0; j < neqs; ++j) {
                J[i][j] = 0.0;
            }
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
        J[0][0] = -c1;
        J[0][1] = c2;
        J[0][2] = c3;
        J[1][0] = c1;
        J[1][1] = -c4;
        J[2][2] = -c5;
        J[2][3] = c2;
        J[2][4] = c6;
        J[3][1] = c3;
        J[3][2] = c1;
        J[3][3] = -c7;
        J[4][4] = -c8;
        J[4][5] = c2;
        J[4][6] = c2;
        J[5][3] = c10;
        J[5][4] = c1;
        J[5][5] = -(c9 * y[7] + c2);
        J[5][6] = c10;
        J[5][7] = -c9 * y[5];
        J[6][5] = c9 * y[7];
        J[6][6] = -c11;
        J[6][7] = c9 * y[5];
        J[7][5] = -c9 * y[7];
        J[7][6] = c11;
        J[7][7] = -c9 * y[5];
    }
};

GPU_TEST GpuTestResult hires_benchmark() {
    const HIRES::state_type y0 = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.7e-3};
    const Real t_end = 321.8122;

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
    const auto rc_ref = integ_ref.integrate(ps_ref, s_ref);
    GPU_REQUIRE(rc_ref == IntegratorResult::SUCCESS);

    Real min_ref = s_ref.y[0];
    for (size_type i = 0; i < HIRES::neqs; ++i) {
        min_ref = std::min(min_ref, s_ref.y[i]);
    }
    GPU_REQUIRE(min_ref >= -1.e-12);

    const Real test_tols[2] = {1.e-6, 1.e-8};
    Real diffs[2] = {};
    for (int k = 0; k < 2; ++k) {
        auto integ = VODE<HIRES>{};
        auto s = VODEState<HIRES::neqs>{};
        s.jacobian_analytic = true;
        s.t = 0.0;
        s.tout = t_end;
        s.y = y0;
        s.rtol = test_tols[k];
        s.atol = 1.e-12;
        s.max_steps = 500000;
        auto ps = y0;
        const auto rc = integ.integrate(ps, s);
        GPU_REQUIRE(rc == IntegratorResult::SUCCESS);

        Real miny = s.y[0];
        for (size_type i = 0; i < HIRES::neqs; ++i) {
            miny = std::min(miny, s.y[i]);
        }
        GPU_REQUIRE(miny >= -1.e-10);
        diffs[k] = max_abs_diff(s.y, s_ref.y);
    }

    GPU_REQUIRE(diffs[1] < diffs[0]);
    GPU_REQUIRE(diffs[0] <= 5.e-5);
    GPU_REQUIRE(diffs[1] <= 5.e-7);
    return gpu_pass();
}

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

    TEST_DEVICE static void rhs(Real, const state_type& u, rhs_type& du) {
        const Real T054 = std::pow(T, 0.54);
        const Real T061 = std::pow(T, 0.61);
        const Real T065 = std::pow(T, 0.65);
        const Real T05 = std::sqrt(T);
        const Real e_m15Av = std::exp(-1.5 * Av);
        const Real e_m17Av = std::exp(-1.7 * Av);
        const Real e_m19Av = std::exp(-1.9 * Av);
        const Real e_m3Av = std::exp(-3.0 * Av);
        const Real& u1 = u[0];
        const Real& u2 = u[1];
        const Real& u3 = u[2];
        const Real& u4 = u[3];
        const Real& u5 = u[4];
        const Real& u6 = u[5];
        const Real& u7 = u[6];
        const Real& u8 = u[7];
        const Real& u9 = u[8];
        const Real& u10 = u[9];
        const Real& u11 = u[10];
        const Real& u12 = u[11];
        const Real& u13 = u[12];
        const Real& u14 = u[13];

        du[0] = -1.2e-17 * u1 + n_H * (1.9e-6 * u2 * u3) / T054 -
                n_H * 4.e-16 * u1 * u12 - n_H * 7.e-15 * u1 * u5 +
                n_H * 1.7e-9 * u10 * u2 + n_H * 2.e-9 * u2 * u6 +
                n_H * 2.e-9 * u2 * u14 + n_H * 8.e-10 * u2 * u8;
        du[1] = 1.2e-17 * u1 + n_H * (-1.9e-6 * u3 * u2) / T054 -
                n_H * 1.7e-9 * u10 * u2 - n_H * 2.e-9 * u2 * u6 -
                n_H * 2.e-9 * u2 * u14 - n_H * 8.e-10 * u2 * u8;
        du[2] = n_H * (-1.4e-10 * u3 * u12) / T061 -
                n_H * (3.8e-10 * u13 * u3) / T065 -
                n_H * (3.3e-5 * u11 * u3) / T + 1.2e-17 * u1 -
                n_H * (1.9e-6 * u3 * u2) / T054 + 6.8e-18 * u4 -
                n_H * (9.e-11 * u3 * u5) / std::pow(T, 0.64) +
                3.e-10 * Go * e_m3Av * u6 + n_H * 2.e-9 * u2 * u13 +
                2.0e-10 * Go * e_m19Av * u14;
        du[3] = n_H * (9.e-11 * u3 * u5) / std::pow(T, 0.64) -
                6.8e-18 * u4 + n_H * 7.e-15 * u1 * u5 +
                n_H * 1.6e-9 * u10 * u5;
        du[4] = 6.8e-18 * u4 -
                n_H * (9.e-11 * u3 * u5) / std::pow(T, 0.64) -
                n_H * 7.e-15 * u1 * u5 - n_H * 1.6e-9 * u10 * u5;
        du[5] = n_H * (1.4e-10 * u3 * u12) / T061 -
                n_H * 2.e-9 * u2 * u6 - n_H * 5.8e-12 * T05 * u9 * u6 +
                1.e-9 * Go * e_m15Av * u7 - 3.e-10 * Go * e_m3Av * u6 +
                1.e-10 * Go * e_m3Av * u10 * shield;
        du[6] = n_H * (-2.e-10) * u7 * u8 + n_H * 4.e-16 * u1 * u12 +
                n_H * 2.e-9 * u2 * u6 - 1.e-9 * Go * u7 * e_m15Av;
        du[7] = n_H * (-2.e-10) * u7 * u8 + n_H * 1.6e-9 * u10 * u5 -
                n_H * 8.e-10 * u2 * u8 + 5.e-10 * Go * e_m17Av * u9 +
                1.e-10 * Go * e_m3Av * u10 * shield;
        du[8] = n_H * (-1.e-9) * u9 * u12 + n_H * 8.e-10 * u2 * u8 -
                n_H * 5.8e-12 * T05 * u9 * u6 -
                5.e-10 * Go * e_m17Av * u9;
        du[9] = n_H * (3.3e-5 * u11 * u3) / T + n_H * 2.e-10 * u7 * u8 -
                n_H * 1.7e-9 * u10 * u2 - n_H * 1.6e-9 * u10 * u5 +
                n_H * 5.8e-12 * T05 * u9 * u6 -
                1.e-10 * Go * e_m3Av * u10 +
                1.5e-10 * Go * std::exp(-2.5 * Av) * u11 * shield;
        du[10] = n_H * (-3.3e-5 * u11 * u3) / T +
                 n_H * 1.e-9 * u9 * u12 + n_H * 1.7e-9 * u10 * u2 -
                 1.5e-10 * Go * std::exp(-2.5 * Av) * u11;
        du[11] = n_H * (-1.4e-10 * u3 * u12) / T061 -
                 n_H * 4.e-16 * u1 * u12 - n_H * 1.e-9 * u9 * u12 +
                 n_H * 1.6e-9 * u10 * u5 + 3.e-10 * Go * e_m3Av * u6;
        du[12] = n_H * (-3.8e-10 * u13 * u3) / T065 +
                 n_H * 2.e-9 * u2 * u14 + 2.0e-10 * Go * e_m19Av * u14;
        du[13] = n_H * (3.8e-10 * u13 * u3) / T065 -
                 n_H * 2.e-9 * u2 * u14 - 2.0e-10 * Go * e_m19Av * u14;
    }

    TEST_DEVICE static void jacobian(Real, const state_type&, jacobian_type& J) {
        for (size_type i = 0; i < neqs; ++i) {
            for (size_type j = 0; j < neqs; ++j) {
                J[i][j] = (i == j ? 1.0 : 0.0);
            }
        }
    }
};

GPU_TEST GpuTestResult nelson_network() {
    const Real seconds_per_year = 3600.0 * 24.0 * 365.0;
    const Real t_end = 30000.0 * seconds_per_year;
    const Nelson::state_type y0 = {
        0.5, 9.059e-9, 2.0e-4, 0.1, 7.866e-7,
        0.0, 0.0, 4.0e-4, 0.0, 0.0, 0.0, 2.0e-4, 2.0e-7, 2.0e-7
    };
    const Nelson::state_type y_ref = {
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
    s.jacobian_analytic = false;
    s.t = 0.0;
    s.tout = t_end;
    s.y = y0;
    s.rtol = 1.e-4;
    s.atol = 1.e-6;
    s.max_steps = 1000;
    auto problem_state = y0;
    const auto rc = integ.integrate(problem_state, s);
    GPU_REQUIRE(rc == IntegratorResult::SUCCESS);

    Real num = 0.0;
    Real den = 0.0;
    for (size_type i = 0; i < Nelson::neqs; ++i) {
        const Real di = s.y[i] - y_ref[i];
        num += di * di;
        den += y_ref[i] * y_ref[i];
    }
    const Real l2_rel_err = std::sqrt(num) / std::sqrt(den);
    GPU_REQUIRE(l2_rel_err < 3.0e-6);
    return gpu_pass();
}

} // namespace vode_cases

enum TestId {
    TEST_LINALG_LU_DECOMPOSITION,
    TEST_LINALG_MATRIX_SOLVE,
    TEST_LINALG_MATRIX_SOLVE_REGRESSION,
    TEST_LINALG_FACTOR_SOLVE,
    TEST_LINALG_VECTOR_NORMS,
    TEST_LINALG_MATRIX_VECTOR,
    TEST_CONVERGENCE_BE,
    TEST_CONVERGENCE_VODE,
    TEST_BE_FAILED_SINGLE_STEP,
    TEST_BE_REVERSE_TIME,
    TEST_BE_ZERO_SOLUTION,
    TEST_BE_NUMERICAL_JACOBIAN_RHS_COUNT,
    TEST_YASS_SCALAR_STEP,
    TEST_YASS_LINEAR_INVARIANT,
    TEST_YASS_NUMERICAL_JACOBIAN,
    TEST_YASS_FACTORY,
    TEST_OPTIONAL_JACOBIAN_BE,
    TEST_OPTIONAL_JACOBIAN_VODE,
    TEST_VODE_EXP_SMOKE,
    TEST_VODE_STIFF_DECAY,
    TEST_VODE_ROBERTSON,
    TEST_VODE_LINEAR_INVARIANTS_ANALYTIC,
    TEST_VODE_LINEAR_INVARIANTS_NUMJAC,
    TEST_VODE_JACOBIAN_COUNTER_PATH,
    TEST_VODE_HIRES,
    TEST_VODE_NELSON,
    TEST_COUNT
};

__global__ void run_case_kernel(int test_id, GpuTestResult* result) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    switch (test_id) {
    case TEST_LINALG_LU_DECOMPOSITION:
        *result = linalg_cases::lu_decomposition();
        break;
    case TEST_LINALG_MATRIX_SOLVE:
        *result = linalg_cases::matrix_solve();
        break;
    case TEST_LINALG_MATRIX_SOLVE_REGRESSION:
        *result = linalg_cases::matrix_solve_regression();
        break;
    case TEST_LINALG_FACTOR_SOLVE:
        *result = linalg_cases::factor_solve();
        break;
    case TEST_LINALG_VECTOR_NORMS:
        *result = linalg_cases::vector_norms();
        break;
    case TEST_LINALG_MATRIX_VECTOR:
        *result = linalg_cases::matrix_vector();
        break;
    case TEST_CONVERGENCE_BE:
        *result = convergence_cases::backward_euler_convergence();
        break;
    case TEST_CONVERGENCE_VODE:
        *result = convergence_cases::vode_convergence();
        break;
    case TEST_BE_FAILED_SINGLE_STEP:
        *result = backward_euler_cases::failed_single_step_preserves_state();
        break;
    case TEST_BE_REVERSE_TIME:
        *result = backward_euler_cases::reverse_time_integration();
        break;
    case TEST_BE_ZERO_SOLUTION:
        *result = backward_euler_cases::zero_solution_converges();
        break;
    case TEST_BE_NUMERICAL_JACOBIAN_RHS_COUNT:
        *result = backward_euler_cases::numerical_jacobian_rhs_count();
        break;
    case TEST_YASS_SCALAR_STEP:
        *result = yass_cases::scalar_step();
        break;
    case TEST_YASS_LINEAR_INVARIANT:
        *result = yass_cases::linear_invariant();
        break;
    case TEST_YASS_NUMERICAL_JACOBIAN:
        *result = yass_cases::numerical_jacobian_fallback();
        break;
    case TEST_YASS_FACTORY:
        *result = yass_cases::factory_support();
        break;
    case TEST_OPTIONAL_JACOBIAN_BE:
        *result = optional_jacobian_cases::backward_euler_rhs_only();
        break;
    case TEST_OPTIONAL_JACOBIAN_VODE:
        *result = optional_jacobian_cases::vode_rhs_only();
        break;
    case TEST_VODE_EXP_SMOKE:
        *result = vode_cases::expode_smoke();
        break;
    case TEST_VODE_STIFF_DECAY:
        *result = vode_cases::stiff_decay_convergence();
        break;
    case TEST_VODE_ROBERTSON:
        *result = vode_cases::robertson_strict();
        break;
    case TEST_VODE_LINEAR_INVARIANTS_ANALYTIC:
        *result = vode_cases::linear_invariants_analytic();
        break;
    case TEST_VODE_LINEAR_INVARIANTS_NUMJAC:
        *result = vode_cases::linear_invariants_numjac();
        break;
    case TEST_VODE_JACOBIAN_COUNTER_PATH:
        *result = vode_cases::jacobian_counter_path();
        break;
    case TEST_VODE_HIRES:
        *result = vode_cases::hires_benchmark();
        break;
    case TEST_VODE_NELSON:
        *result = vode_cases::nelson_network();
        break;
    default:
        *result = gpu_fail(__LINE__, test_id);
        break;
    }
}

const char* test_name(int test_id) {
    static const char* names[TEST_COUNT] = {
        "linear_algebra.lu_decomposition",
        "linear_algebra.matrix_solve",
        "linear_algebra.matrix_solve_regression",
        "linear_algebra.factor_solve",
        "linear_algebra.vector_norms",
        "linear_algebra.matrix_vector",
        "convergence.backward_euler",
        "convergence.vode",
        "backward_euler.failed_single_step_preserves_state",
        "backward_euler.reverse_time_integration",
        "backward_euler.zero_solution_converges",
        "backward_euler.numerical_jacobian_rhs_count",
        "yass.scalar_step",
        "yass.linear_invariant",
        "yass.numerical_jacobian_fallback",
        "yass.factory_support",
        "optional_jacobian.backward_euler_rhs_only",
        "optional_jacobian.vode_rhs_only",
        "vode.expode_smoke",
        "vode.stiff_decay_convergence",
        "vode.robertson_strict",
        "vode.linear_invariants_analytic",
        "vode.linear_invariants_numjac",
        "vode.jacobian_counter_path",
        "vode.hires_benchmark",
        "vode.nelson_network"
    };
    return names[test_id];
}

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaErrorNoDevice || device_count == 0) {
        std::printf("GPU test suite: SKIPPED (no CUDA-capable device)\n");
        return 77;
    }
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    GpuTestResult* device_result = nullptr;
    err = cudaMalloc(&device_result, sizeof(GpuTestResult));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int failed = 0;
    for (int test_id = 0; test_id < TEST_COUNT; ++test_id) {
        GpuTestResult host_result = {0, 0, 0, 0.0, 0.0};
        run_case_kernel<<<1, 1>>>(test_id, device_result);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "launch failed for %s: %s\n",
                         test_name(test_id), cudaGetErrorString(err));
            failed++;
            continue;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::fprintf(stderr, "device execution failed for %s: %s\n",
                         test_name(test_id), cudaGetErrorString(err));
            failed++;
            continue;
        }
        err = cudaMemcpy(&host_result, device_result, sizeof(GpuTestResult),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "cudaMemcpy failed for %s: %s\n",
                         test_name(test_id), cudaGetErrorString(err));
            failed++;
            continue;
        }

        if (host_result.passed) {
            std::printf("[GPU] PASS %s\n", test_name(test_id));
        } else {
            std::fprintf(stderr,
                         "[GPU] FAIL %s at line %d code=%d value=%.17e expected=%.17e\n",
                         test_name(test_id), host_result.line, host_result.code,
                         host_result.value, host_result.expected);
            failed++;
        }
    }

    cudaFree(device_result);
    if (failed != 0) {
        std::fprintf(stderr, "GPU test suite failed: %d failure(s)\n", failed);
        return 1;
    }

    std::printf("GPU test suite: PASSED (%d cases)\n", static_cast<int>(TEST_COUNT));
    return 0;
}
