// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Robertson regression test for DVODPK integrator
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <integrators/integrators.hpp>

using namespace integrators;

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

int main() {
    // Reference solution via VODE at tight tolerance
    constexpr Real final_time = 1.0e-4;

    VODE<Robertson> vode;
    VODEState<Robertson::neqs> vode_state;
    vode_state.t = 0.0;
    vode_state.tout = final_time;
    vode_state.y = {1.0, 0.0, 0.0};
    vode_state.rtol = 1.0e-6;
    vode_state.atol = 1.0e-10;
    vode_state.jacobian_analytic = true;
    vode_state.max_steps = 20000;
    auto problem_state = Robertson::state_type{1.0, 0.0, 0.0};
    const auto vode_start = std::chrono::steady_clock::now();
    const auto vode_result = vode.integrate(problem_state, vode_state);
    const auto vode_end = std::chrono::steady_clock::now();
    if (vode_result != IntegratorResult::SUCCESS) {
        std::cerr << "VODE reference integration failed with code "
                  << static_cast<int>(vode_result) << "\n";
        return 1;
    }
    const auto vode_elapsed =
        std::chrono::duration<double>(vode_end - vode_start).count();
    std::cout << "VODE Robertson integration time: " << vode_elapsed << " s\n";
    std::cout << "VODE steps: " << vode_state.n_step
              << " RHS evals: " << vode_state.n_rhs
              << " Jacobians: " << vode_state.n_jac << "\n";

    const auto reference = vode_state.y;

    // Run DVODPK with moderate tolerances
    DVODPK<Robertson>::State dv_state;
    Robertson problem{};
    dv_state.t = 0.0;
    dv_state.y = {1.0, 0.0, 0.0};
    dv_state.rtol = 1.0e-6;
    dv_state.atol = 1.0e-10;
    dv_state.max_steps = 200000;

    DVODPKConfig config;
    config.max_krylov_iters = 10;
    config.max_nonlinear_iters = 6;
    config.use_analytic_jacobian = true;
    config.max_steps = 2000000;
    config.h_min = 0.0;
    config.h_max = 0.0;

    const auto dv_start = std::chrono::steady_clock::now();
    const auto dv_result = DVODPK<Robertson>::integrate(dv_state, problem, final_time, config);
    const auto dv_end = std::chrono::steady_clock::now();
    if (dv_result != IntegratorResult::SUCCESS) {
        std::cerr << "DVODPK integration failed with code "
                  << static_cast<int>(dv_result) << "\n";
        return 1;
    }
    const auto dv_elapsed =
        std::chrono::duration<double>(dv_end - dv_start).count();
    std::cout << "DVODPK Robertson integration time: " << dv_elapsed << " s\n";
    std::cout << "DVODPK steps: " << dv_state.n_step
              << " RHS evals: " << dv_state.n_rhs
              << " GMRES iters: " << dv_state.n_linear_iters << "\n";

    const Real mass = dv_state.y[0] + dv_state.y[1] + dv_state.y[2];
    if (std::abs(mass - 1.0) >= 1.0e-6) {
        std::cerr << "Mass conservation violated: |m-1|=" << std::abs(mass - 1.0) << "\n";
        return 1;
    }

    Real max_err = 0.0;
    for (size_type i = 0; i < Robertson::neqs; ++i) {
        max_err = std::max(max_err, std::abs(dv_state.y[i] - reference[i]));
    }
    if (max_err >= 5.0e-3) {
        std::cerr << "DVODPK deviation too large: " << max_err << "\n";
        return 1;
    }

    // Longer horizon check with looser tolerances
    constexpr Real final_time_long = 1.0e-3;
    VODEState<Robertson::neqs> vode_state_long;
    vode_state_long.t = 0.0;
    vode_state_long.tout = final_time_long;
    vode_state_long.y = {1.0, 0.0, 0.0};
    vode_state_long.rtol = 1.0e-6;
    vode_state_long.atol = 1.0e-10;
    vode_state_long.jacobian_analytic = true;
    vode_state_long.max_steps = 500000;
    auto problem_state_long = Robertson::state_type{1.0, 0.0, 0.0};
    const auto vode_start_long = std::chrono::steady_clock::now();
    const auto vode_result_long = vode.integrate(problem_state_long, vode_state_long);
    const auto vode_end_long = std::chrono::steady_clock::now();
    if (vode_result_long != IntegratorResult::SUCCESS) {
        std::cerr << "VODE long-horizon integration failed with code "
                  << static_cast<int>(vode_result_long) << "\n";
        return 1;
    }
    const auto vode_elapsed_long =
        std::chrono::duration<double>(vode_end_long - vode_start_long).count();
    std::cout << "VODE Robertson long integration time: " << vode_elapsed_long << " s\n";
    std::cout << "VODE long steps: " << vode_state_long.n_step
              << " RHS evals: " << vode_state_long.n_rhs
              << " Jacobians: " << vode_state_long.n_jac << "\n";

    DVODPK<Robertson>::State dv_state_long;
    dv_state_long.t = 0.0;
    dv_state_long.y = {1.0, 0.0, 0.0};
    dv_state_long.rtol = 1.0e-6;
    dv_state_long.atol = 1.0e-10;
    dv_state_long.max_steps = 2000000;

    DVODPKConfig config_long = config;
    config_long.max_steps = 2000000;
    config_long.max_krylov_iters = 15;
    config_long.max_nonlinear_iters = 6;

    const auto dv_start_long = std::chrono::steady_clock::now();
    const auto dv_result_long = DVODPK<Robertson>::integrate(dv_state_long, problem, final_time_long, config_long);
    const auto dv_end_long = std::chrono::steady_clock::now();
    if (dv_result_long != IntegratorResult::SUCCESS) {
        std::cerr << "DVODPK long-horizon integration failed with code "
                  << static_cast<int>(dv_result_long) << "\n";
        return 1;
    }
    const auto dv_elapsed_long =
        std::chrono::duration<double>(dv_end_long - dv_start_long).count();
    std::cout << "DVODPK Robertson long integration time: " << dv_elapsed_long << " s\n";
    std::cout << "DVODPK long steps: " << dv_state_long.n_step
              << " RHS evals: " << dv_state_long.n_rhs
              << " GMRES iters: " << dv_state_long.n_linear_iters << "\n";

    Real max_err_long = 0.0;
    for (size_type i = 0; i < Robertson::neqs; ++i) {
        max_err_long = std::max(max_err_long, std::abs(dv_state_long.y[i] - vode_state_long.y[i]));
    }
    if (max_err_long >= 1.0e-2) {
        std::cerr << "DVODPK long-horizon deviation too large: " << max_err_long << "\n";
        return 1;
    }

    return 0;
}
