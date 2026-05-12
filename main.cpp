// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>

#include "primordial_chem.hpp"
#include "rodas.hpp"

namespace pc = integrators::primordial_chem;

namespace {

constexpr integrators::Real tff_reduc = 1.0e-1;
constexpr int max_collapse_steps = 1000;
constexpr integrators::Real initial_temperature = 1.0e2;
constexpr integrators::Real rtol_spec = 1.0e-4;
constexpr integrators::Real atol_spec = 1.0e-4;
constexpr integrators::Real rtol_energy = 1.0e-6;
constexpr integrators::Real atol_energy = 1.0e-6;

constexpr std::array<integrators::Real, pc::NumSpec> initial_number_densities{
    1.0e-4, 1.0e-4, 1.0e0,  1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40,
    1.0e-40, 1.0e-6, 1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40, 0.0775};

struct IntegratorStats {
    std::uint64_t internal_steps{};
    std::uint64_t rhs_calls{};
    std::uint64_t jacobian_calls{};
    std::uint64_t decompositions{};
    std::uint64_t linear_solves{};
    std::uint64_t accepted_steps{};
    std::uint64_t rejected_steps{};
};

struct CollapseState {
    pc::burn_t initial{};
    pc::burn_t current{};
    integrators::Real time{};
    integrators::Real density_driver{};
    int completed_steps{};
    IntegratorStats stats{};
};

pc::burn_t make_initial_state() {
    pc::burn_t state;
    state.T = initial_temperature;
    state.xn = initial_number_densities;
    state.rho = pc::density(state.xn);
    pc::normalize_number_densities_to_density(state);
    pc::eos_rt(state);
    return state;
}

CollapseState make_collapse_state() {
    pc::burn_t state = make_initial_state();
    return {state, state, 0.0, state.rho, 0, {}};
}

template<typename State>
void configure_ros2s(State& state) {
    state.use_vector_tolerances = true;
    for (int n = 0; n < pc::NumSpec; ++n) {
        state.rtol_vec[static_cast<std::size_t>(n)] = rtol_spec;
        state.atol_vec[static_cast<std::size_t>(n)] = atol_spec;
    }
    state.rtol_vec[pc::NumSpec] = rtol_energy;
    state.atol_vec[pc::NumSpec] = atol_energy;
    state.rtol = rtol_spec;
    state.atol = atol_spec;
    state.max_steps = 10000000;
}

integrators::IntegratorResult burn_ros2s(pc::burn_t& state, integrators::Real dt,
                                         IntegratorStats& stats) {
    using Integrator = integrators::RODAS<pc::PrimordialChem, true>;

    pc::eos_rt(state);

    Integrator integrator;
    Integrator::State ros2s_state;
    configure_ros2s(ros2s_state);

    ros2s_state.t = 0.0;
    ros2s_state.tout = dt;
    ros2s_state.dt = dt;
    ros2s_state.jacobian_analytic = true;
    ros2s_state.autonomous = true;
    for (int n = 0; n < pc::NumSpec; ++n) {
        ros2s_state.y[static_cast<std::size_t>(n)] = state.xn[static_cast<std::size_t>(n)];
    }
    ros2s_state.y[pc::NumSpec] = state.e;

    pc::PrimordialChem::state_type problem_state{};
    const auto result = integrator.integrate(problem_state, ros2s_state);

    stats.internal_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_step));
    stats.rhs_calls += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_rhs));
    stats.jacobian_calls += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_jac));
    stats.decompositions += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_decomp));
    stats.linear_solves += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_solve));
    stats.accepted_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_accept));
    stats.rejected_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_reject));

    if (result != integrators::IntegratorResult::SUCCESS) {
        state.success = false;
        return result;
    }

    for (int n = 0; n < pc::NumSpec; ++n) {
        state.xn[static_cast<std::size_t>(n)] = ros2s_state.y[static_cast<std::size_t>(n)];
    }
    state.e = ros2s_state.y[pc::NumSpec];
    state.success = true;
    return result;
}

bool advance_collapse_step(CollapseState& collapse, int step,
                           integrators::IntegratorResult& failure) {
    const integrators::Real old_density = collapse.density_driver;
    const integrators::Real rho = pc::density(collapse.current.xn);
    const integrators::Real tff = std::sqrt(pc::pi * 3.0 / (32.0 * rho * pc::grav_constant));
    const integrators::Real dt = tff_reduc * tff;

    collapse.density_driver += dt * (collapse.density_driver / tff);
    if (dt < 10.0 || collapse.density_driver > 2.0e-6) {
        return true;
    }

    const integrators::Real density_ratio = collapse.density_driver / old_density;
    for (auto& xn : collapse.current.xn) {
        xn *= density_ratio;
    }
    collapse.current.rho *= density_ratio;

    const auto result = burn_ros2s(collapse.current, dt, collapse.stats);
    if (result != integrators::IntegratorResult::SUCCESS) {
        std::cerr << "ROS2S failed on collapse step " << step
                  << " with code " << static_cast<int>(result) << "\n";
        failure = result;
        return true;
    }

    pc::floor_and_normalize_number_densities(collapse.current);
    pc::balance_charge(collapse.current);
    pc::floor_and_normalize_number_densities(collapse.current);
    pc::eos_re(collapse.current);

    collapse.time += dt;
    collapse.completed_steps += 1;
    return false;
}

void print_state(const pc::burn_t& state) {
    std::cout << "rho: " << state.rho << "\n";
    std::cout << "T: " << state.T << "\n";
    std::cout << "Eint: " << state.e << "\n";
    for (int n = 0; n < pc::NumSpec; ++n) {
        std::cout << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                  << state.xn[static_cast<std::size_t>(n)] << "\n";
    }
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10);

    pc::set_redshift(30.0);
    CollapseState collapse = make_collapse_state();
    integrators::IntegratorResult failure = integrators::IntegratorResult::SUCCESS;

    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < max_collapse_steps; ++step) {
        if (advance_collapse_step(collapse, step, failure)) {
            break;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    if (failure != integrators::IntegratorResult::SUCCESS) {
        return 1;
    }

    std::cout << "Primordial chemistry one-zone collapse with ROS2S\n";
    std::cout << "completed collapse steps: " << collapse.completed_steps << "\n";
    std::cout << "physical time: " << collapse.time << "\n";
    std::cout << "density driver: " << collapse.density_driver << "\n";
    std::cout << "wall time: " << elapsed << " s\n";
    std::cout << "ROS2S internal steps: " << collapse.stats.internal_steps << "\n";
    std::cout << "ROS2S rhs calls: " << collapse.stats.rhs_calls << "\n";
    std::cout << "ROS2S jacobian calls: " << collapse.stats.jacobian_calls << "\n";
    std::cout << "ROS2S decompositions: " << collapse.stats.decompositions << "\n";
    std::cout << "ROS2S linear solves: " << collapse.stats.linear_solves << "\n";
    std::cout << "ROS2S accepted/rejected: " << collapse.stats.accepted_steps
              << "/" << collapse.stats.rejected_steps << "\n\n";
    print_state(collapse.current);
    return 0;
}
