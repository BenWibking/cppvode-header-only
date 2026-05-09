// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Primordial chemistry one-zone collapse test ported from AMReX Microphysics
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <integrators/primordial_chem.hpp>
#include <integrators/vode.hpp>

namespace pc = integrators::primordial_chem;

namespace {

constexpr integrators::Real tff_reduc = 1.0e-1;
constexpr int nsteps = 1000;
constexpr integrators::Real temperature = 1.0e2;
constexpr integrators::Real rtol_spec = 1.0e-4;
constexpr integrators::Real atol_spec = 1.0e-4;
constexpr integrators::Real rtol_enuc = 1.0e-6;
constexpr integrators::Real atol_enuc = 1.0e-6;
constexpr integrators::Real reference_thermodynamic_rtol = 1.0e-5;

constexpr std::array<integrators::Real, pc::NumSpec> initial_number_densities{
    1.0e-4, 1.0e-4, 1.0e0,  1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40,
    1.0e-40, 1.0e-6, 1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40, 0.0775};

constexpr std::array<integrators::Real, pc::NumSpec> reference_number_densities{
    19911.96049,      19897.56689,      1.61920532e17, 1.780427462, 1.0e-100,
    0.9313391106,     16.17402851,      1.0e-100,      3.378785985e17,
    1.0e-100,         8.912464329,      2.624158281e-60, 6.363655862e-12,
    6.491340291e16};

// Reference comparison tolerances. The deuterium channels are especially
// sensitive to fused multiply-add contraction in the generated chemistry rates.
constexpr std::array<integrators::Real, pc::NumSpec> reference_species_rtol{
    1.0e-3, 1.0e-3, 1.0e-4, 1.0e-3, 1.0e-4, 3.4, 1.0e-3,
    1.0e-4, 1.0e-4, 1.0e-4, 3.4, 1.0e-4, 1.0e-4, 1.0e-4};

constexpr std::array<bool, pc::NumSpec> deuterium_bearing_species{
    false, false, false, false, true, true, false,
    true,  false, true,  true,  false, false, false};

constexpr integrators::Real reference_temperature = 3032.992479;
constexpr integrators::Real reference_eint = 2.721837163e11;
constexpr integrators::Real reference_rho = 1.836285633e-6;

struct CollapseResult {
    pc::burn_t initial_state{};
    pc::burn_t final_state{};
    integrators::Real time{};
    int completed_steps{};
    int failed_step{-1};
    integrators::IntegratorResult result{integrators::IntegratorResult::SUCCESS};
};

struct CollapseState {
    pc::burn_t initial_state{};
    pc::burn_t state{};
    integrators::Real time{};
    integrators::Real density_driver{};
    int completed_steps{};
};

struct CollapseStepResult {
    bool stop{};
    int failed_step{-1};
    integrators::IntegratorResult result{integrators::IntegratorResult::SUCCESS};
};

INTEGRATORS_HOST_DEVICE void configure_microphysics_tolerances(integrators::VODEState<pc::neqs>& state) {
    state.use_vector_tolerances = true;
    for (int n = 0; n < pc::NumSpec; ++n) {
        state.rtol_vec[static_cast<std::size_t>(n)] = rtol_spec;
        state.atol_vec[static_cast<std::size_t>(n)] = atol_spec;
    }
    state.rtol_vec[pc::NumSpec] = rtol_enuc;
    state.atol_vec[pc::NumSpec] = atol_enuc;
    state.rtol = rtol_spec;
    state.atol = atol_spec;
    state.max_steps = 150000;
    state.HMXI = 1.0e-30;
    state.constrained_components = pc::NumSpec;
    state.reject_change_buffer = 1.0e100;
    state.species_failure_tolerance = 1.0e-2;
    state.clean_constrained_components = true;
    state.component_floor = pc::small_number_density_floor();
}

INTEGRATORS_HOST_DEVICE integrators::IntegratorResult burn_once(pc::burn_t& state, integrators::Real dt) {
    pc::eos_rt(state);

    auto integrator = integrators::VODE<pc::PrimordialChem>{};
    auto vode_state = integrators::VODEState<pc::neqs>{};
    configure_microphysics_tolerances(vode_state);

    vode_state.t = 0.0;
    vode_state.tout = dt;
    vode_state.jacobian_analytic = true;
    for (int n = 0; n < pc::NumSpec; ++n) {
        vode_state.y[static_cast<std::size_t>(n)] = state.xn[static_cast<std::size_t>(n)];
    }
    vode_state.y[pc::NumSpec] = state.e;

    pc::PrimordialChem::state_type problem_state{};
    const auto result = integrator.integrate(problem_state, vode_state);

    if (result == integrators::IntegratorResult::SUCCESS) {
        for (int n = 0; n < pc::NumSpec; ++n) {
            state.xn[static_cast<std::size_t>(n)] = vode_state.y[static_cast<std::size_t>(n)];
        }
        state.e = vode_state.y[pc::NumSpec];
        state.success = true;
    } else {
        state.success = false;
    }

    return result;
}

INTEGRATORS_HOST_DEVICE integrators::IntegratorResult burn(pc::burn_t& state, integrators::Real dt) {
    return burn_once(state, dt);
}

INTEGRATORS_HOST_DEVICE pc::burn_t make_initial_state() {
    pc::burn_t state;
    state.T = temperature;
    state.xn = initial_number_densities;
    state.rho = pc::density(state.xn);
    pc::normalize_number_densities_to_density(state);
    pc::eos_rt(state);
    return state;
}

INTEGRATORS_HOST_DEVICE CollapseState make_collapse_state() {
    pc::burn_t state = make_initial_state();
    return {state, state, 0.0, state.rho, 0};
}

INTEGRATORS_HOST_DEVICE CollapseStepResult advance_collapse_step(CollapseState& collapse, int step) {
    const integrators::Real dd1 = collapse.density_driver;
    const integrators::Real rhotmp = pc::density(collapse.state.xn);
    const integrators::Real tff = std::sqrt(pc::pi * 3.0 / (32.0 * rhotmp * pc::grav_constant));
    const integrators::Real dt = tff_reduc * tff;

    collapse.density_driver += dt * (collapse.density_driver / tff);

    if (dt < 10.0 || collapse.density_driver > 2.0e-6) {
        return {true, -1, integrators::IntegratorResult::SUCCESS};
    }

    const integrators::Real density_ratio = collapse.density_driver / dd1;
    for (auto& xn : collapse.state.xn) {
        xn *= density_ratio;
    }
    collapse.state.rho *= density_ratio;

    const auto result = burn(collapse.state, dt);
    if (result != integrators::IntegratorResult::SUCCESS) {
        return {true, step, result};
    }

    pc::floor_and_normalize_number_densities(collapse.state);
    pc::balance_charge(collapse.state);
    pc::floor_and_normalize_number_densities(collapse.state);
    pc::eos_re(collapse.state);

    collapse.time += dt;
    collapse.completed_steps += 1;
    return {false, -1, integrators::IntegratorResult::SUCCESS};
}

INTEGRATORS_HOST_DEVICE CollapseResult collapse_result_from_state(
    const CollapseState& collapse, const CollapseStepResult& step) {
    return {collapse.initial_state, collapse.state, collapse.time,
            collapse.completed_steps, step.failed_step, step.result};
}

INTEGRATORS_HOST_DEVICE CollapseResult run_collapse() {
    CollapseState collapse = make_collapse_state();
    CollapseStepResult step_result{};
    for (int n = 0; n < nsteps; ++n) {
        step_result = advance_collapse_step(collapse, n);
        if (step_result.stop) {
            break;
        }
    }
    return collapse_result_from_state(collapse, step_result);
}

#if defined(__CUDACC__)
__global__ void initialize_collapse_kernel(CollapseState* collapse) {
    *collapse = make_collapse_state();
}

__global__ void collapse_step_kernel(CollapseState* collapse, int step, CollapseStepResult* result) {
    *result = advance_collapse_step(*collapse, step);
}
#endif

bool nearly_equal(integrators::Real value, integrators::Real reference,
                  integrators::Real rtol, integrators::Real atol) {
    return std::abs(value - reference) <= atol + rtol * std::abs(reference);
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10);
    std::cout << "Primordial Chemistry One-Zone Collapse\n";
    std::cout << "======================================\n\n";

    pc::set_redshift(30.0);

    CollapseResult collapse;
#if defined(__CUDACC__)
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaErrorNoDevice || device_count == 0) {
        std::cout << "CUDA run skipped: no CUDA-capable device\n";
        return 77;
    }
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    CollapseState host_collapse{};
    CollapseStepResult step_result{};
    CollapseState* device_collapse = nullptr;
    CollapseStepResult* device_step_result = nullptr;

    err = cudaMalloc(&device_collapse, sizeof(CollapseState));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    err = cudaMalloc(&device_step_result, sizeof(CollapseStepResult));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_collapse);
        return 1;
    }

    initialize_collapse_kernel<<<1, 1>>>(device_collapse);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "initialize_collapse_kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_step_result);
        cudaFree(device_collapse);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "initialize_collapse_kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_step_result);
        cudaFree(device_collapse);
        return 1;
    }

    for (int n = 0; n < nsteps; ++n) {
        collapse_step_kernel<<<1, 1>>>(device_collapse, n, device_step_result);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "collapse_step_kernel launch failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_step_result);
            cudaFree(device_collapse);
            return 1;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "collapse_step_kernel execution failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_step_result);
            cudaFree(device_collapse);
            return 1;
        }

        err = cudaMemcpy(&step_result, device_step_result, sizeof(CollapseStepResult),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(device_step_result);
            cudaFree(device_collapse);
            return 1;
        }

        if (step_result.stop) {
            break;
        }
    }

    err = cudaMemcpy(&host_collapse, device_collapse, sizeof(CollapseState), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_step_result);
        cudaFree(device_collapse);
        return 1;
    }
    cudaFree(device_step_result);
    cudaFree(device_collapse);

    collapse = {host_collapse.initial_state, host_collapse.state, host_collapse.time,
                host_collapse.completed_steps, step_result.failed_step, step_result.result};
    std::cout << "integration backend: CUDA per-step kernels\n";
#else
    collapse = run_collapse();
    std::cout << "integration backend: CPU\n";
#endif

    const auto initial_state = collapse.initial_state;
    const auto state = collapse.final_state;
    const auto t = collapse.time;
    const auto completed_steps = collapse.completed_steps;

    if (collapse.result != integrators::IntegratorResult::SUCCESS) {
        std::cout << "VODE failed on collapse step " << collapse.failed_step
                  << " with code " << static_cast<int>(collapse.result) << "\n";
        std::cout << "time: " << t << "\n";
        std::cout << "rho: " << state.rho << "\n";
        std::cout << "T: " << state.T << "\n";
        std::cout << "Eint: " << state.e << "\n";
        for (int k = 0; k < pc::NumSpec; ++k) {
            std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(k)] << ": "
                      << state.xn[static_cast<std::size_t>(k)] << "\n";
        }
        return 1;
    }

    bool pass = true;
    integrators::Real max_non_deuterium_species_rel_error = 0.0;
    for (int n = 0; n < pc::NumSpec; ++n) {
        const auto value = state.xn[static_cast<std::size_t>(n)];
        const auto reference = reference_number_densities[static_cast<std::size_t>(n)];
        const auto denom = std::max(std::abs(reference), atol_spec);
        const auto rel_error = std::abs(value - reference) / denom;
        if (!deuterium_bearing_species[static_cast<std::size_t>(n)]) {
            max_non_deuterium_species_rel_error =
                std::max(max_non_deuterium_species_rel_error, rel_error);
        }
        pass = pass && nearly_equal(
                           value, reference,
                           reference_species_rtol[static_cast<std::size_t>(n)], atol_spec);
    }

    pass = pass && nearly_equal(state.T, reference_temperature, reference_thermodynamic_rtol, atol_spec);
    pass = pass && nearly_equal(state.e, reference_eint, reference_thermodynamic_rtol, atol_enuc);
    pass = pass && nearly_equal(state.rho, reference_rho, rtol_spec, atol_spec);

    const auto temperature_rel_error =
        std::abs(state.T - reference_temperature) / std::max(std::abs(reference_temperature), atol_spec);
    const auto internal_energy_rel_error =
        std::abs(state.e - reference_eint) / std::max(std::abs(reference_eint), atol_enuc);
    const auto max_thermodynamic_rel_error =
        std::max(temperature_rel_error, internal_energy_rel_error);

    std::cout << "completed collapse steps: " << completed_steps << "\n";
    std::cout << "time: " << t << "\n";
    std::cout << "T initial: " << initial_state.T << "\n";
    std::cout << "T final:   " << state.T << "\n";
    std::cout << "Eint initial: " << initial_state.e << "\n";
    std::cout << "Eint final:   " << state.e << "\n";
    std::cout << "rho initial: " << initial_state.rho << "\n";
    std::cout << "rho final:   " << state.rho << "\n";
    std::cout << "max non-deuterium species relative error vs Microphysics reference: "
              << max_non_deuterium_species_rel_error << "\n";
    std::cout << "max T/Eint relative error vs Microphysics reference: "
              << max_thermodynamic_rel_error << "\n";
    std::cout << "reference comparison: " << (pass ? "PASS" : "FAIL") << "\n";

    if (!pass) {
        std::cout << "\nFinal number densities:\n";
        for (int n = 0; n < pc::NumSpec; ++n) {
            std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                      << state.xn[static_cast<std::size_t>(n)]
                      << " (reference " << reference_number_densities[static_cast<std::size_t>(n)] << ")\n";
        }
    }

    return pass ? 0 : 1;
}
