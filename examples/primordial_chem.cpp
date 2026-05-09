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

INTEGRATORS_HOST_DEVICE CollapseResult run_collapse() {
    pc::burn_t state = make_initial_state();
    const auto initial_state = state;
    integrators::Real t = 0.0;
    integrators::Real dd = state.rho;
    int completed_steps = 0;

    for (int n = 0; n < nsteps; ++n) {
        const integrators::Real dd1 = dd;
        const integrators::Real rhotmp = pc::density(state.xn);
        const integrators::Real tff = std::sqrt(pc::pi * 3.0 / (32.0 * rhotmp * pc::grav_constant));
        const integrators::Real dt = tff_reduc * tff;

        dd += dt * (dd / tff);

        if (dt < 10.0 || dd > 2.0e-6) {
            break;
        }

        const integrators::Real density_ratio = dd / dd1;
        for (auto& xn : state.xn) {
            xn *= density_ratio;
        }
        state.rho *= density_ratio;

        const auto result = burn(state, dt);
        if (result != integrators::IntegratorResult::SUCCESS) {
            return {initial_state, state, t, completed_steps, n, result};
        }

        pc::floor_and_normalize_number_densities(state);
        pc::balance_charge(state);
        pc::floor_and_normalize_number_densities(state);
        pc::eos_re(state);

        t += dt;
        completed_steps += 1;
    }

    return {initial_state, state, t, completed_steps, -1, integrators::IntegratorResult::SUCCESS};
}

#if defined(__CUDACC__)
__global__ void run_collapse_kernel(CollapseResult* result) {
    *result = run_collapse();
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

    CollapseResult* device_result = nullptr;
    err = cudaMalloc(&device_result, sizeof(CollapseResult));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    run_collapse_kernel<<<1, 1>>>(device_result);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "run_collapse_kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_result);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "run_collapse_kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_result);
        return 1;
    }

    err = cudaMemcpy(&collapse, device_result, sizeof(CollapseResult), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_result);
        return 1;
    }
    cudaFree(device_result);
    std::cout << "integration backend: CUDA device kernel\n";
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
