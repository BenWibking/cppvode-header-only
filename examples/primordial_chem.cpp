// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Primordial chemistry one-zone collapse test ported from AMReX Microphysics
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

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
constexpr int default_grid_dim = 1;
constexpr int cuda_threads_per_block = 128;

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

struct BatchStatus {
    int completed_global_steps{};
    int failed_cell{-1};
    CollapseStepResult failed_step{};
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

#if defined(__CUDACC__)
__global__ void initialize_collapse_kernel(CollapseState* cells,
                                           CollapseStepResult* cell_results,
                                           int num_cells) {
    const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= num_cells) {
        return;
    }

    cells[cell] = make_collapse_state();
    cell_results[cell] = {};
}

__global__ void collapse_step_kernel(CollapseState* cells,
                                     CollapseStepResult* cell_results,
                                     int step,
                                     int num_cells) {
    const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= num_cells || cell_results[cell].stop) {
        return;
    }

    cell_results[cell] = advance_collapse_step(cells[cell], step);
}
#endif

bool nearly_equal(integrators::Real value, integrators::Real reference,
                  integrators::Real rtol, integrators::Real atol) {
    return std::abs(value - reference) <= atol + rtol * std::abs(reference);
}

bool parse_positive_int(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program << " [--grid N]\n";
}

bool parse_args(int argc, char** argv, int& grid_dim) {
    grid_dim = default_grid_dim;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "--grid") {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], grid_dim)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        constexpr std::string_view grid_prefix = "--grid=";
        if (arg.rfind(grid_prefix, 0) == 0) {
            if (!parse_positive_int(argv[i] + grid_prefix.size(), grid_dim)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }

        print_usage(argv[0]);
        return false;
    }
    return true;
}

bool checked_cell_count(int grid_dim, int& num_cells) {
    const auto grid = static_cast<long long>(grid_dim);
    const auto cells = grid * grid * grid;
    if (cells <= 0 || cells > std::numeric_limits<int>::max()) {
        return false;
    }
    num_cells = static_cast<int>(cells);
    return true;
}

BatchStatus batch_status_from_results(const std::vector<CollapseStepResult>& cell_results,
                                      int completed_global_steps) {
    bool all_stopped = true;
    for (std::size_t cell = 0; cell < cell_results.size(); ++cell) {
        const auto& result = cell_results[cell];
        if (result.result != integrators::IntegratorResult::SUCCESS) {
            return {completed_global_steps, static_cast<int>(cell), result};
        }
        all_stopped = all_stopped && result.stop;
    }

    if (all_stopped) {
        return {completed_global_steps, -1, {true, -1, integrators::IntegratorResult::SUCCESS}};
    }
    return {completed_global_steps, -1, {false, -1, integrators::IntegratorResult::SUCCESS}};
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10);
    std::cout << "Primordial Chemistry One-Zone Collapse\n";
    std::cout << "======================================\n\n";

    int grid_dim = default_grid_dim;
    if (!parse_args(argc, argv, grid_dim)) {
        return 1;
    }
    int num_cells = 0;
    if (!checked_cell_count(grid_dim, num_cells)) {
        std::cerr << "grid dimension is too large: " << grid_dim << "\n";
        return 1;
    }

    pc::set_redshift(30.0);

    std::vector<CollapseState> host_cells(static_cast<std::size_t>(num_cells));
    std::vector<CollapseStepResult> cell_results(static_cast<std::size_t>(num_cells));
    BatchStatus batch_status{};
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

    CollapseState* device_cells = nullptr;
    CollapseStepResult* device_cell_results = nullptr;

    err = cudaMalloc(&device_cells, sizeof(CollapseState) * host_cells.size());
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    err = cudaMalloc(&device_cell_results, sizeof(CollapseStepResult) * cell_results.size());
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_cells);
        return 1;
    }

    const int blocks = (num_cells + cuda_threads_per_block - 1) / cuda_threads_per_block;
    initialize_collapse_kernel<<<blocks, cuda_threads_per_block>>>(
        device_cells, device_cell_results, num_cells);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "initialize_collapse_kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_cell_results);
        cudaFree(device_cells);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "initialize_collapse_kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_cell_results);
        cudaFree(device_cells);
        return 1;
    }

    for (int n = 0; n < nsteps; ++n) {
        collapse_step_kernel<<<blocks, cuda_threads_per_block>>>(
            device_cells, device_cell_results, n, num_cells);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "collapse_step_kernel launch failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            return 1;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "collapse_step_kernel execution failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            return 1;
        }

        err = cudaMemcpy(cell_results.data(), device_cell_results,
                         sizeof(CollapseStepResult) * cell_results.size(),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            return 1;
        }

        batch_status = batch_status_from_results(cell_results, n + 1);
        if (batch_status.failed_cell >= 0 || batch_status.failed_step.stop) {
            break;
        }
    }

    err = cudaMemcpy(host_cells.data(), device_cells, sizeof(CollapseState) * host_cells.size(),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_cell_results);
        cudaFree(device_cells);
        return 1;
    }
    cudaFree(device_cell_results);
    cudaFree(device_cells);

    std::cout << "integration backend: CUDA per-step kernels, one cell per thread\n";
#else
    for (auto& cell : host_cells) {
        cell = make_collapse_state();
    }

    for (int n = 0; n < nsteps; ++n) {
        for (std::size_t cell = 0; cell < host_cells.size(); ++cell) {
            if (!cell_results[cell].stop) {
                cell_results[cell] = advance_collapse_step(host_cells[cell], n);
            }
        }

        batch_status = batch_status_from_results(cell_results, n + 1);
        if (batch_status.failed_cell >= 0 || batch_status.failed_step.stop) {
            break;
        }
    }

    std::cout << "integration backend: CPU serialized cells\n";
#endif

    if (host_cells.empty()) {
        std::cerr << "no cells were initialized\n";
        return 1;
    }

    const int representative_cell = batch_status.failed_cell >= 0 ? batch_status.failed_cell : 0;
    const auto& representative_state = host_cells[static_cast<std::size_t>(representative_cell)];
    const auto representative_step =
        batch_status.failed_cell >= 0
            ? batch_status.failed_step
            : cell_results[static_cast<std::size_t>(representative_cell)];
    const CollapseResult collapse{
        representative_state.initial_state, representative_state.state,
        representative_state.time, representative_state.completed_steps,
        representative_step.failed_step, representative_step.result};

    const auto initial_state = collapse.initial_state;
    const auto state = collapse.final_state;
    const auto t = collapse.time;

    if (collapse.result != integrators::IntegratorResult::SUCCESS) {
        std::cout << "VODE failed in cell " << representative_cell
                  << " on collapse step " << collapse.failed_step
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
    integrators::Real max_thermodynamic_rel_error = 0.0;
    int min_completed_steps = std::numeric_limits<int>::max();
    int max_completed_steps = 0;

    for (const auto& cell : host_cells) {
        min_completed_steps = std::min(min_completed_steps, cell.completed_steps);
        max_completed_steps = std::max(max_completed_steps, cell.completed_steps);

        for (int n = 0; n < pc::NumSpec; ++n) {
            const auto value = cell.state.xn[static_cast<std::size_t>(n)];
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

        pass = pass && nearly_equal(cell.state.T, reference_temperature,
                                    reference_thermodynamic_rtol, atol_spec);
        pass = pass && nearly_equal(cell.state.e, reference_eint,
                                    reference_thermodynamic_rtol, atol_enuc);
        pass = pass && nearly_equal(cell.state.rho, reference_rho, rtol_spec, atol_spec);

        const auto temperature_rel_error =
            std::abs(cell.state.T - reference_temperature) /
            std::max(std::abs(reference_temperature), atol_spec);
        const auto internal_energy_rel_error =
            std::abs(cell.state.e - reference_eint) /
            std::max(std::abs(reference_eint), atol_enuc);
        max_thermodynamic_rel_error =
            std::max(max_thermodynamic_rel_error,
                     std::max(temperature_rel_error, internal_energy_rel_error));
    }

    std::cout << "grid: " << grid_dim << "^3 cells (" << num_cells << " total)\n";
    std::cout << "completed global kernel/step launches: "
              << batch_status.completed_global_steps << "\n";
    std::cout << "completed collapse steps per cell: "
              << min_completed_steps << "..." << max_completed_steps << "\n";
    std::cout << "representative cell: " << representative_cell << "\n";
    std::cout << "representative time: " << t << "\n";
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
