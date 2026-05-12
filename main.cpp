// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#if defined(PRIMORDIAL_ROS2S_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#include "primordial_chem.hpp"
#include "rodas.hpp"

namespace pc = integrators::primordial_chem;

namespace {

#if defined(PRIMORDIAL_ROS2S_ENABLE_CUDA)
#define PRIMORDIAL_HOST_DEVICE __host__ __device__ __forceinline__
#ifndef PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK
#define PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK 128
#endif
#else
#define PRIMORDIAL_HOST_DEVICE inline
#endif

constexpr integrators::Real tff_reduc = 1.0e-1;
constexpr int max_collapse_steps = 1000;
constexpr integrators::Real initial_temperature = 1.0e2;
constexpr integrators::Real rtol_spec = 1.0e-4;
constexpr integrators::Real atol_spec = 1.0e-4;
constexpr integrators::Real rtol_energy = 1.0e-6;
constexpr integrators::Real atol_energy = 1.0e-6;
constexpr integrators::Real comparison_thermodynamic_rtol = 1.0e-4;
constexpr int default_grid_dim = 1;
constexpr int perturbation_interval = 20;
constexpr integrators::Real perturbation_amplitude = 0.2;
constexpr int backup_suffix_digits = 6;
constexpr int backup_suffix_limit = 1000000;
constexpr int backup_rename_attempts = 100;

constexpr std::array<integrators::Real, pc::NumSpec> initial_number_densities{
    1.0e-4, 1.0e-4, 1.0e0,  1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40,
    1.0e-40, 1.0e-6, 1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40, 0.0775};

// Match the PASS criteria used by the cusolverdx branch for this network.
constexpr std::array<integrators::Real, pc::NumSpec> comparison_species_rtol{
    1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-4, 10.0, 1.0e-3,
    1.0e-4, 1.0e-4, 1.0e-4, 10.0, 1.0e-4, 1.0e-4, 1.0e-4};

struct IntegratorStats {
    std::uint64_t internal_steps{};
    std::uint64_t rhs_calls{};
    std::uint64_t jacobian_calls{};
    std::uint64_t decompositions{};
    std::uint64_t linear_solves{};
    std::uint64_t accepted_steps{};
    std::uint64_t rejected_steps{};
};

struct CounterSummary {
    std::uint64_t min{};
    long double median{};
    std::uint64_t max{};
};

struct ValueSummary {
    long double min{};
    long double median{};
    long double max{};
};

struct CollapseState {
    pc::burn_t current{};
    integrators::Real time{};
    integrators::Real density_driver{};
    int completed_steps{};
    IntegratorStats stats{};
};

struct Options {
    int grid_dim{default_grid_dim};
    bool perturb{false};
    bool show_help{false};
    std::string compare_final_state_path{};
};

using Ros2sIntegrator = integrators::RODAS<pc::PrimordialChem>;

constexpr const char* backend_name() {
#if defined(__CUDACC__)
    return "cuda";
#elif defined(__HIPCC__)
    return "hip";
#else
    return "cpu";
#endif
}

#pragma pack(push, 1)
struct PackedFinalState {
    std::int32_t cell{};
    std::int32_t i{};
    std::int32_t j{};
    std::int32_t k{};
    std::int32_t completed_steps{};
    integrators::Real time{};
    integrators::Real density_driver{};
    integrators::Real rho{};
    integrators::Real T{};
    integrators::Real e{};
    integrators::Real xn[pc::NumSpec]{};
};
#pragma pack(pop)

static_assert(sizeof(PackedFinalState) ==
              5 * sizeof(std::int32_t) + (5 + pc::NumSpec) * sizeof(integrators::Real));

struct ComparisonSummary {
    bool pass{true};
    long double max_deuterium_species_rel_error{};
    long double max_non_deuterium_species_rel_error{};
    long double max_thermodynamic_rel_error{};
    long double max_rho_rel_error{};
    std::string failure_message{};
};

constexpr bool is_deuterium_bearing_species(int n) {
    return n == 4 || n == 5 || n == 7 || n == 9 || n == 10;
}

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
    return {state, 0.0, state.rho, 0, {}};
}

PRIMORDIAL_HOST_DEVICE std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

PRIMORDIAL_HOST_DEVICE integrators::Real perturbation_factor(int cell, int step) {
    const auto seed = (static_cast<std::uint64_t>(static_cast<unsigned int>(cell)) << 32U) ^
                      (static_cast<std::uint64_t>(static_cast<unsigned int>(step)) << 16U);
    const auto bits = splitmix64(seed) >> 11U;
    const auto unit = static_cast<integrators::Real>(bits) *
                      (1.0 / static_cast<integrators::Real>(1ULL << 53U));
    return 1.0 + perturbation_amplitude * (2.0 * unit - 1.0);
}

PRIMORDIAL_HOST_DEVICE void apply_perturbation(CollapseState& collapse, int cell, int step, bool enabled) {
    if (!enabled || step == 0 || step % perturbation_interval != 0) {
        return;
    }

    const integrators::Real factor = perturbation_factor(cell, step);
    collapse.density_driver *= factor;
    collapse.current.rho *= factor;
    for (auto& xn : collapse.current.xn) {
        xn *= factor;
    }

    pc::floor_and_normalize_number_densities(collapse.current);
    pc::balance_charge(collapse.current);
    pc::floor_and_normalize_number_densities(collapse.current);
    pc::eos_re(collapse.current);
}

PRIMORDIAL_HOST_DEVICE void configure_ros2s(Ros2sIntegrator::State& state) {
    for (int n = 0; n < pc::NumSpec; ++n) {
        state.rtol_vec[static_cast<std::size_t>(n)] = rtol_spec;
        state.atol_vec[static_cast<std::size_t>(n)] = atol_spec;
    }
    state.rtol_vec[pc::NumSpec] = rtol_energy;
    state.atol_vec[pc::NumSpec] = atol_energy;
    state.max_steps = 10000000;
}

PRIMORDIAL_HOST_DEVICE integrators::IntegratorResult burn_ros2s(pc::burn_t& state, integrators::Real dt,
                                                                IntegratorStats& stats) {
    pc::eos_rt(state);

    Ros2sIntegrator integrator;
    Ros2sIntegrator::State ros2s_state;
    configure_ros2s(ros2s_state);

    ros2s_state.t = 0.0;
    ros2s_state.tout = dt;
    ros2s_state.dt = dt;
    for (int n = 0; n < pc::NumSpec; ++n) {
        ros2s_state.y[static_cast<std::size_t>(n)] = state.xn[static_cast<std::size_t>(n)];
    }
    ros2s_state.y[pc::NumSpec] = state.e;

    const auto result = integrator.integrate(ros2s_state);

    stats.internal_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_step));
    stats.rhs_calls += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_rhs));
    stats.jacobian_calls += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_jac));
    stats.decompositions += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_decomp));
    stats.linear_solves += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_solve));
    stats.accepted_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_accept));
    stats.rejected_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_reject));

    if (result != integrators::IntegratorResult::SUCCESS) {
        return result;
    }

    for (int n = 0; n < pc::NumSpec; ++n) {
        state.xn[static_cast<std::size_t>(n)] = ros2s_state.y[static_cast<std::size_t>(n)];
    }
    state.e = ros2s_state.y[pc::NumSpec];
    return result;
}

PRIMORDIAL_HOST_DEVICE bool advance_collapse_step(CollapseState& collapse, int cell, int step, bool perturb,
                                                  integrators::IntegratorResult& failure) {
    apply_perturbation(collapse, cell, step, perturb);

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
#if !defined(__CUDA_ARCH__)
        std::cerr << "ROS2S failed on collapse step " << step
                  << " cell " << cell
                  << " with code " << static_cast<int>(result) << "\n";
#endif
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

#if defined(PRIMORDIAL_ROS2S_ENABLE_CUDA)
bool check_cuda(cudaError_t status, const char* action) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << action << " failed: " << cudaGetErrorString(status) << "\n";
    return false;
}

__global__ void advance_collapse_grid_kernel(CollapseState* cells, int num_cells,
                                             int completed_global_steps, int step,
                                             bool perturb, int* all_stopped,
                                             int* failure_code) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells || *failure_code != static_cast<int>(integrators::IntegratorResult::SUCCESS)) {
        return;
    }

    CollapseState& state = cells[cell];
    if (state.completed_steps < completed_global_steps) {
        return;
    }

    auto failure = integrators::IntegratorResult::SUCCESS;
    const bool stopped = advance_collapse_step(state, cell, step, perturb, failure);
    if (failure != integrators::IntegratorResult::SUCCESS) {
        atomicCAS(failure_code, static_cast<int>(integrators::IntegratorResult::SUCCESS),
                  static_cast<int>(failure));
    }
    if (!stopped) {
        atomicExch(all_stopped, 0);
    }
}

integrators::IntegratorResult run_cells_cuda(std::vector<CollapseState>& cells,
                                             bool perturb,
                                             int& completed_global_steps) {
    CollapseState* device_cells = nullptr;
    int* device_all_stopped = nullptr;
    int* device_failure = nullptr;
    const auto bytes = cells.size() * sizeof(CollapseState);

    if (!check_cuda(cudaMalloc(&device_cells, bytes), "cudaMalloc(cells)") ||
        !check_cuda(cudaMalloc(&device_all_stopped, sizeof(int)), "cudaMalloc(all_stopped)") ||
        !check_cuda(cudaMalloc(&device_failure, sizeof(int)), "cudaMalloc(failure)") ||
        !check_cuda(cudaMemcpy(device_cells, cells.data(), bytes, cudaMemcpyHostToDevice),
                    "cudaMemcpy(cells to device)")) {
        cudaFree(device_cells);
        cudaFree(device_all_stopped);
        cudaFree(device_failure);
        return integrators::IntegratorResult::BAD_INPUTS;
    }

    integrators::IntegratorResult result = integrators::IntegratorResult::SUCCESS;
    const int success = static_cast<int>(integrators::IntegratorResult::SUCCESS);
    if (!check_cuda(cudaMemcpy(device_failure, &success, sizeof(int), cudaMemcpyHostToDevice),
                    "cudaMemcpy(failure to device)")) {
        result = integrators::IntegratorResult::BAD_INPUTS;
    }

    constexpr int block_size = PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK;
    static_assert(block_size > 0, "PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK must be positive");
    static_assert(block_size <= 1024,
                  "PRIMORDIAL_ROS2S_CUDA_THREADS_PER_BLOCK cannot exceed 1024");
    const int num_cells = static_cast<int>(cells.size());
    const int grid_size = (num_cells + block_size - 1) / block_size;

    for (int step = 0; result == integrators::IntegratorResult::SUCCESS &&
                       step < max_collapse_steps; ++step) {
        const int all_stopped = 1;
        if (!check_cuda(cudaMemcpy(device_all_stopped, &all_stopped, sizeof(int),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy(all_stopped to device)")) {
            result = integrators::IntegratorResult::BAD_INPUTS;
            break;
        }

        advance_collapse_grid_kernel<<<grid_size, block_size>>>(
            device_cells, num_cells, completed_global_steps, step, perturb,
            device_all_stopped, device_failure);
        if (!check_cuda(cudaGetLastError(), "advance_collapse_grid_kernel launch") ||
            !check_cuda(cudaDeviceSynchronize(), "advance_collapse_grid_kernel synchronize")) {
            result = integrators::IntegratorResult::BAD_INPUTS;
            break;
        }

        int host_failure = success;
        int host_all_stopped = 0;
        if (!check_cuda(cudaMemcpy(&host_failure, device_failure, sizeof(int),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(failure to host)") ||
            !check_cuda(cudaMemcpy(&host_all_stopped, device_all_stopped, sizeof(int),
                                   cudaMemcpyDeviceToHost),
                        "cudaMemcpy(all_stopped to host)")) {
            result = integrators::IntegratorResult::BAD_INPUTS;
            break;
        }

        if (host_failure != success) {
            result = static_cast<integrators::IntegratorResult>(host_failure);
            std::cerr << "ROS2S failed on CUDA collapse step " << step
                      << " with code " << host_failure << "\n";
            break;
        }
        if (host_all_stopped != 0) {
            break;
        }
        completed_global_steps += 1;
    }

    if (!check_cuda(cudaMemcpy(cells.data(), device_cells, bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy(cells to host)") &&
        result == integrators::IntegratorResult::SUCCESS) {
        result = integrators::IntegratorResult::BAD_INPUTS;
    }

    cudaFree(device_cells);
    cudaFree(device_all_stopped);
    cudaFree(device_failure);
    return result;
}
#endif

void add_stats(IntegratorStats& total, const IntegratorStats& value) {
    total.internal_steps += value.internal_steps;
    total.rhs_calls += value.rhs_calls;
    total.jacobian_calls += value.jacobian_calls;
    total.decompositions += value.decompositions;
    total.linear_solves += value.linear_solves;
    total.accepted_steps += value.accepted_steps;
    total.rejected_steps += value.rejected_steps;
}

CounterSummary summarize_counter(const std::vector<CollapseState>& cells,
                                 std::uint64_t IntegratorStats::* counter) {
    std::vector<std::uint64_t> values;
    values.reserve(cells.size());
    for (const auto& cell : cells) {
        values.push_back(cell.stats.*counter);
    }
    std::sort(values.begin(), values.end());

    const auto size = values.size();
    const long double median =
        (size % 2 == 1)
            ? static_cast<long double>(values[size / 2])
            : (static_cast<long double>(values[size / 2 - 1]) +
               static_cast<long double>(values[size / 2])) /
                  2.0L;

    return {values.front(), median, values.back()};
}

void print_counter_summary(const char* label, const CounterSummary& summary) {
    std::cout << label << ": [" << summary.min << ", " << summary.median
              << ", " << summary.max << "]\n";
}

template<typename Value>
ValueSummary summarize_value(const std::vector<CollapseState>& cells,
                             Value CollapseState::* value) {
    std::vector<long double> values;
    values.reserve(cells.size());
    for (const auto& cell : cells) {
        values.push_back(static_cast<long double>(cell.*value));
    }
    std::sort(values.begin(), values.end());

    const auto size = values.size();
    const long double median =
        (size % 2 == 1) ? values[size / 2] : (values[size / 2 - 1] + values[size / 2]) / 2.0L;

    return {values.front(), median, values.back()};
}

void print_value_summary(const char* label, const ValueSummary& summary) {
    std::cout << label << ": [" << summary.min << ", " << summary.median
              << ", " << summary.max << "]\n";
}

std::string format_scientific(long double value) {
    std::ostringstream out;
    out << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10)
        << std::scientific << value;
    return out.str();
}

bool nearly_equal(integrators::Real value, integrators::Real reference,
                  integrators::Real rtol, integrators::Real atol) {
    return std::abs(value - reference) <= atol + rtol * std::abs(reference);
}

long double relative_error(integrators::Real value, integrators::Real reference,
                           integrators::Real atol) {
    const auto denom = std::max(std::abs(reference), atol);
    return static_cast<long double>(std::abs(value - reference) / denom);
}

void print_scientific_summary(const char* label, const ValueSummary& summary) {
    std::cout << label << ": [" << format_scientific(summary.min) << ", "
              << format_scientific(summary.median) << ", "
              << format_scientific(summary.max) << "]\n";
}

void print_collapse_summary(const std::vector<CollapseState>& cells,
                            const CollapseState& representative) {
    if (cells.size() == 1) {
        std::cout << "representative cell completed steps: "
                  << representative.completed_steps << "\n";
        std::cout << "representative cell physical time: "
                  << format_scientific(representative.time) << "\n";
        std::cout << "representative cell density driver: "
                  << representative.density_driver << "\n";
        return;
    }

    print_value_summary("cell completed steps",
                        summarize_value(cells, &CollapseState::completed_steps));
    print_scientific_summary("cell physical time",
                             summarize_value(cells, &CollapseState::time));
    print_value_summary("cell density driver",
                        summarize_value(cells, &CollapseState::density_driver));
}

void print_stats(const std::vector<CollapseState>& cells,
                 const IntegratorStats& total_stats) {
    if (cells.size() == 1) {
        std::cout << "ROS2S internal steps: " << total_stats.internal_steps << "\n";
        std::cout << "ROS2S rhs calls: " << total_stats.rhs_calls << "\n";
        std::cout << "ROS2S jacobian calls: " << total_stats.jacobian_calls << "\n";
        std::cout << "ROS2S decompositions: " << total_stats.decompositions << "\n";
        std::cout << "ROS2S linear solves: " << total_stats.linear_solves << "\n";
        std::cout << "ROS2S accepted/rejected: " << total_stats.accepted_steps
                  << "/" << total_stats.rejected_steps << "\n\n";
        return;
    }

    print_counter_summary("ROS2S internal steps",
                          summarize_counter(cells, &IntegratorStats::internal_steps));
    print_counter_summary("ROS2S rhs calls",
                          summarize_counter(cells, &IntegratorStats::rhs_calls));
    print_counter_summary("ROS2S jacobian calls",
                          summarize_counter(cells, &IntegratorStats::jacobian_calls));
    print_counter_summary("ROS2S decompositions",
                          summarize_counter(cells, &IntegratorStats::decompositions));
    print_counter_summary("ROS2S linear solves",
                          summarize_counter(cells, &IntegratorStats::linear_solves));

    const auto accepted = summarize_counter(cells, &IntegratorStats::accepted_steps);
    const auto rejected = summarize_counter(cells, &IntegratorStats::rejected_steps);
    std::cout << "ROS2S accepted/rejected: [" << accepted.min << "/" << rejected.min
              << ", " << accepted.median << "/" << rejected.median
              << ", " << accepted.max << "/" << rejected.max << "]\n\n";
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

bool checked_cell_count(int grid_dim, int& num_cells) {
    const auto grid = static_cast<long long>(grid_dim);
    const auto cells = grid * grid * grid;
    if (cells <= 0 || cells > std::numeric_limits<int>::max()) {
        return false;
    }
    num_cells = static_cast<int>(cells);
    return true;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program
              << " [--grid N] [--perturb] [--compare-final-state FILE]\n";
}

bool parse_args(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
            return true;
        }
        if (arg == "--grid") {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], options.grid_dim)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        if (arg == "--perturb") {
            options.perturb = true;
            continue;
        }
        if (arg == "--compare-final-state") {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                return false;
            }
            options.compare_final_state_path = argv[++i];
            continue;
        }
        constexpr std::string_view compare_prefix = "--compare-final-state=";
        if (arg.rfind(compare_prefix, 0) == 0) {
            options.compare_final_state_path = argv[i] + compare_prefix.size();
            if (options.compare_final_state_path.empty()) {
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

void print_state(const pc::burn_t& state) {
    std::cout << "rho: " << state.rho << "\n";
    std::cout << "T: " << state.T << "\n";
    std::cout << "Eint: " << state.e << "\n";
    for (int n = 0; n < pc::NumSpec; ++n) {
        std::cout << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                  << state.xn[static_cast<std::size_t>(n)] << "\n";
    }
}

PackedFinalState make_packed_final_state(const CollapseState& state, int cell, int grid_dim) {
    PackedFinalState packed{};
    packed.cell = static_cast<std::int32_t>(cell);
    packed.i = static_cast<std::int32_t>(cell % grid_dim);
    packed.j = static_cast<std::int32_t>((cell / grid_dim) % grid_dim);
    packed.k = static_cast<std::int32_t>(cell / (grid_dim * grid_dim));
    packed.completed_steps = static_cast<std::int32_t>(state.completed_steps);
    packed.time = state.time;
    packed.density_driver = state.density_driver;
    packed.rho = state.current.rho;
    packed.T = state.current.T;
    packed.e = state.current.e;
    for (int n = 0; n < pc::NumSpec; ++n) {
        packed.xn[static_cast<std::size_t>(n)] = state.current.xn[static_cast<std::size_t>(n)];
    }
    return packed;
}

std::string make_backup_path(const std::string& path, int suffix) {
    std::ostringstream backup;
    backup << path << ".old." << std::setw(backup_suffix_digits) << std::setfill('0')
           << suffix;
    return backup.str();
}

bool rotate_existing_output(const std::string& path) {
    namespace fs = std::filesystem;

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        if (ec) {
            std::cerr << "failed to check final-state output file: " << path
                      << " (" << ec.message() << ")\n";
            return false;
        }
        return true;
    }

    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_int_distribution<int> suffix_dist(0, backup_suffix_limit - 1);

    for (int attempt = 0; attempt < backup_rename_attempts; ++attempt) {
        const std::string backup_path = make_backup_path(path, suffix_dist(generator));
        if (fs::exists(backup_path, ec)) {
            if (ec) {
                std::cerr << "failed to check backup final-state file: " << backup_path
                          << " (" << ec.message() << ")\n";
                return false;
            }
            continue;
        }

        fs::rename(path, backup_path, ec);
        if (!ec) {
            std::cout << "existing final states moved to: " << backup_path << "\n";
            return true;
        }

        std::cerr << "failed to move existing final-state output file from "
                  << path << " to " << backup_path << " (" << ec.message() << ")\n";
        return false;
    }

    std::cerr << "failed to choose a unique backup filename for: " << path << "\n";
    return false;
}

bool write_final_states(const std::vector<CollapseState>& cells, int grid_dim,
                        const std::string& path) {
    if (!rotate_existing_output(path)) {
        return false;
    }

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        std::cerr << "failed to open final-state output file: " << path << "\n";
        return false;
    }

    for (std::size_t cell = 0; cell < cells.size(); ++cell) {
        const auto packed = make_packed_final_state(cells[cell], static_cast<int>(cell), grid_dim);
        output.write(reinterpret_cast<const char*>(&packed), sizeof(packed));
        if (!output) {
            std::cerr << "failed to write final-state output file: " << path << "\n";
            return false;
        }
    }
    return true;
}

bool read_final_states(const std::string& path, std::size_t expected_records,
                       std::vector<PackedFinalState>& records) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        std::cerr << "failed to open final-state comparison file: " << path << "\n";
        return false;
    }

    const auto file_size = input.tellg();
    if (file_size < 0) {
        std::cerr << "failed to determine size of final-state comparison file: "
                  << path << "\n";
        return false;
    }
    const auto byte_count = static_cast<std::uintmax_t>(file_size);
    const auto expected_bytes =
        static_cast<std::uintmax_t>(expected_records) * sizeof(PackedFinalState);
    if (byte_count != expected_bytes) {
        std::cerr << "final-state comparison file has " << byte_count
                  << " bytes, expected " << expected_bytes << " bytes ("
                  << expected_records << " packed records of "
                  << sizeof(PackedFinalState) << " bytes)\n";
        return false;
    }

    records.resize(expected_records);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(records.data()),
               static_cast<std::streamsize>(expected_bytes));
    if (!input && expected_bytes != 0) {
        std::cerr << "failed to read final-state comparison file: " << path << "\n";
        return false;
    }
    return true;
}

ComparisonSummary compare_final_states(const std::vector<CollapseState>& cells,
                                       int grid_dim,
                                       const std::vector<PackedFinalState>& reference) {
    ComparisonSummary summary{};

    for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
        const auto actual =
            make_packed_final_state(cells[cell_index], static_cast<int>(cell_index), grid_dim);
        const auto& expected = reference[cell_index];

        const bool metadata_match =
            actual.cell == expected.cell &&
            actual.i == expected.i &&
            actual.j == expected.j &&
            actual.k == expected.k &&
            actual.completed_steps == expected.completed_steps;
        if (!metadata_match && summary.failure_message.empty()) {
            std::ostringstream message;
            message << "metadata mismatch at cell " << cell_index
                    << " (actual cell/i/j/k/steps "
                    << actual.cell << "/" << actual.i << "/" << actual.j << "/"
                    << actual.k << "/" << actual.completed_steps
                    << ", reference " << expected.cell << "/" << expected.i << "/"
                    << expected.j << "/" << expected.k << "/"
                    << expected.completed_steps << ")";
            summary.failure_message = message.str();
        }
        summary.pass = summary.pass && metadata_match;

        for (int n = 0; n < pc::NumSpec; ++n) {
            const auto idx = static_cast<std::size_t>(n);
            const auto value = actual.xn[idx];
            const auto expected_value = expected.xn[idx];
            const auto rel = relative_error(value, expected_value, atol_spec);
            auto& max_rel = is_deuterium_bearing_species(n)
                                ? summary.max_deuterium_species_rel_error
                                : summary.max_non_deuterium_species_rel_error;
            max_rel = std::max(max_rel, rel);
            if (is_deuterium_bearing_species(n)) {
                continue;
            }
            const bool species_match =
                nearly_equal(value, expected_value, comparison_species_rtol[idx], atol_spec);
            if (!species_match && summary.failure_message.empty()) {
                std::ostringstream message;
                message << "species " << pc::short_spec_names[idx]
                        << " mismatch at cell " << cell_index
                        << " (actual " << format_scientific(value)
                        << ", reference " << format_scientific(expected_value)
                        << ", relative error " << format_scientific(rel)
                        << ", rtol " << comparison_species_rtol[idx]
                        << ", atol " << atol_spec << ")";
                summary.failure_message = message.str();
            }
            summary.pass = summary.pass && species_match;
        }

        const auto temperature_rel = relative_error(actual.T, expected.T, atol_spec);
        const auto energy_rel = relative_error(actual.e, expected.e, atol_energy);
        const auto rho_rel = relative_error(actual.rho, expected.rho, atol_spec);
        summary.max_thermodynamic_rel_error =
            std::max(summary.max_thermodynamic_rel_error,
                     std::max(temperature_rel, energy_rel));
        summary.max_rho_rel_error = std::max(summary.max_rho_rel_error, rho_rel);

        const bool temperature_match =
            nearly_equal(actual.T, expected.T, comparison_thermodynamic_rtol, atol_spec);
        const bool energy_match =
            nearly_equal(actual.e, expected.e, comparison_thermodynamic_rtol, atol_energy);
        const bool rho_match = nearly_equal(actual.rho, expected.rho, rtol_spec, atol_spec);
        if ((!temperature_match || !energy_match || !rho_match) &&
            summary.failure_message.empty()) {
            std::ostringstream message;
            message << "thermodynamic mismatch at cell " << cell_index
                    << " (T rel " << format_scientific(temperature_rel)
                    << ", e rel " << format_scientific(energy_rel)
                    << ", rho rel " << format_scientific(rho_rel) << ")";
            summary.failure_message = message.str();
        }
        summary.pass = summary.pass && temperature_match && energy_match && rho_match;
    }

    return summary;
}

bool compare_final_states_from_file(const std::vector<CollapseState>& cells, int grid_dim,
                                    const std::string& path) {
    std::vector<PackedFinalState> reference;
    if (!read_final_states(path, cells.size(), reference)) {
        return false;
    }

    const auto summary = compare_final_states(cells, grid_dim, reference);
    std::cout << "final-state comparison file: " << path << "\n";
    std::cout << "final-state comparison: " << (summary.pass ? "PASS" : "FAIL")
              << " (max deuterium-bearing species rel error "
              << format_scientific(summary.max_deuterium_species_rel_error)
              << ", max non-deuterium-bearing species rel error "
              << format_scientific(summary.max_non_deuterium_species_rel_error)
              << ", max thermodynamic rel error "
              << format_scientific(summary.max_thermodynamic_rel_error)
              << ", max rho rel error "
              << format_scientific(summary.max_rho_rel_error) << ")\n";
    if (!summary.pass && !summary.failure_message.empty()) {
        std::cout << "first comparison failure: " << summary.failure_message << "\n";
    }
    return summary.pass;
}

std::string final_state_filename(int grid_dim) {
    std::ostringstream name;
    name << "final_states_grid" << grid_dim << "_" << backend_name() << ".bin";
    return name.str();
}

} // namespace

#undef PRIMORDIAL_HOST_DEVICE

#ifndef PRIMORDIAL_ROS2S_NO_MAIN
int main(int argc, char** argv) {
    std::cout << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10);

    Options options;
    if (!parse_args(argc, argv, options)) {
        return 1;
    }
    if (options.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    int num_cells = 0;
    if (!checked_cell_count(options.grid_dim, num_cells)) {
        std::cerr << "grid dimension is too large: " << options.grid_dim << "\n";
        return 1;
    }

    pc::set_redshift(30.0);
    std::vector<CollapseState> cells(static_cast<std::size_t>(num_cells));
    for (auto& cell : cells) {
        cell = make_collapse_state();
    }

    integrators::IntegratorResult failure = integrators::IntegratorResult::SUCCESS;
    int completed_global_steps = 0;

    const auto start = std::chrono::steady_clock::now();
#if defined(PRIMORDIAL_ROS2S_ENABLE_CUDA)
    failure = run_cells_cuda(cells, options.perturb, completed_global_steps);
#else
    for (int step = 0; step < max_collapse_steps; ++step) {
        bool all_stopped = true;
        for (int cell = 0; cell < num_cells; ++cell) {
            auto& state = cells[static_cast<std::size_t>(cell)];
            if (state.completed_steps < completed_global_steps) {
                continue;
            }
            const bool stopped =
                advance_collapse_step(state, cell, step, options.perturb, failure);
            if (failure != integrators::IntegratorResult::SUCCESS) {
                break;
            }
            all_stopped = all_stopped && stopped;
        }
        if (failure != integrators::IntegratorResult::SUCCESS || all_stopped) {
            break;
        }
        completed_global_steps += 1;
    }
#endif
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    if (failure != integrators::IntegratorResult::SUCCESS) {
        return 1;
    }

    IntegratorStats total_stats{};
    for (const auto& cell : cells) {
        add_stats(total_stats, cell.stats);
    }
    const auto& representative = cells.front();

    std::cout << "Primordial chemistry collapse grid with ROS2S\n";
    std::cout << "grid: " << options.grid_dim << "^3 (" << num_cells << " cells)\n";
    std::cout << "perturbations: " << (options.perturb ? "enabled" : "disabled") << "\n";
    std::cout << "completed global collapse steps: " << completed_global_steps << "\n";
    print_collapse_summary(cells, representative);
    std::cout << "wall time: " << elapsed << " s\n";
    print_stats(cells, total_stats);

    bool comparison_pass = true;
    if (!options.compare_final_state_path.empty()) {
        comparison_pass = compare_final_states_from_file(
            cells, options.grid_dim, options.compare_final_state_path);
    }

    if (options.grid_dim > 1) {
        const std::string final_state_file = final_state_filename(options.grid_dim);
        if (!write_final_states(cells, options.grid_dim, final_state_file)) {
            return 1;
        }
        std::cout << "final states: " << final_state_file << " ("
                  << cells.size() << " packed records, "
                  << sizeof(PackedFinalState) << " bytes each)\n";
    } else {
        print_state(representative.current);
    }

    if (!comparison_pass) {
        return 1;
    }
    return 0;
}
#endif
