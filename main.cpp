// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

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
constexpr int default_grid_dim = 1;
constexpr int perturbation_interval = 20;
constexpr integrators::Real perturbation_amplitude = 0.1;

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
    pc::burn_t initial{};
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

std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

integrators::Real perturbation_factor(int cell, int step) {
    const auto seed = (static_cast<std::uint64_t>(static_cast<unsigned int>(cell)) << 32U) ^
                      (static_cast<std::uint64_t>(static_cast<unsigned int>(step)) << 16U);
    const auto bits = splitmix64(seed) >> 11U;
    const auto unit = static_cast<integrators::Real>(bits) *
                      (1.0 / static_cast<integrators::Real>(1ULL << 53U));
    return 1.0 + perturbation_amplitude * (2.0 * unit - 1.0);
}

void apply_perturbation(CollapseState& collapse, int cell, int step, bool enabled) {
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

bool advance_collapse_step(CollapseState& collapse, int cell, int step, bool perturb,
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
        std::cerr << "ROS2S failed on collapse step " << step
                  << " cell " << cell
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
    std::cerr << "usage: " << program << " [--grid N] [--perturb]\n";
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

} // namespace

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
    print_state(representative.current);
    return 0;
}
