// SPDX-License-Identifier: BSD-3-Clause
#include <atomic>
#include <thread>

#define PRIMORDIAL_ROS2S_NO_MAIN
#include "main.cpp"

namespace {

constexpr int default_thread_count = 1;

struct RefOptions {
    Options base{};
    int num_threads{default_thread_count};
};

void print_ref_usage(const char* program) {
    std::cerr << "usage: " << program
              << " [--grid N] [--perturb|--no-perturb]"
                 " [--compare-final-state FILE|--no-compare-final-state] [-N THREADS]\n";
}

bool parse_ref_args(int argc, char** argv, RefOptions& options) {
    std::vector<char*> forwarded_args;
    forwarded_args.reserve(static_cast<std::size_t>(argc));
    forwarded_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "-N") {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], options.num_threads)) {
                print_ref_usage(argv[0]);
                return false;
            }
            continue;
        }

        constexpr std::string_view threads_prefix = "-N=";
        if (arg.rfind(threads_prefix, 0) == 0) {
            if (!parse_positive_int(argv[i] + threads_prefix.size(), options.num_threads)) {
                print_ref_usage(argv[0]);
                return false;
            }
            continue;
        }

        forwarded_args.push_back(argv[i]);
    }

    if (!parse_args(static_cast<int>(forwarded_args.size()), forwarded_args.data(),
                    options.base)) {
        print_ref_usage(argv[0]);
        return false;
    }
    return true;
}

void run_cell(CollapseState& state, int cell, bool perturb,
              integrators::IntegratorResult& failure) {
    for (int step = 0; step < max_collapse_steps; ++step) {
        const bool stopped = advance_collapse_step(state, cell, step, perturb, failure);
        if (failure != integrators::IntegratorResult::SUCCESS || stopped) {
            return;
        }
    }
}

integrators::IntegratorResult run_cells_threaded(std::vector<CollapseState>& cells,
                                                 bool perturb, int num_threads) {
    std::atomic<int> next_cell{0};
    std::atomic<bool> failed{false};
    std::vector<integrators::IntegratorResult> failures(
        static_cast<std::size_t>(num_threads), integrators::IntegratorResult::SUCCESS);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(num_threads));

    for (int thread = 0; thread < num_threads; ++thread) {
        workers.emplace_back([&, thread] {
            for (;;) {
                if (failed.load(std::memory_order_relaxed)) {
                    return;
                }

                const int cell = next_cell.fetch_add(1, std::memory_order_relaxed);
                if (cell >= static_cast<int>(cells.size())) {
                    return;
                }

                auto result = integrators::IntegratorResult::SUCCESS;
                run_cell(cells[static_cast<std::size_t>(cell)], cell, perturb, result);
                if (result != integrators::IntegratorResult::SUCCESS) {
                    failures[static_cast<std::size_t>(thread)] = result;
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    for (const auto failure : failures) {
        if (failure != integrators::IntegratorResult::SUCCESS) {
            return failure;
        }
    }
    return integrators::IntegratorResult::SUCCESS;
}

int max_completed_steps(const std::vector<CollapseState>& cells) {
    int completed = 0;
    for (const auto& cell : cells) {
        completed = std::max(completed, cell.completed_steps);
    }
    return completed;
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10);

    RefOptions options;
    if (!parse_ref_args(argc, argv, options)) {
        return 1;
    }
    if (options.base.show_help) {
        print_ref_usage(argv[0]);
        return 0;
    }

    int num_cells = 0;
    if (!checked_cell_count(options.base.grid_dim, num_cells)) {
        std::cerr << "grid dimension is too large: " << options.base.grid_dim << "\n";
        return 1;
    }

    const int num_threads = std::min(options.num_threads, num_cells);

    pc::set_redshift(30.0);
    std::vector<CollapseState> cells(static_cast<std::size_t>(num_cells));
    for (auto& cell : cells) {
        cell = make_collapse_state();
    }

    const auto start = std::chrono::steady_clock::now();
    const auto failure = run_cells_threaded(cells, options.base.perturb, num_threads);
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
    std::cout << "grid: " << options.base.grid_dim << "^3 (" << num_cells << " cells)\n";
    std::cout << "threads: " << num_threads << "\n";
    std::cout << "perturbations: " << (options.base.perturb ? "enabled" : "disabled") << "\n";
    std::cout << "completed global collapse steps: " << max_completed_steps(cells) << "\n";
    print_collapse_summary(cells, representative);
    std::cout << "wall time: " << elapsed << " s\n";
    print_stats(cells, total_stats);

    bool comparison_pass = true;
    if (!options.base.compare_final_state_path.empty()) {
        comparison_pass = compare_final_states_from_file(
            cells, options.base.grid_dim, options.base.compare_final_state_path);
    }

    if (options.base.grid_dim > 1) {
        const std::string final_state_file = final_state_filename(options.base.grid_dim);
        if (!write_final_states(cells, options.base.grid_dim, final_state_file)) {
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
