// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Steady-state solver using GTH factorization with Picard iteration
#ifndef INTEGRATORS_STEADY_STATE_GTH_HPP
#define INTEGRATORS_STEADY_STATE_GTH_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <type_traits>

#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

struct SteadyStateGthConfig {
    Real atol{1.0e-14};
    Real rtol{1.0e-8};
    Real iteration_tol{1.0e-10};
    int max_iters{50};
    Real generator_tol{1.0e-12};
    bool enforce_generator_structure{true};
    bool verbose{false};
    Real norm_target{std::numeric_limits<Real>::quiet_NaN()};
    Real constraint_tol{1.0e-12};
    bool enforce_constraints{true};
};

enum class SteadyStateStatus {
    Success,
    GeneratorCheckFailed,
    FactorizationFailure,
    NormalizationFailure,
    ConstraintFailure,
    MaxIterations
};

struct SteadyStateResult {
    SteadyStateStatus status{SteadyStateStatus::MaxIterations};
    int iterations{0};
    Real residual{std::numeric_limits<Real>::infinity()};
};

namespace detail {

template<typename Problem, typename = void>
struct has_steady_state_generator : std::false_type {};

template<typename Problem>
struct has_steady_state_generator<
    Problem,
    std::void_t<decltype(Problem::steady_state_generator(
        std::declval<const typename ProblemTraits<Problem>::state_type&>(),
        std::declval<typename ProblemTraits<Problem>::jacobian_type&>()))>> : std::true_type {};

template<typename Problem, typename = void>
struct has_steady_state_order : std::false_type {};

template<typename Problem>
struct has_steady_state_order<
    Problem,
    std::void_t<decltype(Problem::steady_state_order(
        std::declval<std::array<size_type, ProblemTraits<Problem>::neqs>&>()))>> : std::true_type {};

template<typename Problem, typename = void>
struct has_steady_state_constraints : std::false_type {};

template<typename Problem>
struct has_steady_state_constraints<
    Problem,
    std::void_t<decltype(Problem::steady_state_constraints(
        std::declval<const typename ProblemTraits<Problem>::state_type&>(),
        std::declval<std::array<std::array<Real, ProblemTraits<Problem>::neqs>, ProblemTraits<Problem>::neqs>&>()))>>
    : std::true_type {};

template<typename Problem, typename = void>
struct has_steady_state_constraint_count : std::false_type {};

template<typename Problem>
struct has_steady_state_constraint_count<
    Problem,
    std::void_t<decltype(Problem::steady_state_constraint_count())>> : std::true_type {};

} // namespace detail

template<typename Problem>
SteadyStateResult steady_state_gth(typename ProblemTraits<Problem>::state_type& y,
                                   const SteadyStateGthConfig& config = {}) {
    static_assert(detail::has_steady_state_generator<Problem>::value,
                  "Problem must provide steady_state_generator(state, matrix)");

    constexpr size_type N = ProblemTraits<Problem>::neqs;
    using Vector = typename ProblemTraits<Problem>::state_type;
    using Matrix = typename ProblemTraits<Problem>::jacobian_type;

    static_assert(std::is_same_v<Vector, std::array<Real, N>>,
                  "steady_state_gth currently requires state_type = std::array<Real, N>");
    static_assert(std::is_same_v<Matrix, std::array<std::array<Real, N>, N>>,
                  "steady_state_gth currently requires jacobian_type = std::array<std::array<Real,N>,N>");

    SteadyStateResult result;
    Vector y_curr = y;
    const Vector y_ref = y_curr;

    const Real inferred_norm = std::accumulate(y_curr.begin(), y_curr.end(), Real{0});
    const Real norm_target = std::isfinite(config.norm_target) ? config.norm_target : inferred_norm;

    if (norm_target <= 0.0) {
        result.status = SteadyStateStatus::NormalizationFailure;
        result.residual = norm_target;
        return result;
    }

    size_type constraint_count = size_type{1};
    if constexpr (detail::has_steady_state_constraint_count<Problem>::value) {
        const size_type declared = Problem::steady_state_constraint_count();
        if (declared > 0) {
            constraint_count = std::min<size_type>(declared, N);
        }
    }
    constraint_count = std::max<size_type>(constraint_count, size_type{1});

    std::array<std::array<Real, N>, N> constraint_matrix{};
    for (auto& row : constraint_matrix) {
        row.fill(Real{0});
    }
    constraint_matrix[0].fill(Real{1});

    if constexpr (detail::has_steady_state_constraints<Problem>::value) {
        Problem::steady_state_constraints(y_ref, constraint_matrix);
    }

    std::array<Real, N> constraint_rhs{};
    auto compute_constraint_rhs = [&](const Vector& state) {
        constraint_rhs.fill(Real{0});
        for (size_type row = 0; row < constraint_count; ++row) {
            Real sum = 0.0;
            for (size_type col = 0; col < N; ++col) {
                sum += constraint_matrix[row][col] * state[col];
            }
            constraint_rhs[row] = sum;
        }
    };
    compute_constraint_rhs(y_ref);
    if (constraint_count >= 1) {
        constraint_rhs[0] = norm_target;
    }

    auto constraint_residual = [&](const Vector& state, std::array<Real, N>& residual) -> Real {
        residual.fill(Real{0});
        Real worst = 0.0;
        for (size_type row = 0; row < constraint_count; ++row) {
            Real sum = 0.0;
            for (size_type col = 0; col < N; ++col) {
                sum += constraint_matrix[row][col] * state[col];
            }
            const Real diff = constraint_rhs[row] - sum;
            residual[row] = diff;
            worst = std::max(worst, std::abs(diff));
        }
        return worst;
    };

    auto project_to_constraints = [&](Vector& state) -> bool {
        if (!config.enforce_constraints) {
            return true;
        }

        std::array<Real, N> residual{};
        Real worst = constraint_residual(state, residual);
        if (worst <= config.constraint_tol) {
            return true;
        }

        std::array<std::array<Real, N>, N> gram{};
        for (size_type i = 0; i < N; ++i) {
            gram[i].fill(Real{0});
            gram[i][i] = Real{1};
        }
        for (size_type i = 0; i < constraint_count; ++i) {
            for (size_type j = 0; j < constraint_count; ++j) {
                Real sum = 0.0;
                for (size_type col = 0; col < N; ++col) {
                    sum += constraint_matrix[i][col] * constraint_matrix[j][col];
                }
                gram[i][j] = sum;
            }
        }

        std::array<int, N> ipvt{};
        auto gram_factor = gram;
        const int info = linalg::lu_decomposition<N>(gram_factor, ipvt);
        if (info != 0) {
            return false;
        }

        std::array<Real, N> lambda{};
        lambda.fill(Real{0});
        for (size_type i = 0; i < constraint_count; ++i) {
            lambda[i] = residual[i];
        }

        linalg::lu_solve<N>(gram_factor, ipvt, lambda);

        Vector correction{};
        correction.fill(Real{0});
        for (size_type row = 0; row < constraint_count; ++row) {
            const Real lambda_row = lambda[row];
            if (lambda_row == Real{0}) {
                continue;
            }
            for (size_type col = 0; col < N; ++col) {
                correction[col] += constraint_matrix[row][col] * lambda_row;
            }
        }

        for (size_type i = 0; i < N; ++i) {
            state[i] += correction[i];
            if (state[i] < -config.constraint_tol) {
                return false;
            }
            if (state[i] < Real{0} && state[i] > -config.constraint_tol) {
                state[i] = Real{0};
            }
        }

        worst = constraint_residual(state, residual);
        return worst <= config.constraint_tol;
    };

    std::array<size_type, N> order{};
    std::iota(order.begin(), order.end(), size_type{0});
    if constexpr (detail::has_steady_state_order<Problem>::value) {
        Problem::steady_state_order(order);
    }
    std::array<size_type, N> inverse{};
    for (size_type i = 0; i < N; ++i) {
        inverse[order[i]] = i;
    }

    Matrix generator{};
    Matrix permuted{};
    Vector y_perm{};
    std::array<Real, N> inv_piv{};

    for (int iter = 0; iter < config.max_iters; ++iter) {
        Problem::steady_state_generator(y_curr, generator);

        if (config.enforce_generator_structure) {
            for (size_type i = 0; i < N; ++i) {
                Real row_sum = 0.0;
                for (size_type j = 0; j < N; ++j) {
                    const Real val = generator[i][j];
                    row_sum += val;
                    if (i != j && val > config.generator_tol) {
                        result.status = SteadyStateStatus::GeneratorCheckFailed;
                        result.residual = val;
                        return result;
                    }
                }
                if (std::abs(row_sum) > config.generator_tol) {
                    result.status = SteadyStateStatus::GeneratorCheckFailed;
                    result.residual = row_sum;
                    return result;
                }
                if (generator[i][i] < -config.generator_tol) {
                    result.status = SteadyStateStatus::GeneratorCheckFailed;
                    result.residual = generator[i][i];
                    return result;
                }
            }
        }

        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                permuted[i][j] = generator[order[i]][order[j]];
            }
        }

        Matrix factored = permuted;
        int info = linalg::gth_factorization<N>(factored, inv_piv, linalg::GthMatrixKind::Generator);
        if (info != 0) {
            result.status = SteadyStateStatus::FactorizationFailure;
            result.residual = static_cast<Real>(info);
            return result;
        }

        std::fill(y_perm.begin(), y_perm.end(), Real{0});
        info = linalg::gth_solve<N>(factored, inv_piv, y_perm, norm_target, linalg::GthMatrixKind::Generator);
        if (info != 0) {
            result.status = SteadyStateStatus::NormalizationFailure;
            result.residual = static_cast<Real>(info);
            return result;
        }

        Vector y_next{};
        for (size_type i = 0; i < N; ++i) {
            y_next[order[i]] = y_perm[i];
        }

        if (!project_to_constraints(y_next)) {
            result.status = SteadyStateStatus::ConstraintFailure;
            result.residual = std::numeric_limits<Real>::infinity();
            return result;
        }

        Real max_rel_change = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real denom = config.atol + config.rtol * std::abs(y_next[i]);
            const Real rel = denom > 0.0 ? std::abs(y_next[i] - y_curr[i]) / denom
                                         : std::abs(y_next[i] - y_curr[i]);
            max_rel_change = std::max(max_rel_change, rel);
        }

        result.iterations = iter + 1;
        result.residual = max_rel_change;

        if (config.verbose) {
            std::cout << "[steady_state_gth] iter=" << result.iterations
                      << " residual=" << result.residual << "\n";
        }

        y_curr = y_next;

        if (max_rel_change <= config.iteration_tol) {
            y = y_curr;
            result.status = SteadyStateStatus::Success;
            return result;
        }
    }

    y = y_curr;
    result.status = SteadyStateStatus::MaxIterations;
    return result;
}

} // namespace integrators

#endif // INTEGRATORS_STEADY_STATE_GTH_HPP
