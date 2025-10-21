// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Steady-state departure integrator using frozen-Jacobian exponential propagation
#ifndef INTEGRATORS_STEADY_STATE_DEPARTURE_HPP
#define INTEGRATORS_STEADY_STATE_DEPARTURE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

struct DepartureIntegrationConfig {
    Real defect_tolerance{0.3};
    Real rebase_defect{0.8};
    Real max_departure_norm{1.0};
    Real min_step{1.0e-12};
    Real max_step{1.0};
    Real shrink_factor{0.5};
    Real growth_factor{2.0};
    Real safety{0.9};
    int max_attempts{6};
};

enum class DepartureStepStatus {
    Accepted,
    RebaseRequested,
    Failure
};

template<typename Problem>
struct DepartureState {
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = typename ProblemTraits<Problem>::state_type;
    using Matrix = typename ProblemTraits<Problem>::jacobian_type;

    Real current_time{0.0};
    State y_star{};
    State y{};
    State delta{};
    State atol{};
    State rtol{};
    Matrix jacobian{};
    State rhs_star{};
    Real defect_last{0.0};
};

template<typename Problem>
struct DepartureWorkspace {
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using Matrix = typename ProblemTraits<Problem>::jacobian_type;

    Matrix exp_cache{};
    Matrix phi1_cache{};
    Real cached_step{0.0};
    bool cache_valid{false};

    void invalidate() {
        cache_valid = false;
        cached_step = Real{0.0};
    }
};

struct DepartureStepResult {
    DepartureStepStatus status{DepartureStepStatus::Failure};
    Real actual_step{0.0};
    Real suggested_step{0.0};
    Real defect{0.0};
    Real delta_norm{0.0};
};

namespace detail {

inline Real clamp_step(Real value, Real min_step, Real max_step) {
    return std::clamp(value, min_step, max_step);
}

} // namespace detail

template<typename Problem>
DepartureStepResult integrate_departure_step(DepartureState<Problem>& state,
                                             DepartureWorkspace<Problem>& workspace,
                                             const DepartureIntegrationConfig& config,
                                             Real requested_step) {
    DepartureStepResult result{};
    constexpr size_type N = ProblemTraits<Problem>::neqs;

    if (requested_step <= Real{0.0}) {
        result.status = DepartureStepStatus::Failure;
        return result;
    }

    Real h = detail::clamp_step(requested_step, config.min_step, config.max_step);
    const Real step_tol = std::max(config.min_step * Real{1e-6}, Real{1e-16});

    typename ProblemTraits<Problem>::state_type delta_trial{};
    typename ProblemTraits<Problem>::state_type y_trial{};
    typename ProblemTraits<Problem>::rhs_type rhs_trial{};
    typename ProblemTraits<Problem>::state_type lin_term{};
    typename ProblemTraits<Problem>::state_type defect{};
    typename ProblemTraits<Problem>::state_type temp{};

    for (int attempt = 0; attempt < config.max_attempts; ++attempt) {
        if (!workspace.cache_valid ||
            std::abs(workspace.cached_step - h) > std::max(step_tol, std::abs(h) * Real{1e-10})) {
            const auto scaled = linalg::matrix_scale(state.jacobian, h);
            if (!linalg::matrix_exponential(scaled, workspace.exp_cache)) {
                workspace.invalidate();
                h = detail::clamp_step(config.safety * config.shrink_factor * h, config.min_step, config.max_step);
                continue;
            }
            if (!linalg::matrix_phi1(scaled, workspace.phi1_cache)) {
                workspace.invalidate();
                h = detail::clamp_step(config.safety * config.shrink_factor * h, config.min_step, config.max_step);
                continue;
            }
            workspace.cached_step = h;
            workspace.cache_valid = true;
        }

        linalg::matvec(workspace.exp_cache, state.delta, delta_trial);
        linalg::matvec(workspace.phi1_cache, state.rhs_star, temp);
        for (size_type i = 0; i < N; ++i) {
            delta_trial[i] += h * temp[i];
            y_trial[i] = state.y_star[i] + delta_trial[i];
        }

        Problem::rhs(state.current_time + h, y_trial, rhs_trial);
        linalg::matvec(state.jacobian, delta_trial, lin_term);
        Real max_weighted_defect = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real model_rhs = lin_term[i] + state.rhs_star[i];
            defect[i] = rhs_trial[i] - model_rhs;
            Real weight = state.atol[i] + state.rtol[i] * std::abs(y_trial[i]);
            weight = std::max(weight, Real{1e-30});
            const Real scaled = std::abs(defect[i]) / weight;
            max_weighted_defect = std::max(max_weighted_defect, scaled);
        }

        if (!std::isfinite(max_weighted_defect)) {
            workspace.invalidate();
            h = detail::clamp_step(config.safety * config.shrink_factor * h, config.min_step, config.max_step);
            continue;
        }

        if (max_weighted_defect <= config.defect_tolerance) {
            state.delta = delta_trial;
            state.y = y_trial;
            state.current_time += h;
            state.defect_last = max_weighted_defect;

            Real growth = config.growth_factor;
            if (max_weighted_defect > Real{0.0}) {
                const Real control = config.safety * std::pow(max_weighted_defect / config.defect_tolerance, Real{-0.5});
                growth = std::clamp(control, config.shrink_factor, config.growth_factor);
            }
            const Real suggested = detail::clamp_step(h * growth, config.min_step, config.max_step);

            const Real delta_norm = linalg::norm_inf(state.delta);
            result.status = DepartureStepStatus::Accepted;
            if (delta_norm >= config.max_departure_norm || max_weighted_defect >= config.rebase_defect) {
                result.status = DepartureStepStatus::RebaseRequested;
                workspace.invalidate();
            }

            result.actual_step = h;
            result.suggested_step = suggested;
            result.defect = max_weighted_defect;
            result.delta_norm = linalg::norm_inf(state.delta);
            return result;
        }

        workspace.invalidate();
        const Real new_step = detail::clamp_step(config.safety * config.shrink_factor * h,
                                                 config.min_step,
                                                 config.max_step);
        if (new_step >= h - step_tol) {
            break;
        }
        h = new_step;
    }

    result.status = DepartureStepStatus::Failure;
    result.actual_step = 0.0;
    result.suggested_step = std::max(config.min_step, h * config.shrink_factor);
    result.defect = std::numeric_limits<Real>::infinity();
    result.delta_norm = linalg::norm_inf(state.delta);
    workspace.invalidate();
    return result;
}

} // namespace integrators

#endif // INTEGRATORS_STEADY_STATE_DEPARTURE_HPP
