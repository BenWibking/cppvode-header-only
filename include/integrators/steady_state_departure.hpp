// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Steady-state snap helper that either rebases to LTE or defers to VODE
#ifndef INTEGRATORS_STEADY_STATE_DEPARTURE_HPP
#define INTEGRATORS_STEADY_STATE_DEPARTURE_HPP

#include <array>
#include <cmath>
#include <limits>

#include "integrator_types.hpp"
#include "steady_state_gth.hpp"

namespace integrators {

struct SteadyStateSnapConfig {
    SteadyStateGthConfig steady_state{};
    Real balance_tolerance{1.0e-3};
    Real balance_floor{1.0e-30};
    Real timescale_safety{0.3};
    Real min_diagonal{1.0e-30};
    Real max_departure_tolerance{std::numeric_limits<Real>::infinity()};
    Real departure_floor{1.0e-30};
};

enum class SteadyStateSnapResult {
    Snapped,
    FallbackToVode,
    Failure
};

struct SteadyStateSnapOutcome {
    SteadyStateSnapResult result{SteadyStateSnapResult::FallbackToVode};
    Real max_balance_mismatch{std::numeric_limits<Real>::infinity()};
    Real max_timescale{std::numeric_limits<Real>::infinity()};
    Real max_weighted_departure{std::numeric_limits<Real>::infinity()};
};

namespace detail {

template<typename State>
inline Real max_weighted_departure(const State& y,
                                   const State& y_star,
                                   const State& atol,
                                   const State& rtol,
                                   Real floor) {
    Real worst = 0.0;
    for (size_type i = 0; i < y.size(); ++i) {
        Real weight = atol[i] + rtol[i] * std::max(std::abs(y[i]), std::abs(y_star[i]));
        weight = std::max(weight, floor);
        const Real scaled = std::abs(y[i] - y_star[i]) / weight;
        worst = std::max(worst, scaled);
    }
    return worst;
}

template<typename Matrix, typename State>
inline Real max_balance_mismatch(const Matrix& generator,
                                 const State& y_star,
                                 Real floor) {
    Real worst = 0.0;
    const size_type N = y_star.size();
    for (size_type i = 0; i < N; ++i) {
        for (size_type j = i + 1; j < N; ++j) {
            const Real forward = std::max<Real>(0.0, -generator[i][j]) * y_star[i];
            const Real backward = std::max<Real>(0.0, -generator[j][i]) * y_star[j];
            const Real denom = std::max(floor, std::max(forward, backward));
            const Real mismatch = std::abs(forward - backward) / denom;
            worst = std::max(worst, mismatch);
        }
    }
    return worst;
}

template<typename Matrix>
inline Real max_relaxation_timescale(const Matrix& generator, Real min_diagonal) {
    Real worst = 0.0;
    const size_type N = generator.size();
    for (size_type i = 0; i < N; ++i) {
        const Real diag = generator[i][i];
        if (!(diag > min_diagonal)) {
            return std::numeric_limits<Real>::infinity();
        }
        const Real tau = Real{1} / diag;
        worst = std::max(worst, tau);
    }
    return worst;
}

} // namespace detail

template<typename Problem>
SteadyStateSnapOutcome attempt_steady_state_snap(
    [[maybe_unused]] Real time,
    typename ProblemTraits<Problem>::state_type& y,
    const typename ProblemTraits<Problem>::state_type& atol,
    const typename ProblemTraits<Problem>::state_type& rtol,
    Real hydro_dt,
    bool allow_fallback_to_vode,
    const SteadyStateSnapConfig& config = {}) {

    using Traits = ProblemTraits<Problem>;
    using State = typename Traits::state_type;
    using Matrix = typename Traits::jacobian_type;

    SteadyStateSnapOutcome outcome{};

    State y_star = y;
    auto steady_cfg = config.steady_state;
    const auto ss_result = steady_state_gth<Problem>(y_star, steady_cfg);
    if (ss_result.status != SteadyStateStatus::Success) {
        outcome.result = allow_fallback_to_vode ? SteadyStateSnapResult::FallbackToVode
                                                : SteadyStateSnapResult::Failure;
        return outcome;
    }

    Matrix generator{};
    Problem::steady_state_generator(y_star, generator);

    outcome.max_balance_mismatch = detail::max_balance_mismatch(generator,
                                                                y_star,
                                                                config.balance_floor);
    outcome.max_timescale = detail::max_relaxation_timescale(generator,
                                                             config.min_diagonal);
    outcome.max_weighted_departure = detail::max_weighted_departure(y,
                                                                    y_star,
                                                                    atol,
                                                                    rtol,
                                                                    config.departure_floor);

    const bool balance_ok = outcome.max_balance_mismatch <= config.balance_tolerance;
    const bool timescale_ok = (hydro_dt <= Real{0})
                                  ? true
                                  : (outcome.max_timescale <= config.timescale_safety * hydro_dt);
    const bool departure_ok = std::isfinite(config.max_departure_tolerance)
                                  ? (outcome.max_weighted_departure <= config.max_departure_tolerance)
                                  : true;

    if (balance_ok && timescale_ok && departure_ok) {
        y = y_star;
        outcome.result = SteadyStateSnapResult::Snapped;
        return outcome;
    }

    outcome.result = allow_fallback_to_vode ? SteadyStateSnapResult::FallbackToVode
                                            : SteadyStateSnapResult::Failure;
    return outcome;
}

} // namespace integrators

#endif // INTEGRATORS_STEADY_STATE_DEPARTURE_HPP
