// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: YASS stiff integrator implementation
// ABOUTME: Non-iterative first-order method after Khokhlov's YASS solver
#ifndef YASS_HPP
#define YASS_HPP

#include <array>
#include <cmath>
#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

template<size_type N>
struct YASSState : public IntegratorState<N> {
    std::array<std::array<Real, N>, N> jacobian{};
    std::array<int, N> pivot{};
    bool allow_pivoting{true};
};

template<typename Problem>
class YASS {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = YASSState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;
    using RHS = typename ProblemTraits<Problem>::rhs_type;

private:
    // YASS step: (I - dt J(y0)) delta = dt F(y0), y1 = y0 + delta.
    IntegratorResult single_step(ProblemState& problem_state, State& state, Real dt) {
        const std::array<Real, N> y_old = state.y;
        std::array<Real, N> rhs_val{};

        Problem::rhs(state.t, state.y, rhs_val);
        state.n_rhs++;

        if (state.jacobian_analytic &&
            ProblemTraits<Problem>::has_analytic_jacobian) {
            if constexpr (ProblemTraits<Problem>::has_analytic_jacobian) {
                Problem::jacobian(state.t, state.y, state.jacobian);
            }
        } else {
            numerical_jacobian(problem_state, state);
            if (state.jacobian_analytic &&
                !ProblemTraits<Problem>::has_analytic_jacobian) {
                // Sticky fallback avoids probing an unavailable analytic
                // Jacobian repeatedly.
                state.jacobian_analytic = false;
            }
        }
        state.n_jac++;

        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                state.jacobian[i][j] *= -dt;
                if (i == j) {
                    state.jacobian[i][j] += 1.0;
                }
            }
        }

        std::array<Real, N> delta{};
        for (size_type i = 0; i < N; ++i) {
            delta[i] = dt * rhs_val[i];
        }

        int ierr;
        if (state.allow_pivoting) {
            ierr =
                linalg::lu_decomposition<N, true>(state.jacobian, state.pivot);
            if (ierr != 0) {
                state.y = y_old;
                return IntegratorResult::LU_DECOMPOSITION_ERROR;
            }
            linalg::lu_solve<N, true>(state.jacobian, state.pivot, delta);
        } else {
            ierr =
                linalg::lu_decomposition<N, false>(state.jacobian, state.pivot);
            if (ierr != 0) {
                state.y = y_old;
                return IntegratorResult::LU_DECOMPOSITION_ERROR;
            }
            linalg::lu_solve<N, false>(state.jacobian, state.pivot, delta);
        }

        for (size_type i = 0; i < N; ++i) {
            state.y[i] += delta[i];
        }

        return IntegratorResult::SUCCESS;
    }

    void numerical_jacobian([[maybe_unused]] ProblemState& problem_state, State& state) {
        std::array<Real, N> rhs_base{}, rhs_pert{};
        const std::array<Real, N> y_save = state.y;

        Problem::rhs(state.t, state.y, rhs_base);

        for (size_type j = 0; j < N; ++j) {
            const Real h = std::sqrt(math::UROUND) *
                           std::max(std::abs(state.y[j]), Real(1.0));
            state.y[j] += h;

            Problem::rhs(state.t, state.y, rhs_pert);

            for (size_type i = 0; i < N; ++i) {
                state.jacobian[i][j] = (rhs_pert[i] - rhs_base[i]) / h;
            }

            state.y[j] = y_save[j];
        }

        state.n_rhs += static_cast<int>(N + 1);
    }

public:
    IntegratorResult integrate(ProblemState& problem_state, State& state) {
        state.n_step = 0;
        state.n_rhs = 0;
        state.n_jac = 0;

        const Real remaining0 = state.tout - state.t;
        if (remaining0 == 0.0) {
            return IntegratorResult::SUCCESS;
        }

        if (state.dt == 0.0) {
            state.dt = remaining0;
        }

        const Real direction = (remaining0 > 0.0) ? 1.0 : -1.0;
        if (state.dt * direction < 0.0) {
            state.dt = -state.dt;
        }

        if (std::abs(state.dt - remaining0) <=
            1.e-12 * std::max(std::abs(remaining0), Real(1.0))) {
            const Real t_old = state.t;
            const std::array<Real, N> y_old = state.y;

            const auto result = single_step(problem_state, state, state.dt);

            if (result == IntegratorResult::SUCCESS) {
                state.t = state.tout;
                state.n_step = 1;
            } else {
                state.t = t_old;
                state.y = y_old;
                state.n_step = 0;
            }
            return result;
        }

        const Real safety_factor = 1.e-12;
        const int max_steps = 10000;

        Real dt_current = state.dt;

        while (direction * (state.tout - state.t) >
                   safety_factor * std::max(std::abs(state.tout), Real(1.0)) &&
               state.n_step < max_steps) {

            if (direction * (state.t + dt_current - state.tout) > 0.0) {
                dt_current = state.tout - state.t;
            }

            const Real t_save = state.t;
            const std::array<Real, N> y_save = state.y;

            auto result1 = single_step(problem_state, state, dt_current * 0.5);
            if (result1 != IntegratorResult::SUCCESS) {
                state.t = t_save;
                state.y = y_save;
                dt_current *= 0.5;
                continue;
            }

            state.t = t_save + dt_current * 0.5;
            auto result2 = single_step(problem_state, state, dt_current * 0.5);
            if (result2 != IntegratorResult::SUCCESS) {
                state.t = t_save;
                state.y = y_save;
                dt_current *= 0.5;
                continue;
            }

            const std::array<Real, N> y_fine = state.y;

            state.t = t_save;
            state.y = y_save;
            auto result3 = single_step(problem_state, state, dt_current);
            if (result3 != IntegratorResult::SUCCESS) {
                state.t = t_save;
                state.y = y_save;
                dt_current *= 0.5;
                continue;
            }

            Real error = 0.0;
            for (size_type i = 0; i < N; ++i) {
                const Real weight =
                    1.0 / (state.rtol * std::abs(y_fine[i]) + state.atol);
                const Real diff = std::abs(y_fine[i] - state.y[i]);
                error = std::max(error, weight * diff);
            }

            if (error < 1.0) {
                state.y = y_fine;
                state.t = t_save + dt_current;
                state.n_step++;

                const Real growth =
                    (error > 0.0) ? std::min(std::sqrt(1.0 / error), Real(2.0))
                                  : Real(2.0);
                dt_current *= growth;
            } else {
                state.t = t_save;
                state.y = y_save;
                dt_current *= 0.5;
            }
        }

        if (state.n_step >= max_steps) {
            return IntegratorResult::TOO_MANY_STEPS;
        }

        return IntegratorResult::SUCCESS;
    }
};

} // namespace integrators

#endif // YASS_HPP
