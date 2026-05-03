// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Backward Euler integrator implementation
// ABOUTME: Extracted from AMReX Microphysics with simplified interface
#ifndef BACKWARD_EULER_HPP
#define BACKWARD_EULER_HPP

#include <array>
#include <cmath>
#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

template<size_type N>
struct BackwardEulerState : public IntegratorState<N> {
    std::array<std::array<Real, N>, N> jacobian{};
    std::array<int, N> pivot{};
    int max_iter{50};
    Real tolerance{1.e-6};
    bool allow_pivoting{true};
};

template<typename Problem>
class BackwardEuler {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = BackwardEulerState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;
    using RHS = typename ProblemTraits<Problem>::rhs_type;
    
private:
    // Take a single implicit step
    INTEGRATORS_HOST_DEVICE IntegratorResult single_step(ProblemState& problem_state, State& state, Real dt) {
        std::array<Real, N> y_old = state.y;
        std::array<Real, N> rhs_val{};
        
        // Initial explicit predictor step
        Problem::rhs(state.t, state.y, rhs_val);
        state.n_rhs++;
        
        for (size_type i = 0; i < N; ++i) {
            state.y[i] = y_old[i] + dt * rhs_val[i];
        }
        
        // Newton iteration
        bool converged = false;
        for (int iter = 0; iter < state.max_iter; ++iter) {
            // Evaluate RHS at current state
            Problem::rhs(state.t + dt, state.y, rhs_val);
            state.n_rhs++;
            
            // Get Jacobian
            if (state.jacobian_analytic && ProblemTraits<Problem>::has_analytic_jacobian) {
                if constexpr (ProblemTraits<Problem>::has_analytic_jacobian) {
                    Problem::jacobian(state.t + dt, state.y, state.jacobian);
                }
            } else {
                numerical_jacobian(problem_state, state, dt);
                if (state.jacobian_analytic && !ProblemTraits<Problem>::has_analytic_jacobian) {
                    // Sticky fallback avoids probing an unavailable analytic Jacobian repeatedly.
                    state.jacobian_analytic = false;
                }
            }
            state.n_jac++;
            
            // Build system matrix: I - dt * J
            for (size_type i = 0; i < N; ++i) {
                for (size_type j = 0; j < N; ++j) {
                    state.jacobian[i][j] *= -dt;
                    if (i == j) {
                        state.jacobian[i][j] += 1.0;
                    }
                }
            }
            
            // Build RHS: y_old - y_current + dt * f(y_current)
            std::array<Real, N> rhs_newton{};
            for (size_type i = 0; i < N; ++i) {
                rhs_newton[i] = y_old[i] - state.y[i] + dt * rhs_val[i];
            }
            
            // Solve linear system
            if (state.allow_pivoting) {
                const int ierr = linalg::lu_factor_solve<N, true>(state.jacobian, state.pivot, rhs_newton);
                if (ierr != 0) {
                    state.y = y_old;
                    return IntegratorResult::LU_DECOMPOSITION_ERROR;
                }
            } else {
                const int ierr = linalg::lu_factor_solve<N, false>(state.jacobian, state.pivot, rhs_newton);
                if (ierr != 0) {
                    state.y = y_old;
                    return IntegratorResult::LU_DECOMPOSITION_ERROR;
                }
            }
            
            // Update solution
            for (size_type i = 0; i < N; ++i) {
                state.y[i] += rhs_newton[i];
            }
            
            // Check convergence
            Real y_norm = linalg::norm2(state.y);
            Real correction_norm = linalg::norm2(rhs_newton);
            const Real scale = std::max(y_norm, Real(1.0));
            
            if (correction_norm <= state.tolerance * scale) {
                converged = true;
                break;
            }
        }
        
        if (!converged) {
            // Reset to old solution
            state.y = y_old;
            return IntegratorResult::CORRECTOR_CONVERGENCE;
        }
        
        return IntegratorResult::SUCCESS;
    }
    
    INTEGRATORS_HOST_DEVICE void numerical_jacobian([[maybe_unused]] ProblemState& problem_state, State& state, Real dt) {
        std::array<Real, N> rhs_base{}, rhs_pert{};
        std::array<Real, N> y_save = state.y;
        
        // Base RHS evaluation
        Problem::rhs(state.t + dt, state.y, rhs_base);
        
        // Compute finite difference approximation
        for (size_type j = 0; j < N; ++j) {
            Real h = std::sqrt(math::UROUND) * std::max(std::abs(state.y[j]), 1.0);
            state.y[j] += h;
            
            Problem::rhs(state.t + dt, state.y, rhs_pert);
            
            for (size_type i = 0; i < N; ++i) {
                state.jacobian[i][j] = (rhs_pert[i] - rhs_base[i]) / h;
            }
            
            state.y[j] = y_save[j]; // Restore
        }
        
        state.n_rhs += static_cast<int>(N + 1);
    }
    
public:
    INTEGRATORS_HOST_DEVICE IntegratorResult integrate(ProblemState& problem_state, State& state) {
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
        
        // Single step mode
        if (std::abs(state.dt - remaining0) <= 1.e-12 * std::max(std::abs(remaining0), Real(1.0))) {
            const Real t_old = state.t;
            const std::array<Real, N> y_old = state.y;

            auto result = single_step(problem_state, state, state.dt);

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
        
        // Adaptive timestepping with Richardson extrapolation
        const Real safety_factor = 1.e-12;
        const int max_steps = 10000;
        
        Real dt_current = state.dt;
        
        while (direction * (state.tout - state.t) > safety_factor * std::max(std::abs(state.tout), Real(1.0)) &&
               state.n_step < max_steps) {
            
            // Don't overshoot
            if (direction * (state.t + dt_current - state.tout) > 0.0) {
                dt_current = state.tout - state.t;
            }
            
            std::array<Real, N> y_save = state.y;
            
            // Take two half steps
            auto result1 = single_step(problem_state, state, dt_current * 0.5);
            if (result1 != IntegratorResult::SUCCESS) {
                state.y = y_save;
                dt_current *= 0.5;
                continue;
            }
            
            auto result2 = single_step(problem_state, state, dt_current * 0.5);
            if (result2 != IntegratorResult::SUCCESS) {
                state.y = y_save;
                dt_current *= 0.5;
                continue;
            }
            
            std::array<Real, N> y_fine = state.y;
            
            // Take one full step
            state.y = y_save;
            auto result3 = single_step(problem_state, state, dt_current);
            if (result3 != IntegratorResult::SUCCESS) {
                state.y = y_save;
                dt_current *= 0.5;
                continue;
            }
            
            // Estimate error
            Real error = 0.0;
            for (size_type i = 0; i < N; ++i) {
                Real weight = 1.0 / (state.rtol * std::abs(y_fine[i]) + state.atol);
                Real diff = std::abs(y_fine[i] - state.y[i]);
                error = std::max(error, weight * diff);
            }
            
            if (error < 1.0) {
                // Accept step with fine solution
                state.y = y_fine;
                state.t += dt_current;
                state.n_step++;
                
                // Adjust timestep for next step
                const Real growth = (error > 0.0) ? std::min(std::sqrt(1.0 / error), Real(2.0)) : Real(2.0);
                dt_current *= growth;
            } else {
                // Reject step
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

#endif // BACKWARD_EULER_HPP
