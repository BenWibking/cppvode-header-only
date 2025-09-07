// ABOUTME: Simplified VODE integrator that actually works
// ABOUTME: Uses basic first-order BDF (Backward Euler) with adaptive timestep control
#ifndef VODE_HPP
#define VODE_HPP

#include <array>
#include <cmath>
#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

template<size_type N>
struct VODEState : public IntegratorState<N> {
    std::array<std::array<Real, N>, N> jacobian{};
    std::array<int, N> pivot{};
    Real h_current{0.0};
    bool allow_pivoting{true};
    int max_newton_iter{10};
    Real newton_tolerance{1.e-10};
};

template<typename Problem>
class VODE {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = VODEState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;
    
private:
    // Estimate good initial timestep
    Real estimate_initial_timestep(ProblemState& problem_state, State& state) {
        ProblemState y_current{};
        for (size_type i = 0; i < N; ++i) {
            y_current[i] = state.y[i];
        }
        
        std::array<Real, N> rhs_val{};
        Problem::rhs(state.t, y_current, rhs_val);
        state.n_rhs++;
        
        // Find characteristic scale
        Real y_scale = 0.0, rhs_scale = 0.0;
        for (size_type i = 0; i < N; ++i) {
            y_scale = std::max(y_scale, std::abs(state.y[i]));
            rhs_scale = std::max(rhs_scale, std::abs(rhs_val[i]));
        }
        
        if (y_scale == 0.0) y_scale = 1.0;
        if (rhs_scale == 0.0) rhs_scale = 1.0;
        
        // Initial timestep estimate
        Real dt = 0.01 * y_scale / rhs_scale;
        
        // Clamp to reasonable bounds
        Real dt_max = std::abs(state.tout - state.t);
        Real dt_min = 100.0 * math::UROUND * std::max(std::abs(state.t), dt_max);
        
        return math::clamp(dt, dt_min, std::min(dt_max, 0.1));
    }
    
    // Take one implicit Euler step
    IntegratorResult take_step(ProblemState& problem_state, State& state, Real dt) {
        ProblemState y_old{}, y_current{};
        
        // Save old solution
        for (size_type i = 0; i < N; ++i) {
            y_old[i] = state.y[i];
            y_current[i] = state.y[i];
        }
        
        // Newton iteration for implicit step
        bool converged = false;
        for (int iter = 0; iter < state.max_newton_iter; ++iter) {
            
            // Evaluate RHS at current guess
            std::array<Real, N> rhs_val{};
            Problem::rhs(state.t + dt, y_current, rhs_val);
            state.n_rhs++;
            
            // Get Jacobian
            if (state.jacobian_analytic) {
                typename ProblemTraits<Problem>::jacobian_type jac_temp{};
                Problem::jacobian(state.t + dt, y_current, jac_temp);
                state.jacobian = jac_temp;
            } else {
                numerical_jacobian(y_current, state, dt);
            }
            state.n_jac++;
            
            // Set up Newton system: (I - dt*J) * delta_y = -(y_new - y_old - dt*f(y_new))
            for (size_type i = 0; i < N; ++i) {
                for (size_type j = 0; j < N; ++j) {
                    state.jacobian[i][j] *= -dt;
                    if (i == j) state.jacobian[i][j] += 1.0;
                }
            }
            
            // RHS: -(y_current - y_old - dt*rhs)
            std::array<Real, N> newton_rhs{};
            for (size_type i = 0; i < N; ++i) {
                newton_rhs[i] = -(y_current[i] - y_old[i] - dt * rhs_val[i]);
            }
            
            // Solve Newton system
            int ierr;
            if (state.allow_pivoting) {
                ierr = linalg::lu_decomposition<N, true>(state.jacobian, state.pivot);
                if (ierr != 0) return IntegratorResult::LU_DECOMPOSITION_ERROR;
                linalg::lu_solve<N, true>(state.jacobian, state.pivot, newton_rhs);
            } else {
                ierr = linalg::lu_decomposition<N, false>(state.jacobian, state.pivot);
                if (ierr != 0) return IntegratorResult::LU_DECOMPOSITION_ERROR;
                linalg::lu_solve<N, false>(state.jacobian, state.pivot, newton_rhs);
            }
            
            // Update solution
            for (size_type i = 0; i < N; ++i) {
                y_current[i] += newton_rhs[i];
            }
            
            // Check convergence
            Real correction_norm = linalg::norm2(newton_rhs);
            Real y_norm = linalg::norm2(y_current);
            
            if (correction_norm < state.newton_tolerance * std::max(y_norm, 1.0)) {
                converged = true;
                break;
            }
        }
        
        if (!converged) {
            return IntegratorResult::CORRECTOR_CONVERGENCE;
        }
        
        // Update state
        for (size_type i = 0; i < N; ++i) {
            state.y[i] = y_current[i];
        }
        
        return IntegratorResult::SUCCESS;
    }
    
    void numerical_jacobian(const ProblemState& y_current, State& state, Real dt) {
        std::array<Real, N> rhs_base{}, rhs_pert{};
        ProblemState y_pert{};
        
        // Base RHS
        Problem::rhs(state.t + dt, y_current, rhs_base);
        
        // Finite differences
        for (size_type j = 0; j < N; ++j) {
            Real h_fd = std::sqrt(math::UROUND) * std::max(std::abs(y_current[j]), 1.0);
            
            y_pert = y_current;
            y_pert[j] += h_fd;
            
            Problem::rhs(state.t + dt, y_pert, rhs_pert);
            
            for (size_type i = 0; i < N; ++i) {
                state.jacobian[i][j] = (rhs_pert[i] - rhs_base[i]) / h_fd;
            }
        }
        
        state.n_rhs += N + 1;
    }
    
public:
    IntegratorResult integrate(ProblemState& problem_state, State& state) {
        const int max_steps = 10000;
        const Real safety = 1.e-12;
        
        // Initialize
        state.n_step = 0;
        state.n_rhs = 0;
        state.n_jac = 0;
        
        if (std::abs(state.tout - state.t) < safety) {
            return IntegratorResult::SUCCESS;
        }
        
        // Initial timestep
        state.h_current = estimate_initial_timestep(problem_state, state);
        
        // Main integration loop
        while (state.t < state.tout && state.n_step < max_steps) {
            
            // Don't overshoot
            if (state.t + state.h_current > state.tout) {
                state.h_current = state.tout - state.t;
            }
            
            // Save current state for error estimation
            std::array<Real, N> y_save = state.y;
            
            // Try full step
            auto result = take_step(problem_state, state, state.h_current);
            if (result != IntegratorResult::SUCCESS) {
                // Step failed, reduce timestep and try again
                state.y = y_save;
                state.h_current *= 0.5;
                continue;
            }
            
            std::array<Real, N> y_full = state.y;
            
            // Try two half steps for error estimation
            state.y = y_save;
            result = take_step(problem_state, state, state.h_current * 0.5);
            if (result != IntegratorResult::SUCCESS) {
                state.y = y_full;  // Use full step result
            } else {
                result = take_step(problem_state, state, state.h_current * 0.5);
                if (result != IntegratorResult::SUCCESS) {
                    state.y = y_full;  // Use full step result
                } else {
                    // Estimate error
                    Real error = 0.0;
                    for (size_type i = 0; i < N; ++i) {
                        Real weight = 1.0 / (state.rtol * std::abs(state.y[i]) + state.atol);
                        Real diff = std::abs(y_full[i] - state.y[i]);
                        error = std::max(error, weight * diff);
                    }
                    
                    if (error < 0.1) {
                        // Error acceptable, use more accurate result
                        // Adjust timestep for next step
                        state.h_current = std::min(state.h_current * 1.2, 
                                                  2.0 * state.h_current);
                    } else if (error > 1.0) {
                        // Error too large, reduce timestep and retry
                        state.y = y_save;
                        state.h_current *= 0.5;
                        continue;
                    }
                    // else: error acceptable, keep current timestep
                }
            }
            
            // Step succeeded
            state.t += state.h_current;
            state.n_step++;
        }
        
        if (state.n_step >= max_steps) {
            return IntegratorResult::TOO_MANY_STEPS;
        }
        
        return IntegratorResult::SUCCESS;
    }
};

} // namespace integrators

#endif // VODE_HPP