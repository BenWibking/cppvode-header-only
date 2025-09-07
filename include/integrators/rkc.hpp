// ABOUTME: Runge-Kutta-Chebyshev (RKC) integrator implementation
// ABOUTME: Extracted from AMReX Microphysics - explicit method with large stability region
#ifndef RKC_HPP
#define RKC_HPP

#include <array>
#include <cmath>
#include "integrator_types.hpp"

namespace integrators {

template<size_type N>
struct RKCState : public IntegratorState<N> {
    std::array<Real, N> yn{};    // solution at current step
    std::array<Real, N> fn{};    // RHS at current step  
    std::array<Real, N> yjm1{};  // work arrays
    std::array<Real, N> yjm2{};
    std::array<Real, N> sprad{}; // spectral radius estimation work
    
    Real hmax{0.0};
    int n_accept{0};
    int n_reject{0};
    int n_sprad{0};
    int max_stages{0};
    bool use_circle_theorem{false};
};

template<typename Problem>
class RKC {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = RKCState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;
    
private:
    // Take an RKC step with m stages
    void take_step(ProblemState& problem_state, State& state, Real h, int m) {
        // RKC parameters
        Real w0 = 1.0 + 2.0 / (13.0 * m * m);
        Real temp1 = w0 * w0 - 1.0;
        Real temp2 = std::sqrt(temp1);
        Real arg = m * std::log(w0 + temp2);
        Real w1 = std::sinh(arg) * temp1 / 
                  (std::cosh(arg) * m * temp2 - w0 * std::sinh(arg));
        Real bjm1 = 1.0 / math::powi<2>(2.0 * w0);
        Real bjm2 = bjm1;
        
        // First stage
        for (size_type i = 0; i < N; ++i) {
            state.yjm2[i] = state.yn[i];
        }
        Real mus = w1 * bjm1;
        for (size_type i = 0; i < N; ++i) {
            state.yjm1[i] = state.yn[i] + h * mus * state.fn[i];
        }
        
        Real thjm2 = 0.0;
        Real thjm1 = mus;
        Real zjm1 = w0;
        Real zjm2 = 1.0;
        Real dzjm1 = 1.0;
        Real dzjm2 = 0.0;
        Real d2zjm1 = 0.0;
        Real d2zjm2 = 0.0;
        
        // Stages 2 through m
        for (int j = 2; j <= m; ++j) {
            Real zj = 2.0 * w0 * zjm1 - zjm2;
            Real dzj = 2.0 * w0 * dzjm1 - dzjm2 + 2.0 * zjm1;
            Real d2zj = 2.0 * w0 * d2zjm1 - d2zjm2 + 4.0 * dzjm1;
            Real bj = d2zj / math::powi<2>(dzj);
            Real ajm1 = 1.0 - zjm1 * bjm1;
            Real mu = 2.0 * w0 * bj / bjm1;
            Real nu = -bj / bjm2;
            mus = mu * w1 / w0;
            
            // Evaluate RHS at intermediate stage
            ProblemState y_current{};
            for (size_type i = 0; i < N; ++i) {
                y_current[i] = state.yjm1[i];
            }
            
            std::array<Real, N> ydot{};
            Problem::rhs(state.t + h * thjm1, y_current, ydot);
            
            // Compute next stage
            for (size_type i = 0; i < N; ++i) {
                state.y[i] = mu * state.yjm1[i] + 
                            nu * state.yjm2[i] + 
                            (1.0 - mu - nu) * state.yn[i] + 
                            h * mus * (ydot[i] - ajm1 * state.fn[i]);
            }
            Real thj = mu * thjm1 + nu * thjm2 + mus * (1.0 - ajm1);
            
            // Shift data for next stage
            if (j < m) {
                for (size_type i = 0; i < N; ++i) {
                    state.yjm2[i] = state.yjm1[i];
                    state.yjm1[i] = state.y[i];
                }
                thjm2 = thjm1;
                thjm1 = thj;
                bjm2 = bjm1;
                bjm1 = bj;
                zjm2 = zjm1;
                zjm1 = zj;
                dzjm2 = dzjm1;
                dzjm1 = dzj;
                d2zjm2 = d2zjm1;
                d2zjm1 = d2zj;
            }
        }
    }
    
    // Estimate spectral radius using power iteration
    Real estimate_spectral_radius(ProblemState& problem_state, State& state, Real hmax) {
        const Real tol = 0.01;
        const int max_iter = 50;
        
        // Initialize with random-like vector
        for (size_type i = 0; i < N; ++i) {
            state.sprad[i] = 1.0 + 0.1 * std::sin(static_cast<Real>(i));
        }
        
        Real sigma = 0.0;
        Real sigma_old = -1.0;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // Normalize
            Real norm = 0.0;
            for (size_type i = 0; i < N; ++i) {
                norm += state.sprad[i] * state.sprad[i];
            }
            norm = std::sqrt(norm);
            
            if (norm == 0.0) return 0.0;
            
            for (size_type i = 0; i < N; ++i) {
                state.sprad[i] /= norm;
            }
            
            // Apply finite difference approximation to J*v
            Real h_fd = std::sqrt(math::UROUND);
            std::array<Real, N> rhs_base{}, rhs_pert{};
            ProblemState y_current{}, y_pert{};
            
            // Convert current state
            for (size_type i = 0; i < N; ++i) {
                y_current[i] = state.y[i];
            }
            
            Problem::rhs(state.t, y_current, rhs_base);
            
            // Perturbed state
            for (size_type i = 0; i < N; ++i) {
                y_pert[i] = y_current[i] + h_fd * state.sprad[i];
            }
            
            Problem::rhs(state.t, y_pert, rhs_pert);
            state.n_rhs += 2;
            
            // J*v approximation
            std::array<Real, N> Jv{};
            for (size_type i = 0; i < N; ++i) {
                Jv[i] = (rhs_pert[i] - rhs_base[i]) / h_fd;
            }
            
            
            // Compute eigenvalue estimate
            Real num = 0.0, den = 0.0;
            for (size_type i = 0; i < N; ++i) {
                num += state.sprad[i] * Jv[i];
                den += state.sprad[i] * state.sprad[i];
            }
            
            sigma = (den > 0.0) ? num / den : 0.0;
            state.sprad = Jv; // Update for next iteration
            
            if (iter > 0 && std::abs(sigma - sigma_old) < tol * std::abs(sigma)) {
                break;
            }
            sigma_old = sigma;
        }
        
        return std::abs(sigma);
    }
    
    // Initial timestep estimation
    Real estimate_initial_timestep(ProblemState& problem_state, State& state, 
                                   Real hmax, Real sprad) {
        Real hmin = 10.0 * math::UROUND * std::max(std::abs(state.t), hmax);
        Real absh = hmax;
        
        if (sprad * absh > 1.0) {
            absh = 1.0 / sprad;
        }
        absh = std::max(absh, hmin);
        
        // Test step
        ProblemState y_test{};
        for (size_type i = 0; i < N; ++i) {
            y_test[i] = state.yn[i] + absh * state.fn[i];
        }
        
        std::array<Real, N> rhs_test{};
        Problem::rhs(state.t + absh, y_test, rhs_test);
        state.n_rhs++;
        
        // Estimate local truncation error
        Real est = 0.0;
        for (size_type i = 0; i < N; ++i) {
            Real wt = state.rtol * std::max(std::abs(state.yn[i]), 
                                           std::abs(state.yjm1[i])) + state.atol;
            Real diff = 0.8 * (state.yn[i] - state.yjm1[i]) + 
                       0.4 * absh * (state.fn[i] + rhs_test[i]);
            est = std::max(est, std::abs(diff / wt));
        }
        
        if (est > 1.0) {
            absh = absh * std::pow(est, -1.0/3.0);
        }
        
        return std::max(absh, hmin);
    }
    
public:
    IntegratorResult integrate(ProblemState& problem_state, State& state) {
        const Real rmax = 0.1;
        const Real rmin = 10.0 * math::UROUND;
        const int max_steps = 10000;
        
        // Validate inputs
        if (state.rtol > rmax || state.rtol < rmin || state.atol < 0.0) {
            return IntegratorResult::BAD_INPUTS;
        }
        
        // Initialize counters
        state.n_step = 0;
        state.n_rhs = 0;
        state.n_accept = 0;
        state.n_reject = 0;
        state.n_sprad = 0;
        state.max_stages = 0;
        
        // Maximum number of stages
        int mmax = static_cast<int>(std::round(std::sqrt(state.rtol / (10.0 * math::UROUND))));
        mmax = std::max(mmax, 2);
        
        bool need_sprad = true;
        int nstsig = 0;
        
        // Initialize solution
        for (size_type i = 0; i < N; ++i) {
            state.yn[i] = state.y[i];
        }
        
        // Initial RHS evaluation
        ProblemState y_init{};
        for (size_type i = 0; i < N; ++i) {
            y_init[i] = state.yn[i];
        }
        Problem::rhs(state.t, y_init, state.fn);
        state.n_rhs++;
        
        Real tdir = std::copysign(1.0, state.tout - state.t);
        state.hmax = std::abs(state.tout - state.t);
        
        Real sprad = 0.0;
        Real absh = 0.0;
        Real errold = 0.0;
        Real hold = 0.0;
        
        // Main integration loop
        while (state.n_step < max_steps) {
            
            // Estimate spectral radius if needed
            if (need_sprad) {
                sprad = estimate_spectral_radius(problem_state, state, state.hmax);
                state.n_sprad++;
            }
            
            // Initial timestep estimate
            if (state.n_step == 0) {
                absh = estimate_initial_timestep(problem_state, state, state.hmax, sprad);
            }
            
            // Adjust timestep and determine number of stages
            bool last = false;
            if (1.1 * absh >= std::abs(state.tout - state.t)) {
                absh = std::abs(state.tout - state.t);
                last = true;
            }
            
            int m = 1 + static_cast<int>(std::sqrt(1.54 * absh * sprad + 1.0));
            if (m > mmax) {
                m = mmax;
                absh = static_cast<Real>(m * m - 1) / (1.54 * sprad);
                last = false;
            }
            state.max_stages = std::max(m, state.max_stages);
            
            // Take step
            Real h = tdir * absh;
            Real hmin = 10.0 * math::UROUND * std::max(std::abs(state.t), 
                                                      std::abs(state.t + h));
            
            take_step(problem_state, state, h, m);
            ProblemState y_final{};
            for (size_type i = 0; i < N; ++i) {
                y_final[i] = state.y[i];
            }
            Problem::rhs(state.t + h, y_final, state.yjm1);
            state.n_rhs += m;
            state.n_step++;
            
            // Error estimation
            Real err = 0.0;
            for (size_type i = 0; i < N; ++i) {
                Real wt = state.rtol * std::max(std::abs(state.yn[i]), 
                                               std::abs(state.y[i])) + state.atol;
                Real est = 0.8 * (state.yn[i] - state.y[i]) + 
                          0.4 * h * (state.fn[i] + state.yjm1[i]);
                err = std::max(err, math::powi<2>(est / wt));
            }
            err = std::sqrt(err / N);
            
            if (err > 1.0) {
                // Reject step
                state.n_reject++;
                absh = 0.8 * absh / std::cbrt(err);
                if (absh < hmin) {
                    return IntegratorResult::DT_UNDERFLOW;
                }
                need_sprad = true;
                continue;
            }
            
            // Accept step
            state.n_accept++;
            state.t += h;
            need_sprad = false;
            nstsig = (nstsig + 1) % 25;
            if (nstsig == 0) need_sprad = true;
            
            // Update solution
            for (size_type i = 0; i < N; ++i) {
                Real ylast = state.yn[i];
                Real yplast = state.fn[i];
                state.yn[i] = state.y[i];
                state.fn[i] = state.yjm1[i];
                state.yjm1[i] = ylast;
                state.yjm2[i] = yplast;
            }
            
            // Timestep adjustment for next step
            Real fac = 10.0;
            if (state.n_accept == 1) {
                Real temp2 = std::cbrt(err);
                if (0.8 < fac * temp2) fac = 0.8 / temp2;
            } else {
                Real temp1 = 0.8 * absh * std::cbrt(errold);
                Real temp2 = std::abs(hold) * math::powi<2>(std::cbrt(err));
                if (temp1 < fac * temp2) fac = temp1 / temp2;
            }
            absh = std::max(0.1, fac) * absh;
            absh = math::clamp(absh, hmin, state.hmax);
            errold = err;
            hold = h;
            
            if (last) {
                // Copy final solution
                state.y = state.yn;
                return IntegratorResult::SUCCESS;
            }
        }
        
        return IntegratorResult::TOO_MANY_STEPS;
    }
};

} // namespace integrators

#endif // RKC_HPP