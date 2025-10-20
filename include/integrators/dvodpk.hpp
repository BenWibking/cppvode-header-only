// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: DVODPK integrator (BDF with preconditioned GMRES) C++ port scaffold
#ifndef DVODPK_HPP
#define DVODPK_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#define INTEGRATORS_DVODPK_DEBUG
#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

// Configuration options for the DVODPK integrator.
struct DVODPKConfig {
    int max_order{5};                 // Maximum BDF order allowed (1-5)
    int max_steps{1000};              // Global step limit
    int max_error_failures{10};       // Consecutive error test failures before giving up
    int max_nonlinear_iters{3};       // Newton corrector iterations per step
    int max_krylov_iters{20};         // GMRES subspace dimension (MAXL in DVODPK)
    int max_krylov_restarts{3};       // Number of GMRES restarts allowed
    int max_consecutive_corrector_failures{10}; // MXNCF analogue
    int max_steps_between_jacobian{50};         // MSBJ analogue
    int max_steps_between_preconditioner{20};   // MSBP analogue

    Real krylov_rtol{1.0e-6};         // Relative tolerance for GMRES solves
    Real krylov_atol{1.0e-12};        // Absolute tolerance for GMRES solves
    Real h_min{0.0};                  // Minimum step size (0 => disabled)
    Real h_max{0.0};                  // Maximum step size (0 => disabled)

    // Legacy DVODPK tuning constants.
    Real eta_cf{0.25};                // Step reduction factor after corrector failure (ETACF)
    Real eta_min{0.1};                // Minimum shrink factor (ETAMIN)
    Real eta_max_fail{0.2};           // Max shrink factor after many failures (ETAMXF)
    Real eta_max_first{1.0e4};        // Max growth on the very first successful step (ETAMX1)
    Real eta_max_early{10.0};         // Max growth during initial steps (ETAMX2)
    Real eta_max_late{10.0};          // Max growth in later steps (ETAMX3)
    Real order_change_threshold{1.5}; // Minimum eta to allow order/step increase (THRESH)
    Real bias1{6.0};                  // DSM bias for q-1 candidate
    Real bias2{6.0};                  // DSM bias for current order
    Real bias3{10.0};                 // DSM bias for q+1 candidate
    Real addon{1.0e-6};               // ADDON regularisation term
    Real hmin_relax{1.00001};         // ONEPSM factor for HMIN checks
    int kflag_error_threshold{-3};    // KFC threshold for drastic recovery
    int kflag_failure_stop{-7};       // KFH threshold for hard failure
    Real rc_update_threshold{0.3};    // CCMAX analogue
    bool enable_trace{false};         // Enable per-step tracing
    std::string trace_filename{"dvodpk_trace.log"};
    bool trace_append{false};

    bool use_analytic_jacobian{true}; // Prefer Problem::jacobian when available
    bool enable_preconditioner{true}; // Allow Problem-provided preconditioners
    PreconditionerSide preconditioner_side{PreconditionerSide::RIGHT};
};

// State container holding persistent DVODPK data across integration calls.
template<size_type N, typename Preconditioner = IdentityPreconditioner>
struct DVODPKState : public IntegratorState<N> {
    // Method order and step control.
    int nq{1};             // Current method order
    int l{2};              // L = nq + 1
    short newq{1};         // Desired order after successful step
    short newh{0};         // Flag for pending step size change
    short nqwait{2};       // Counter controlling order changes
    short icf{0};          // Corrector convergence flag
    short ipup{1};         // Jacobian/preconditioner update request
    short jcur{0};         // Jacobian currently evaluated at tn?
    short kflag{0};        // Step completion flag (DVSTEP analogue)

    Real h{0.0};           // Current step size
    Real h_used{0.0};      // Last successful step size
    Real hmxi{0.0};        // Inverse of HMAX (0 => disabled)
    Real hmin{0.0};        // Minimum allowed step size (absolute)
    Real hscal{0.0};       // Step size scale for predictors
    Real eta{1.0};         // Step size ratio
    Real conp{0.0};        // Error test constant
    Real crate{1.0};       // Convergence rate estimate
    Real rl1{1.0};         // Reciprocal coefficient used in linear systems
    Real pdnorm{0.0};      // Norm of preconditioned residual
    Real etamax{10.0};     // Max allowed eta growth factor
    Real rc{0.0};          // Ratio of new to old step
    Real drc{0.0};         // Change in rc for Jacobian updates
    Real prl1{1.0};        // Predictor constant
    Real tn{0.0};          // Current internal time
    Real acnrm_last{0.0};  // Last ACNRM value

    // Counters mirroring DVODPK statistics.
    int max_steps{1000};
    int n_error_fail{0};   // Consecutive error test failures
    int n_linear_conv_fail{0};
    int n_linear_iters{0}; // Accumulated Krylov iterations
    int ncf{0};            // Consecutive nonlinear failures
    int netf{0};           // Total error test failures
    int nni{0};            // Total nonlinear iterations
    int nls{0};            // Total linear solves
    int steps_since_jacobian{0};
    int steps_since_preconditioner{0};

    // Flags for Jacobian/preconditioner freshness.
    bool jacobian_is_current{false};
    bool preconditioner_is_current{false};
    bool trace_enabled{false};
    std::ofstream trace_stream;

    // Error weights and working vectors.
    std::array<Real, N> ewt{};
    std::array<Real, N> savf{};
    std::array<Real, N> acor{};
    std::array<Real, N> y_temp{};
    std::array<Real, N> rhs_temp{};

    // Nordsieck history array YH(:, j), j=1..max_order+1.
    std::array<std::array<Real, 6>, N> yh{};
    std::array<Real, 6> el{};    // Coefficient array EL(1..L)
    std::array<Real, 6> tau{};   // Step size history TAU(1..L)
    std::array<Real, 5> tq{};    // Error weight factors TQ(1..5)

    // Krylov workspace (Arnoldi basis).
    std::vector<std::array<Real, N>> v;        // Basis vectors V(:,1..max_krylov_iters+1)
    std::vector<std::array<Real, N>> z;        // Preconditioned vectors
    std::vector<Real> givens_cos;              // Cosines for Givens rotations
    std::vector<Real> givens_sin;              // Sines for Givens rotations
    std::vector<Real> hes;                     // Upper Hessenberg entries (stored column-major)
    std::vector<Real> residual;                // GMRES residual history

    Preconditioner preconditioner{};
    std::array<std::array<Real, N>, N> jacobian{};

    // 1-based indexing helpers mimicking the Fortran layout.
    inline Real& EL(int i) { return el[static_cast<std::size_t>(i - 1)]; }
    inline const Real& EL(int i) const { return el[static_cast<std::size_t>(i - 1)]; }
    inline Real& TAU(int i) { return tau[static_cast<std::size_t>(i - 1)]; }
    inline const Real& TAU(int i) const { return tau[static_cast<std::size_t>(i - 1)]; }
    inline Real& TQ(int i) { return tq[static_cast<std::size_t>(i - 1)]; }
    inline const Real& TQ(int i) const { return tq[static_cast<std::size_t>(i - 1)]; }
    inline std::array<Real, 6>& column(int row) { return yh[static_cast<std::size_t>(row - 1)]; }
    inline const std::array<Real, 6>& column(int row) const { return yh[static_cast<std::size_t>(row - 1)]; }
    inline Real& YH(int i, int j) { return column(i)[static_cast<std::size_t>(j - 1)]; }
    inline const Real& YH(int i, int j) const { return column(i)[static_cast<std::size_t>(j - 1)]; }

    // Helper to resize Krylov buffers when configuration changes.
    void ensure_krylov_capacity(int maxl) {
        const std::size_t desired = static_cast<std::size_t>(maxl + 1);
        if (v.size() < desired) v.resize(desired);
        if (z.size() < desired) z.resize(desired);
        const std::size_t hes_size = static_cast<std::size_t>((maxl + 1) * maxl);
        if (hes.size() < hes_size) hes.resize(hes_size);
        if (givens_cos.size() < static_cast<std::size_t>(maxl)) givens_cos.resize(static_cast<std::size_t>(maxl));
        if (givens_sin.size() < static_cast<std::size_t>(maxl)) givens_sin.resize(static_cast<std::size_t>(maxl));
        if (residual.size() < desired) residual.resize(desired);
    }

    // Reset step statistics before a new integration attempt.
    void reset_statistics() {
        this->n_step = 0;
        this->n_rhs = 0;
        this->n_jac = 0;
        n_error_fail = 0;
        n_linear_conv_fail = 0;
        n_linear_iters = 0;
        icf = 0;
        ipup = 1;
        jcur = 0;
        nq = 1;
        l = 2;
        newq = 1;
        newh = 0;
        nqwait = 2;
        crate = 1.0;
        conp = 0.0;
        rc = 0.0;
        drc = 0.0;
        prl1 = 1.0;
        etamax = 10.0;
        eta = 1.0;
        h = 0.0;
        h_used = 0.0;
        hscal = 0.0;
        hmxi = 0.0;
        hmin = 0.0;
        rl1 = 1.0;
        pdnorm = 0.0;
        tn = 0.0;
        acnrm_last = 0.0;
        jacobian_is_current = false;
        preconditioner_is_current = false;
        ncf = 0;
        netf = 0;
        nni = 0;
        nls = 0;
        steps_since_jacobian = 0;
        steps_since_preconditioner = 0;
    }
};

template<typename Problem>
class DVODPK {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = DVODPKState<N, typename ProblemTraits<Problem>::preconditioner_type>;
    using Config = DVODPKConfig;
    using ProblemState = typename ProblemTraits<Problem>::state_type;

    struct InitialStepResult {
        Real h0{0.0};
        int iterations{0};
        bool success{false};
    };

    static constexpr bool has_problem_jacobian = JacobianFunction<Problem>;

    static inline void rhs(Real t, const std::array<Real, N>& y, std::array<Real, N>& out) {
        Problem::rhs(t, y, out);
    }

    static inline void jacobian(Real t, const std::array<Real, N>& y, std::array<std::array<Real, N>, N>& J) {
        if constexpr (requires { Problem::jacobian(t, y, J); }) {
            Problem::jacobian(t, y, J);
        } else {
            (void)t; (void)y; (void)J;
        }
    }

    static IntegratorResult integrate(State& state, Problem& problem, Real tout) {
        return integrate(state, problem, tout, Config{});
    }

    static IntegratorResult integrate(State& state, Problem& problem, Real tout, const Config& config) {
        state.ensure_krylov_capacity(config.max_krylov_iters);
        state.max_steps = config.max_steps;
        state.reset_statistics();
        if (config.enable_trace) {
            state.trace_enabled = true;
            if (state.trace_stream.is_open()) {
                state.trace_stream.close();
            }
            std::ios::openmode mode = std::ios::out;
            if (config.trace_append) {
                mode |= std::ios::app;
            }
            state.trace_stream.open(config.trace_filename, mode);
            if (state.trace_stream.good() && !config.trace_append) {
                state.trace_stream << "# tag NST NQ NEWQ H ETA ETAMAX IPUP JCUR KFLAG ICF NQWAIT STEPSJ STEPSP\n";
            }
        } else {
            state.trace_enabled = false;
            if (state.trace_stream.is_open()) {
                state.trace_stream.close();
            }
        }
        IntegratorResult result = advance_to(state, problem, tout, config);
        if (state.trace_stream.is_open()) {
            state.trace_stream.flush();
        }
        return result;
    }

private:
    struct KrylovResult {
        bool converged{false};
        int iterations{0};
        Real residual_norm{0.0};
    };

    template<typename P = Problem>
    static constexpr bool has_preconditioner_setup_v = [] {
        if constexpr (!ProblemTraits<P>::has_custom_preconditioner) {
            return false;
        } else {
            using Precond = typename ProblemTraits<P>::preconditioner_type;
            return requires(Real t, const ProblemState& state_vec, Real gamma, Precond& precond) {
                P::preconditioner_setup(t, state_vec, gamma, precond);
            };
        }
    }();

    template<typename P = Problem>
    static constexpr bool has_preconditioner_apply_v = [] {
        if constexpr (!ProblemTraits<P>::has_custom_preconditioner) {
            return false;
        } else {
            using Precond = typename ProblemTraits<P>::preconditioner_type;
            return requires(const Precond& precond, std::array<Real, N>& vec, PreconditionerSide side) {
                P::apply_preconditioner(precond, vec, side);
            };
        }
    }();

    static Real weighted_rms_norm(const std::array<Real, N>& vec, const std::array<Real, N>& ewt) {
        Real sum = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real term = vec[i] * ewt[i];
            sum += term * term;
        }
        return std::sqrt(sum / static_cast<Real>(N));
    }

    template<typename Vector>
    static Real weighted_rms_norm(const Vector& vec, const std::array<Real, N>& ewt) {
        Real sum = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real term = vec[static_cast<size_type>(i)] * ewt[i];
            sum += term * term;
        }
        return std::sqrt(sum / static_cast<Real>(N));
    }

    static bool update_error_weights(State& state) {
        bool ok = true;
        for (size_type i = 0; i < N; ++i) {
            const Real weight = state.rtol * std::abs(state.YH(static_cast<int>(i + 1), 1)) + state.atol;
            if (weight <= 0.0) {
                ok = false;
                state.ewt[i] = 1.0;
            } else {
                state.ewt[i] = 1.0 / weight;
            }
        }
        return ok;
    }

    static IntegratorResult advance_to(State& state, Problem& problem, Real tout, const Config& config) {
        state.tout = tout;
        if (std::abs(tout - state.t) < std::numeric_limits<Real>::epsilon()) {
            return IntegratorResult::SUCCESS;
        }
        state.tn = state.t;

        // Initial RHS evaluation.
        rhs(state.t, state.y, state.savf);
        state.n_rhs += 1;

        for (size_type i = 0; i < N; ++i) {
            state.YH(static_cast<int>(i + 1), 1) = state.y[i];
            state.YH(static_cast<int>(i + 1), 2) = state.savf[i];
        }

        state.nq = 1;
        state.l = state.nq + 1;
        state.newq = static_cast<short>(state.nq);
        state.jacobian_analytic = has_problem_jacobian && config.use_analytic_jacobian;

        if (!update_error_weights(state)) {
            return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
        }

        const InitialStepResult init = compute_initial_step(state, problem, config);
        state.n_rhs += init.iterations;
        if (!init.success) {
            return IntegratorResult::DT_UNDERFLOW;
        }

        state.h = init.h0;
        if (state.h == 0.0) {
            return IntegratorResult::DT_UNDERFLOW;
        }
        state.TAU(1) = state.h;
        for (int j = 2; j <= state.l; ++j) {
            state.TAU(j) = state.h;
        }
        for (size_type i = 0; i < N; ++i) {
            state.YH(static_cast<int>(i + 1), 2) *= state.h;
        }
        set_coefficients(state);

        state.h_used = state.h;
        state.hscal = state.h;
        state.hmxi = (config.h_max > 0.0) ? (1.0 / config.h_max) : 0.0;
        state.hmin = config.h_min;
        state.eta = 1.0;
        state.etamax = config.eta_max_first;
        state.rc = 0.0;
        state.drc = 0.0;
        state.prl1 = 1.0;
        state.newh = 0;
        state.nqwait = 2;
        state.icf = 0;
        state.ipup = 1;
        state.jcur = 0;
        state.tn = state.t;
        state.jacobian_is_current = false;
        state.preconditioner_is_current = false;

        // Main advance loop (skeleton).
        while (true) {
            if (state.n_step >= state.max_steps) {
                for (size_type i = 0; i < N; ++i) {
                    state.y[i] = state.YH(static_cast<int>(i + 1), 1);
                }
                state.t = state.tn;
                return IntegratorResult::TOO_MANY_STEPS;
            }

            if (!update_error_weights(state)) {
                for (size_type i = 0; i < N; ++i) {
                    state.y[i] = state.YH(static_cast<int>(i + 1), 1);
                }
                state.t = state.tn;
                return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
            }

            Real tolsf_norm = 0.0;
            for (size_type i = 0; i < N; ++i) {
                const Real term = state.YH(static_cast<int>(i + 1), 1) * state.ewt[i];
                tolsf_norm += term * term;
            }
            const Real tolsf = math::UROUND * std::sqrt(tolsf_norm / static_cast<Real>(N));
            if (tolsf > 1.0) {
                if (state.n_step == 0) {
                    return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
                }
                for (size_type i = 0; i < N; ++i) {
                    state.y[i] = state.YH(static_cast<int>(i + 1), 1);
                }
                state.t = state.tn;
                return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
            }

            const IntegratorResult step_result = step_once(state, problem, config);
            if (step_result != IntegratorResult::SUCCESS) {
                return step_result;
            }

            if ((state.tn - state.tout) * state.h < 0.0) {
                continue;
            }

            // Interpolate current solution to tout (linear for now).
            for (size_type i = 0; i < N; ++i) {
                state.y[i] = state.YH(static_cast<int>(i + 1), state.l);
            }
            const Real S = (state.tout - state.tn) / state.h;
            for (int jb = 1; jb <= state.nq; ++jb) {
                const int j = state.nq - jb;
                for (size_type i = 0; i < N; ++i) {
                    state.y[i] = state.YH(static_cast<int>(i + 1), j + 1) + S * state.y[i];
                }
            }
            state.t = state.tout;
            return IntegratorResult::SUCCESS;
        }
    }

    static void set_coefficients(State& state) {
        constexpr Real CORTES = static_cast<Real>(0.1);
        if (state.nq < 1) {
            state.nq = 1;
        }
        state.l = static_cast<short>(state.nq + 1);
        const int l = state.l;
        const Real flotl = static_cast<Real>(l);

        for (int i = 3; i <= l; ++i) {
            state.EL(i) = 0.0;
        }
        state.EL(1) = 1.0;
        state.EL(2) = 1.0;

        Real alph0 = -1.0;
        Real ahatn0 = -1.0;
        Real hsum = state.h;
        Real rxi = 1.0;
        Real rxis = 1.0;

        if (state.nq != 1) {
            const int nqm2 = state.nq - 2;
            for (int j = 1; j <= nqm2; ++j) {
                hsum += state.TAU(j);
                rxi = state.h / hsum;
                alph0 -= 1.0 / static_cast<Real>(j + 1);
                for (int iback = 1; iback <= j + 1; ++iback) {
                    const int idx = (j + 3) - iback;
                    state.EL(idx) += state.EL(idx - 1) * rxi;
                }
            }
            alph0 -= 1.0 / static_cast<Real>(state.nq);
            rxis = -state.EL(2) - alph0;
            hsum += state.TAU(state.nq - 1);
            rxi = state.h / hsum;
            ahatn0 = -state.EL(2) - rxi;
            for (int iback = 1; iback <= state.nq; ++iback) {
                const int idx = (state.nq + 2) - iback;
                state.EL(idx) += state.EL(idx - 1) * rxis;
            }
        }

        const Real t1 = 1.0 - ahatn0 + alph0;
        const Real t2 = 1.0 + static_cast<Real>(state.nq) * t1;
        state.TQ(2) = std::abs(alph0 * t2 / t1);
        const Real denom = state.EL(l) * rxi / rxis;
        state.TQ(5) = (denom != 0.0) ? std::abs(t2 / denom) : 0.0;

        if (state.nqwait == 1 && state.nq > 1) {
            const Real cnqm1 = rxis / state.EL(l);
            const Real t3 = alph0 + 1.0 / static_cast<Real>(state.nq);
            const Real t4 = ahatn0 + rxi;
            const Real denom1 = 1.0 - t4 + t3;
            const Real elp1 = (denom1 != 0.0) ? (t3 / denom1) : 0.0;
            state.TQ(1) = (cnqm1 != 0.0) ? std::abs(elp1 / cnqm1) : 0.0;

            hsum += state.TAU(state.nq);
            rxi = state.h / hsum;
            const Real t5 = alph0 - 1.0 / static_cast<Real>(state.nq + 1);
            const Real t6 = ahatn0 - rxi;
            const Real denom2 = 1.0 - t6 + t5;
            const Real elp2 = (denom2 != 0.0) ? (t2 / denom2) : 0.0;
            state.TQ(3) = std::abs(elp2 * rxi * (flotl + 1.0) * t5);
        }

        if (state.nq <= 1) {
            state.TQ(1) = 0.0;
            state.TQ(3) = 0.0;
        }

        state.TQ(4) = CORTES * state.TQ(2);
    }

    static InitialStepResult compute_initial_step(State& state, Problem& problem, const Config& config) {
        (void)problem;
        const Real t0 = state.t;
        const Real tout = state.tout;
        const Real uround = math::UROUND;
        InitialStepResult result{};

        const Real tdist = std::abs(tout - t0);
        const Real tround = uround * std::max(std::abs(t0), std::abs(tout));
        if (tdist < 2.0 * tround) {
            result.success = false;
            return result;
        }

        Real hlb = 100.0 * tround;
        Real hub = 0.1 * tdist;

        for (size_type i = 0; i < N; ++i) {
            const Real atol_i = state.atol; // Scalar tolerance for now.
            const Real delyi = 0.1 * std::abs(state.y[i]) + atol_i;
            const Real afi = std::abs(state.savf[i]);
            if (afi > 0.0) {
                const Real bound = delyi / afi;
                if (afi * hub > delyi) {
                    hub = std::max(bound, std::numeric_limits<Real>::min());
                }
            }
        }

        if (config.h_min > 0.0) {
            hlb = std::max(hlb, config.h_min);
        }
        if (config.h_max > 0.0) {
            hub = std::min(hub, config.h_max);
        }

        Real hg = std::sqrt(std::max(hlb * hub, std::numeric_limits<Real>::min()));
        if (hub < hlb) {
            result.h0 = std::copysign(hg, tout - t0);
            result.success = true;
            return result;
        }

        constexpr Real half = 0.5;
        constexpr Real two = 2.0;
        Real hnew = hg;
        int iter = 0;

        while (true) {
            const Real htrial = std::copysign(hg, tout - t0);
            const Real t1 = t0 + htrial;
            for (size_type i = 0; i < N; ++i) {
                state.y_temp[i] = state.y[i] + htrial * state.savf[i];
            }
            rhs(t1, state.y_temp, state.rhs_temp);
            ++iter;
            for (size_type i = 0; i < N; ++i) {
                state.rhs_temp[i] = (state.rhs_temp[i] - state.savf[i]) / htrial;
            }
            const Real yddnrm = weighted_rms_norm(state.rhs_temp, state.ewt);
            if (yddnrm * hub * hub > two) {
                hnew = std::sqrt(two / std::max(yddnrm, std::numeric_limits<Real>::min()));
            } else {
                hnew = std::sqrt(hg * hub);
            }

            if (iter >= 4) {
                break;
            }
            const Real hrat = (hg > 0.0) ? (hnew / hg) : two;
            if (hrat > half && hrat < two) {
                break;
            }
            if (iter >= 2 && hnew > two * hg) {
                hnew = hg;
                break;
            }
            hg = hnew;
        }

        Real h0 = half * hnew;
        if (h0 < hlb) h0 = hlb;
        if (hub > 0.0 && h0 > hub) h0 = hub;
        h0 = std::copysign(h0, tout - t0);
        result.h0 = h0;
        result.iterations = iter;
        result.success = true;
        return result;
    }

    struct NewtonResult {
        bool converged{false};
        Real correction_norm{0.0};
        Real residual_norm{0.0};
        int iterations{0};
    };

    static NewtonResult newton_solve(State& state,
                                     Problem& problem,
                                     Real h,
                                     Real t_old,
                                     const std::array<Real, N>& y_old,
                                     const Config& config) {
        NewtonResult result{};
        const Real t_next = t_old + h;
        std::array<Real, N> residual_vec{};

        for (size_type i = 0; i < N; ++i) {
            state.y_temp[i] = y_old[i];
            state.acor[i] = 0.0;
        }

        Real previous_correction_norm = 1.0;

        const bool use_preconditioner = config.enable_preconditioner && ProblemTraits<Problem>::has_custom_preconditioner;

        for (int iter = 0; iter < config.max_nonlinear_iters; ++iter) {
            rhs(t_next, state.y_temp, state.savf);
            state.n_rhs++;

            for (size_type i = 0; i < N; ++i) {
                residual_vec[i] = state.y_temp[i] - y_old[i] - h * state.savf[i];
            }
            result.residual_norm = weighted_rms_norm(residual_vec, state.ewt);

            if (!state.jacobian_is_current) {
                evaluate_jacobian(state, problem, t_next, state.y_temp, config);
                state.jacobian_is_current = true;
            }

            state.rl1 = h;

            std::array<Real, N> delta{};
            for (size_type i = 0; i < N; ++i) {
                delta[i] = -residual_vec[i];
            }

            if (use_preconditioner && (!state.preconditioner_is_current || state.ipup != 0)) {
                setup_preconditioner(state, problem, h, config);
            }

            state.nls += 1;
            KrylovResult lin_res = solve_linear_system(state, delta, h, config);
            if (!lin_res.converged) {
                state.icf = 2;
                state.jacobian_is_current = false;
                result.converged = false;
                return result;
            }
            state.icf = (lin_res.iterations > config.max_krylov_iters / 2) ? 1 : 0;
            state.n_linear_iters += lin_res.iterations;

            for (size_type i = 0; i < N; ++i) {
                state.y_temp[i] += delta[i];
                state.acor[i] += delta[i];
            }

            result.correction_norm = weighted_rms_norm(state.acor, state.ewt);
            state.crate = (iter == 0) ? 1.0
                                      : std::max<Real>(0.01, result.correction_norm / std::max(previous_correction_norm, math::UROUND));
            previous_correction_norm = std::max(result.correction_norm, math::UROUND);

            if (result.correction_norm <= 1.0) {
                result.converged = true;
                result.iterations = iter + 1;
                break;
            }
        }

        if (result.converged) {
            rhs(t_next, state.y_temp, state.savf);
            state.n_rhs++;
        }

        return result;
    }

    static bool adjust_step_down(State& state,
                                 Real& h,
                                 Real error_norm,
                                 const Config& config,
                                 std::optional<Real> forced_eta = std::nullopt) {
        const Real original_h = h;
        if (original_h == 0.0) {
            return false;
        }
        const Real order = std::max<Real>(1.0, static_cast<Real>(state.nq));
        const Real safety = 0.9;
        const Real min_factor = 0.1;
        const Real max_factor = 0.5;
        const Real err = std::max(error_norm, static_cast<Real>(1.0e-4));
        Real factor = 1.0;
        if (forced_eta.has_value()) {
            factor = std::max(forced_eta.value(), static_cast<Real>(0.0));
        } else {
            factor = safety / std::pow(err, 1.0 / order);
            factor = math::clamp(factor, min_factor, max_factor);
        }
        Real new_h = original_h * factor;
        if (config.h_min > 0.0 && std::abs(new_h) < config.h_min) {
            new_h = std::copysign(config.h_min, new_h == 0.0 ? original_h : new_h);
        }
        if (config.h_max > 0.0 && std::abs(new_h) > config.h_max) {
            new_h = std::copysign(config.h_max, new_h);
        }

        Real eta = new_h / original_h;
        if (!std::isfinite(eta) || eta <= 0.0) {
            return false;
        }

        state.eta = eta;
        state.newh = (std::abs(eta - 1.0) > 1.0e-12) ? 1 : 0;
        h = original_h;
        state.h = original_h;
        state.hscal = original_h;
        state.jacobian_is_current = false;
        state.preconditioner_is_current = false;
        return true;
    }

    static Real weighted_rms_column(const State& state, int column) {
        Real sum = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real term = state.YH(static_cast<int>(i + 1), column) * state.ewt[i];
            sum += term * term;
        }
        return std::sqrt(sum / static_cast<Real>(N));
    }

    static void update_order_and_step(State& state,
                                      Real h_accepted,
                                      Real error_norm,
                                      const Config& config) {
        const Real ADDON = config.addon;
        const Real BIAS1 = config.bias1;
        const Real BIAS2 = config.bias2;
        const Real BIAS3 = config.bias3;
        const Real THRESH = config.order_change_threshold;

        const int max_order = std::min<int>(config.max_order, static_cast<int>(state.el.size()) - 1);
        const int max_l = max_order + 1;

        if (state.nqwait > 0) {
            state.nqwait = static_cast<short>(std::max<int>(state.nqwait - 1, 0));
        }

        if (state.l != max_l && state.nqwait == 1) {
            for (size_type i = 0; i < N; ++i) {
                state.YH(static_cast<int>(i + 1), max_l) = state.acor[i];
            }
            if (state.TQ(5) != 0.0) {
                state.conp = state.TQ(5);
            }
        }

        const Real h_current = h_accepted;
        const Real tq2 = std::max(state.TQ(2), math::UROUND);
        const Real dsm = error_norm / tq2;
        const Real flo = static_cast<Real>(state.l);

        if (state.nqwait > 0) {
            state.nqwait = static_cast<short>(std::max<int>(state.nqwait - 1, 0));
        }

        if (state.l != max_l && state.nqwait == 1) {
            for (size_type i = 0; i < N; ++i) {
                state.YH(static_cast<int>(i + 1), max_l) = state.acor[i];
            }
            state.conp = state.TQ(5);
        }

        if (state.etamax == 1.0) {
            if (state.nqwait < 2) {
                state.nqwait = 2;
            }
            state.newq = static_cast<short>(state.nq);
            state.newh = 0;
            state.eta = 1.0;
        } else {
            const Real dsm_clamped = std::max(dsm, static_cast<Real>(1.0e-12));
            Real eta_q = 1.0 / (std::pow(BIAS2 * dsm_clamped, 1.0 / flo) + ADDON);
            Real eta = eta_q;
            short proposed_order = static_cast<short>(state.nq);
            bool considered_change = false;
            Real eta_qm1 = 0.0;
            Real eta_qp1 = 0.0;

            if (state.nqwait == 0) {
                state.nqwait = 2;
                if (state.nq > 1) {
                    const Real ddn = weighted_rms_column(state, state.l) / std::max(state.TQ(1), math::UROUND);
                    const Real ddn_clamped = std::max(ddn, static_cast<Real>(1.0e-12));
                    eta_qm1 = 1.0 / (std::pow(BIAS1 * ddn_clamped, 1.0 / (flo - 1.0)) + ADDON);
                }
                if (state.l != max_l && std::abs(state.TAU(2)) > math::UROUND) {
                    const Real tau2 = std::max(std::abs(state.TAU(2)), math::UROUND);
                    const Real conp = (state.conp != 0.0) ? state.conp : state.TQ(5);
                    Real cnquot = 0.0;
                    if (conp != 0.0) {
                        cnquot = (state.TQ(5) / conp) *
                                 std::pow(state.h / tau2, static_cast<Real>(state.l));
                    }
                    std::array<Real, N> work{};
                    for (size_type i = 0; i < N; ++i) {
                        work[i] = state.acor[i];
                        if (cnquot != 0.0) {
                            work[i] -= cnquot * state.YH(static_cast<int>(i + 1), max_l);
                        }
                    }
                    const Real dup = weighted_rms_norm(work, state.ewt) / std::max(state.TQ(3), math::UROUND);
                    const Real dup_clamped = std::max(dup, static_cast<Real>(1.0e-12));
                    eta_qp1 = 1.0 / (std::pow(BIAS3 * dup_clamped, 1.0 / (flo + 1.0)) + ADDON);
                }

                if (eta_q < eta_qp1 && state.nq < max_order) {
                    if (eta_qp1 > eta_qm1 || state.nq == 1) {
                        eta = eta_qp1;
                        proposed_order = static_cast<short>(state.nq + 1);
                        considered_change = true;
                    }
                }
                if (!considered_change && eta_q < eta_qm1 && state.nq > 1) {
                    eta = eta_qm1;
                    proposed_order = static_cast<short>(state.nq - 1);
                    considered_change = true;
                }
            } else {
                eta = eta_q;
            }

            if (!std::isfinite(eta) || eta <= 0.0) {
                eta = 1.0;
                proposed_order = static_cast<short>(state.nq);
            }

            bool accept_change = (eta >= THRESH) && (state.etamax != 1.0);
            Real eta_limited = 1.0;
            if (accept_change) {
                eta_limited = std::min(eta, state.etamax);
                if (state.hmxi != 0.0 && std::abs(h_current) > 0.0) {
                    const Real max_growth = 1.0 / std::max(std::abs(h_current) * state.hmxi, math::UROUND);
                    eta_limited = std::min(eta_limited, max_growth);
                }
                if (config.h_max > 0.0 && std::abs(h_current) > 0.0) {
                    const Real max_growth = config.h_max / std::abs(h_current);
                    eta_limited = std::min(eta_limited, max_growth);
                }
                if (config.h_min > 0.0 && std::abs(h_current) > 0.0) {
                    const Real min_growth = config.h_min / std::abs(h_current);
                    eta_limited = std::max(eta_limited, min_growth);
                }
                if (!std::isfinite(eta_limited) || eta_limited <= 0.0) {
                    eta_limited = 1.0;
                    accept_change = false;
                    proposed_order = static_cast<short>(state.nq);
                }
            }

            if (!accept_change) {
                state.newq = static_cast<short>(state.nq);
                state.newh = 0;
                state.eta = 1.0;
            } else {
                if (proposed_order == static_cast<short>(state.nq + 1) && state.l != max_l) {
                    for (size_type i = 0; i < N; ++i) {
                        state.YH(static_cast<int>(i + 1), max_l) = state.acor[i];
                    }
                }
                state.newq = static_cast<short>(std::clamp<int>(proposed_order, 1, max_order));
                state.newh = (std::abs(eta_limited - 1.0) > 1.0e-12) ? 1 : 0;
                state.eta = eta_limited;
            }
        }

        state.h = h_current;
        state.hscal = h_current;
        state.etamax = (state.n_step <= 10) ? config.eta_max_early : config.eta_max_late;

        const Real inv_tq2 = 1.0 / std::max(state.TQ(2), math::UROUND);
        for (size_type i = 0; i < N; ++i) {
            state.acor[i] *= inv_tq2;
        }
    }

    static IntegratorResult step_once(State& state, Problem& problem, const Config& config) {
        if (state.h == 0.0) {
            return IntegratorResult::DT_UNDERFLOW;
        }

        const int max_consecutive_failures = config.max_consecutive_corrector_failures;
        const Real t_old = state.tn;
        std::array<Real, N> y_old{};
        for (size_type i = 0; i < N; ++i) {
            y_old[i] = state.YH(static_cast<int>(i + 1), 1);
        }

        bool prediction_active = false;
        auto retract_prediction = [&]() {
            if (prediction_active) {
                retract_nordsieck(state);
                prediction_active = false;
            }
        };

        while (true) {
            state.kflag = 0;
            if (state.newq != state.nq || state.newh != 0) {
                apply_pending_adjustments(state);
            }

            advance_nordsieck(state);
            prediction_active = true;

            const Real h_trial = state.h;
            const Real t_next = t_old + h_trial;

            set_coefficients(state);
            if (std::abs(state.EL(2)) > math::UROUND) {
                state.rl1 = 1.0 / state.EL(2);
            } else {
                state.rl1 = 1.0;
            }
            const Real prl1_safe = std::max(std::abs(state.prl1), math::UROUND);
            state.rc *= state.rl1 / prl1_safe;
            state.drc = state.rc - 1.0;
            state.prl1 = state.rl1;

            if (std::abs(state.drc) > config.rc_update_threshold) {
                state.ipup = 1;
            }
            if (config.max_steps_between_jacobian > 0 &&
                state.steps_since_jacobian >= config.max_steps_between_jacobian) {
                state.jacobian_is_current = false;
                state.ipup = 1;
            }
            if (config.enable_preconditioner &&
                config.max_steps_between_preconditioner > 0 &&
                state.steps_since_preconditioner >= config.max_steps_between_preconditioner) {
                state.ipup = 1;
            }

            NewtonResult newton = newton_solve(state, problem, h_trial, t_old, y_old, config);
            state.nni += newton.iterations;

            if (!newton.converged) {
                state.kflag -= 1;
                retract_prediction();
                state.netf += 1;
                state.ncf += 1;
                state.n_error_fail += 1;
                state.icf = 2;
                state.ipup = 1;
                state.jacobian_is_current = false;
                state.etamax = 1.0;

                if (state.ncf >= max_consecutive_failures) {
                    if (state.trace_enabled) trace_log(state, "FAIL_MAXCF");
                    retract_prediction();
                    return IntegratorResult::CORRECTOR_CONVERGENCE;
                }
                if (state.ncf >= 3 && state.nq > 1) {
                    state.newq = static_cast<short>(state.nq - 1);
                    state.nqwait = state.l;
                }

                Real eta = config.eta_cf;
                if (config.h_min > 0.0) {
                    eta = std::max(eta, config.h_min / std::max(std::abs(state.h), math::UROUND));
                }
                if (!adjust_step_down(state,
                                      state.h,
                                      std::max(newton.residual_norm, static_cast<Real>(1.0)),
                                      config,
                                      eta)) {
                    if (state.trace_enabled) trace_log(state, "FAIL_UNDER");
                    retract_prediction();
                    return IntegratorResult::DT_UNDERFLOW;
                }
                continue;
            }

            state.ncf = 0;
            state.acnrm_last = newton.correction_norm;

            const Real error_norm = std::max(newton.correction_norm, math::UROUND);
            if (error_norm > 1.0) {
                state.kflag -= 1;
                retract_prediction();
                state.netf += 1;
                state.n_error_fail += 1;
                state.icf = std::max<short>(state.icf, 1);
                state.ipup = 1;
                state.jacobian_is_current = false;
                state.etamax = 1.0;

                if (state.n_error_fail >= 3 && state.nq > 1) {
                    state.newq = static_cast<short>(state.nq - 1);
                    state.nqwait = state.l;
                }

                Real eta = config.eta_min;
                const Real flo = static_cast<Real>(state.l);
                const Real tq2 = std::max(state.TQ(2), math::UROUND);
                const Real dsm = error_norm / tq2;
                Real eta_q = 1.0 / (std::pow(config.bias2 * std::max(dsm, static_cast<Real>(1.0e-12)), 1.0 / flo) + config.addon);
                eta = std::max(eta, eta_q);
                if (config.h_min > 0.0) {
                    eta = std::max(eta, config.h_min / std::max(std::abs(state.h), math::UROUND));
                }
                if (state.kflag <= -2) {
                    eta = std::min(eta, config.eta_max_fail);
                }
                if (!adjust_step_down(state, state.h, error_norm, config, eta)) {
                    if (state.trace_enabled) trace_log(state, "FAIL_UNDER");
                    retract_prediction();
                    return IntegratorResult::DT_UNDERFLOW;
                }
                continue;
            }

            const Real abs_h = std::abs(state.h);
            if (config.h_min > 0.0 && abs_h <= config.hmin_relax * config.h_min) {
                if (state.trace_enabled) trace_log(state, "FAIL_HMIN");
                return IntegratorResult::DT_UNDERFLOW;
            }
            if (state.kflag <= config.kflag_error_threshold) {
                if (state.kflag <= config.kflag_failure_stop) {
                    if (state.trace_enabled) trace_log(state, "FAIL_KFLAG");
                    return IntegratorResult::DT_UNDERFLOW;
                }
                retract_prediction();
                state.etamax = 1.0;
                if (state.nq > 1) {
                    Real eta = config.eta_min;
                    if (config.h_min > 0.0 && abs_h > 0.0) {
                        eta = std::max(eta, config.h_min / abs_h);
                    }
                    state.newq = static_cast<short>(state.nq - 1);
                    state.nqwait = state.l;
                    state.eta = eta;
                    state.newh = 1;
                    prediction_active = false;
                    continue;
                } else {
                    Real eta = config.eta_min;
                    if (config.h_min > 0.0 && abs_h > 0.0) {
                        eta = std::max(eta, config.h_min / abs_h);
                    }
                    state.h *= eta;
                    state.hscal = state.h;
                    state.TAU(1) = state.h;
                    std::array<Real, N> y_current{};
                    for (size_type i = 0; i < N; ++i) {
                        y_current[i] = state.YH(static_cast<int>(i + 1), 1);
                    }
                    rhs(state.tn, y_current, state.savf);
                    state.n_rhs++;
                    for (size_type i = 0; i < N; ++i) {
                        state.YH(static_cast<int>(i + 1), 2) = state.h * state.savf[i];
                    }
                    state.nqwait = 10;
                    state.eta = 1.0;
                    state.newh = 0;
                    prediction_active = false;
                    continue;
                }
            }

            prediction_active = false;
            state.n_error_fail = 0;
            state.icf = 0;
            state.etamax = (state.n_step <= 10) ? config.eta_max_early : config.eta_max_late;
            state.tn = t_next;

            for (size_type i = 0; i < N; ++i) {
                state.YH(static_cast<int>(i + 1), 1) = state.y_temp[i];
                state.YH(static_cast<int>(i + 1), 2) = h_trial * state.savf[i];
            }
            for (int j = state.l; j >= 2; --j) {
                state.TAU(j) = state.TAU(j - 1);
            }
            state.TAU(1) = h_trial;

            state.h_used = h_trial;
            state.rc = 1.0;
            state.drc = 0.0;
            state.n_step++;
            state.steps_since_jacobian++;
            state.steps_since_preconditioner++;
            state.ipup = 0;

            update_order_and_step(state, h_trial, error_norm, config);

#ifdef INTEGRATORS_DVODPK_DEBUG
            if (state.n_step < 20) {
                std::cerr << "[DVODPK] step=" << state.n_step
                          << " t=" << state.tn
                          << " h=" << state.h
                          << " eta=" << state.eta
                          << " nq=" << state.nq
                          << " newq=" << state.newq
                          << " nqwait=" << state.nqwait
                          << " err_norm=" << error_norm
                          << std::endl;
            }
#endif
            if (state.trace_enabled) trace_log(state, "SUCCESS");
            return IntegratorResult::SUCCESS;
        }
    }

    static void setup_preconditioner(State& state, Problem& problem, Real gamma, const Config& config) {
        (void)problem;
        (void)gamma;
        if (!config.enable_preconditioner) {
            state.preconditioner_is_current = false;
            state.ipup = 0;
            return;
        }
        if constexpr (has_preconditioner_setup_v<Problem>) {
            Problem::preconditioner_setup(state.t, state.y, gamma, state.preconditioner);
            state.preconditioner_is_current = true;
            state.steps_since_preconditioner = 0;
            state.ipup = 0;
        } else {
            state.preconditioner_is_current = false;
            state.steps_since_preconditioner = 0;
            state.ipup = 0;
        }
    }

    static Real euclidean_norm(const std::array<Real, N>& vec) {
        Real sum = 0.0;
        for (size_type i = 0; i < N; ++i) {
            sum += vec[i] * vec[i];
        }
        return std::sqrt(sum);
    }

    static Real dot_product(const std::array<Real, N>& a, const std::array<Real, N>& b) {
        Real sum = 0.0;
        for (size_type i = 0; i < N; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static void apply_matrix(const State& state,
                             Real gamma,
                             const std::array<Real, N>& x,
                             std::array<Real, N>& out) {
        for (size_type i = 0; i < N; ++i) {
            Real jac_sum = 0.0;
            for (size_type j = 0; j < N; ++j) {
                jac_sum += state.jacobian[i][j] * x[j];
            }
            out[i] = x[i] - gamma * jac_sum;
        }
    }

    static void evaluate_jacobian(State& state,
                                  Problem& /*problem*/,
                                  Real t,
                                  const std::array<Real, N>& y,
                                  const Config& /*config*/) {
        if (state.jacobian_analytic && has_problem_jacobian) {
            jacobian(t, y, state.jacobian);
            state.n_jac++;
            state.jcur = 1;
            state.steps_since_jacobian = 0;
            return;
        }

        // Finite difference approximation.
        const Real squr = std::sqrt(std::max(math::UROUND, state.rtol));
        std::array<Real, N> y_pert{};
        for (size_type j = 0; j < N; ++j) {
            y_pert[j] = y[j];
        }

        for (size_type j = 0; j < N; ++j) {
            const Real yj = y[j];
            Real inc = squr * std::max(std::abs(yj), 1.0 / state.ewt[j]);
            if (inc == 0.0) {
                inc = squr;
            }
            y_pert[j] = yj + inc;
            rhs(t, y_pert, state.rhs_temp);
            state.n_rhs++;
            for (size_type i = 0; i < N; ++i) {
                state.jacobian[i][j] = (state.rhs_temp[i] - state.savf[i]) / inc;
            }
            y_pert[j] = yj;
        }
        state.n_jac++;
        state.jcur = 1;
        state.steps_since_jacobian = 0;
    }

    static void advance_nordsieck(State& state) {
        for (int k = state.nq; k >= 1; --k) {
            for (int j = k; j <= state.nq; ++j) {
                for (size_type i = 1; i <= N; ++i) {
                    state.YH(static_cast<int>(i), j) += state.YH(static_cast<int>(i), j + 1);
                }
            }
        }
    }

    static void retract_nordsieck(State& state) {
        for (int k = state.nq; k >= 1; --k) {
            for (int j = k; j <= state.nq; ++j) {
                for (size_type i = 1; i <= N; ++i) {
                    state.YH(static_cast<int>(i), j) -= state.YH(static_cast<int>(i), j + 1);
                }
            }
        }
    }

    static void apply_order_transform(State& state, int direction) {
        if (direction == 0) {
            return;
        }

        const int nq = state.nq;
        const int lmax = static_cast<int>(state.el.size());
        const Real hscale = (std::abs(state.hscal) > math::UROUND) ? state.hscal
                                                                   : ((std::abs(state.h) > math::UROUND) ? state.h : 1.0);

        if (direction < 0) {
            if (nq == 2) {
                return;
            }
            for (auto& e : state.el) {
                e = 0.0;
            }
            state.EL(3) = 1.0;
            Real hsum = 0.0;
            const int nqm2 = nq - 2;
            for (int j = 1; j <= nqm2; ++j) {
                hsum += state.TAU(j);
                const Real xi = hsum / hscale;
                const int jp1 = j + 1;
                for (int ib = 1; ib <= jp1; ++ib) {
                    const int idx = (j + 4) - ib;
                    if (idx < 1 || idx > lmax) continue;
                    const int prev = idx - 1;
                    const Real prev_val = (prev >= 1) ? state.EL(prev) : 0.0;
                    state.EL(idx) = state.EL(idx) * xi + prev_val;
                }
            }
            for (int j = 3; j <= nq; ++j) {
                const Real coeff = state.EL(j);
                if (coeff == 0.0) continue;
                for (size_type i = 1; i <= N; ++i) {
                    const Real correction = coeff * state.YH(static_cast<int>(i), state.l);
                    state.YH(static_cast<int>(i), j) -= correction;
                }
            }
            return;
        }

        // direction > 0 (order increase)
        for (auto& e : state.el) {
            e = 0.0;
        }
        state.EL(3) = 1.0;
        Real alph0 = -1.0;
        Real alph1 = 1.0;
        Real prod = 1.0;
        Real xiold = 1.0;
        Real hsum = hscale;

        if (nq != 1) {
            const int nqm1 = nq - 1;
            for (int j = 1; j <= nqm1; ++j) {
                const int jp1 = j + 1;
                if (jp1 + 1 <= static_cast<int>(state.tau.size())) {
                    hsum += state.TAU(jp1);
                }
                const Real xi = hsum / hscale;
                prod *= xi;
                alph0 -= 1.0 / static_cast<Real>(jp1);
                alph1 += 1.0 / xi;
                for (int ib = 1; ib <= jp1; ++ib) {
                    const int idx = (j + 4) - ib;
                    if (idx < 1 || idx > lmax) continue;
                    const int prev = idx - 1;
                    const Real prev_val = (prev >= 1) ? state.EL(prev) : 0.0;
                    state.EL(idx) = state.EL(idx) * xiold + prev_val;
                }
                xiold = xi;
            }
        }

        const Real t1 = (-alph0 - alph1) / prod;
        const int lp1 = state.l + 1;
        const int lmax_index = lmax;
        for (size_type i = 1; i <= N; ++i) {
            state.YH(static_cast<int>(i), lp1) = t1 * state.YH(static_cast<int>(i), lmax_index);
        }
        const int nqp1 = nq + 1;
        for (int j = 3; j <= nqp1; ++j) {
            const Real coeff = state.EL(j);
            if (coeff == 0.0) continue;
            for (size_type i = 1; i <= N; ++i) {
                state.YH(static_cast<int>(i), j) += coeff * state.YH(static_cast<int>(i), lp1);
            }
        }
    }

    static void apply_pending_adjustments(State& state) {
        if (state.newq != state.nq) {
            const int direction = (state.newq > state.nq) ? 1 : -1;
            apply_order_transform(state, direction);
            state.nq = state.newq;
            state.l = static_cast<short>(state.nq + 1);
            state.nqwait = state.l;
        }
        if (state.newh != 0) {
            Real scale = state.eta;
            if (scale != 1.0) {
                Real r = 1.0;
                for (int j = 2; j <= state.l; ++j) {
                    r *= scale;
                    for (size_type i = 1; i <= N; ++i) {
                        state.YH(static_cast<int>(i), j) *= r;
                    }
                }
                state.h *= scale;
                state.hscal = state.h;
                state.TAU(1) = state.h;
            }
            state.newh = 0;
        }
    }

    static KrylovResult solve_linear_system(State& state,
                                            std::array<Real, N>& rhs,
                                            Real gamma,
                                            const Config& config) {
        KrylovResult result{};
        const int maxl = std::max(1, std::min(config.max_krylov_iters, static_cast<int>(state.v.size()) - 1));
        const int ldhes = maxl + 1;

        auto& arnoldi = state.v;
        auto& solution_basis = state.z;
        auto& hes = state.hes;
        auto& gc = state.givens_cos;
        auto& gs = state.givens_sin;
        auto& g = state.residual;

        const bool use_preconditioner = config.enable_preconditioner && ProblemTraits<Problem>::has_custom_preconditioner;
        bool apply_right = use_preconditioner && (config.preconditioner_side == PreconditionerSide::RIGHT || config.preconditioner_side == PreconditionerSide::BOTH);
        bool apply_left = use_preconditioner && (config.preconditioner_side == PreconditionerSide::LEFT || config.preconditioner_side == PreconditionerSide::BOTH);
        if (apply_left && !apply_right) {
            // Left-only support is not implemented; fall back to right preconditioning.
            apply_right = true;
            apply_left = false;
        }

        const Real normb = euclidean_norm(rhs);
        if (normb == 0.0) {
            for (size_type i = 0; i < N; ++i) rhs[i] = 0.0;
            result.converged = true;
            result.iterations = 0;
            result.residual_norm = 0.0;
            return result;
        }

        for (int i = 0; i < (maxl + 1) * maxl; ++i) hes[static_cast<std::size_t>(i)] = 0.0;
        for (int i = 0; i <= maxl; ++i) g[static_cast<std::size_t>(i)] = 0.0;

        for (size_type i = 0; i < N; ++i) {
            arnoldi[0][i] = rhs[i] / normb;
            solution_basis[0][i] = arnoldi[0][i];
        }
        if (apply_right) {
            apply_preconditioner(state, solution_basis[0], PreconditionerSide::RIGHT);
        }
        g[0] = normb;

        Real residual_norm = normb;
        const Real tol = config.krylov_atol + config.krylov_rtol * normb;
        int converged_iteration = maxl - 1;

        std::array<Real, N> matvec{};

        for (int j = 0; j < maxl; ++j) {
            matvec = solution_basis[static_cast<std::size_t>(j)];
            apply_matrix(state, gamma, matvec, matvec);
            if (apply_left) {
                apply_preconditioner(state, matvec, PreconditionerSide::LEFT);
            }

            for (int i = 0; i <= j; ++i) {
                const Real hij = dot_product(matvec, arnoldi[static_cast<std::size_t>(i)]);
                hes[i + j * ldhes] = hij;
                for (size_type k = 0; k < N; ++k) {
                    matvec[k] -= hij * arnoldi[static_cast<std::size_t>(i)][k];
                }
            }

            const Real hnext = euclidean_norm(matvec);
            hes[(j + 1) + j * ldhes] = hnext;

            if (hnext != 0.0 && j + 1 < maxl) {
                for (size_type k = 0; k < N; ++k) {
                    arnoldi[static_cast<std::size_t>(j + 1)][k] = matvec[k] / hnext;
                    solution_basis[static_cast<std::size_t>(j + 1)][k] = arnoldi[static_cast<std::size_t>(j + 1)][k];
                }
                if (apply_right) {
                    apply_preconditioner(state, solution_basis[static_cast<std::size_t>(j + 1)], PreconditionerSide::RIGHT);
                }
            }

            for (int i = 0; i < j; ++i) {
                const Real temp = gc[static_cast<std::size_t>(i)] * hes[i + j * ldhes]
                                + gs[static_cast<std::size_t>(i)] * hes[(i + 1) + j * ldhes];
                hes[(i + 1) + j * ldhes] = -gs[static_cast<std::size_t>(i)] * hes[i + j * ldhes]
                                          + gc[static_cast<std::size_t>(i)] * hes[(i + 1) + j * ldhes];
                hes[i + j * ldhes] = temp;
            }

            const Real h_ii = hes[j + j * ldhes];
            const Real h_ip1 = hes[(j + 1) + j * ldhes];
            const Real denom = std::hypot(h_ii, h_ip1);
            Real c_new = 1.0;
            Real s_new = 0.0;
            if (denom != 0.0) {
                c_new = h_ii / denom;
                s_new = h_ip1 / denom;
            }
            gc[static_cast<std::size_t>(j)] = c_new;
            gs[static_cast<std::size_t>(j)] = s_new;
            hes[j + j * ldhes] = denom;
            if (j + 1 < ldhes) {
                hes[(j + 1) + j * ldhes] = 0.0;
            }

            const Real temp = c_new * g[static_cast<std::size_t>(j)];
            g[static_cast<std::size_t>(j + 1)] = -s_new * g[static_cast<std::size_t>(j)];
            g[static_cast<std::size_t>(j)] = temp;

            residual_norm = std::abs(g[static_cast<std::size_t>(j + 1)]);
            result.residual_norm = residual_norm;

            if (residual_norm <= tol || hnext == 0.0) {
                converged_iteration = j;
                result.converged = true;
                break;
            }
            converged_iteration = j;
        }

        const int krylov_dim = converged_iteration + 1;
        std::vector<Real> y(static_cast<std::size_t>(krylov_dim), 0.0);
        for (int i = krylov_dim - 1; i >= 0; --i) {
            Real sum = g[static_cast<std::size_t>(i)];
            for (int k = i + 1; k < krylov_dim; ++k) {
                sum -= hes[i + k * ldhes] * y[static_cast<std::size_t>(k)];
            }
            const Real diagonal = hes[i + i * ldhes];
            y[static_cast<std::size_t>(i)] = (diagonal != 0.0) ? (sum / diagonal) : 0.0;
        }

        for (size_type i = 0; i < N; ++i) rhs[i] = 0.0;
        for (int j = 0; j < krylov_dim; ++j) {
            for (size_type i = 0; i < N; ++i) {
                rhs[i] += y[static_cast<std::size_t>(j)] * solution_basis[static_cast<std::size_t>(j)][i];
            }
        }

        result.iterations = krylov_dim;

        if (!result.converged && result.residual_norm > tol) {
            std::array<std::array<Real, N>, N> fallback_matrix = state.jacobian;
            for (size_type i = 0; i < N; ++i) {
                for (size_type j = 0; j < N; ++j) {
                    fallback_matrix[i][j] = (i == j ? 1.0 : 0.0) - gamma * fallback_matrix[i][j];
                }
            }
            std::array<int, N> piv{};
            const int info = linalg::lu_decomposition<N>(fallback_matrix, piv);
            if (info == 0) {
                linalg::lu_solve<N>(fallback_matrix, piv, rhs);
                result.converged = true;
            }
        }

        return result;
    }

    static void trace_log(State& state, const char* tag) {
        if (!state.trace_enabled) return;
        if (!state.trace_stream.is_open()) return;
        auto old_flags = state.trace_stream.flags();
        auto old_prec = state.trace_stream.precision();
        state.trace_stream << tag
                           << " NST=" << state.n_step
                           << " NQ=" << state.nq
                           << " NEWQ=" << state.newq
                           << std::scientific << std::setprecision(6)
                           << " H=" << state.h
                           << " ETA=" << state.eta
                           << " ETAMAX=" << state.etamax
                           << std::defaultfloat
                           << " IPUP=" << state.ipup
                           << " JCUR=" << state.jcur
                           << " KFLAG=" << state.kflag
                           << " ICF=" << state.icf
                           << " NQWAIT=" << state.nqwait
                           << " STEPSJ=" << state.steps_since_jacobian
                           << " STEPSP=" << state.steps_since_preconditioner
                           << '\n';
        state.trace_stream.flags(old_flags);
        state.trace_stream.precision(old_prec);
        state.trace_stream.flush();
    }

    static void apply_preconditioner(const State& state, std::array<Real, N>& vec, PreconditionerSide side) {
        if constexpr (has_preconditioner_apply_v<Problem>) {
            Problem::apply_preconditioner(state.preconditioner, vec, side);
        } else {
            (void)state;
            (void)vec;
            (void)side;
        }
    }
};

} // namespace integrators

#endif // DVODPK_HPP
