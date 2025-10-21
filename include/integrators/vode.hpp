// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: VODE integrator (BDF up to order 5) ported from Microphysics
#ifndef VODE_HPP
#define VODE_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include "integrator_types.hpp"
#include "linear_algebra.hpp"
#include "steady_state_departure.hpp"

namespace integrators {

// Debug logging macro
#ifdef VODE_DEBUG
#define VODE_DBG(MSG) do { \
    std::cout.setf(std::ios::scientific, std::ios::floatfield); \
    std::cout.precision(17); \
    std::cout << "[VODE] " << MSG << std::endl; \
} while(0)
#else
#define VODE_DBG(MSG) do {} while(0)
#endif

// Constants mirroring Microphysics VODE
constexpr int VODE_MAXORD = 5;
constexpr int VODE_LMAX = VODE_MAXORD + 1; // L = NQ + 1
constexpr Real HMIN = 0.0; // no hard lower bound

template<size_type N>
struct VODEState : public IntegratorState<N> {
    // VODE control/state scalars
    Real CONP{0.0};
    Real CRATE{1.0};
    Real DRC{0.0};
    Real ETA{1.0};
    Real ETAMAX{1.0};
    Real H{0.0};
    Real HSCAL{0.0};
    Real PRL1{1.0};
    Real HMXI{0.0}; // inverse HMAX; 0 disables limit
    Real RC{0.0};
    Real RL1{1.0};
    Real tn{0.0};

    // Counters
    int n_rhs{0};
    int n_jac{0};
    int n_step{0};
    int err_fails{0}; // consecutive error test failures

    // Flags
    short ICF{0};
    short IPUP{1};
    short JCUR{0};
    short L{2};
    short NEWH{0};
    short NEWQ{1};
    short NQ{1};
    short NQWAIT{2};
    int NSLJ{0};
    int NSLP{0};
    
    // Integration limits
    int max_steps{1000};

    // Steady-state snap configuration
    bool steady_state_snap_enabled{true};
    Real steady_state_snap_balance_tolerance{1.0e-3};
    Real steady_state_snap_balance_floor{1.0e-30};
    Real steady_state_snap_timescale_safety{0.3};
    Real steady_state_snap_min_diagonal{1.0e-30};
    Real steady_state_snap_max_departure_tolerance{std::numeric_limits<Real>::infinity()};
    Real steady_state_snap_departure_floor{1.0e-30};
    Real steady_state_snap_symmetrization_floor{1.0e-30};

    // Coefficients and arrays (1-based in algorithm; we map i->i-1)
    std::array<Real, VODE_LMAX> el{};   // 1..L
    std::array<Real, VODE_LMAX> tau{};  // 1..L
    std::array<Real, 5> tq{};           // 1..5

    // Nordsieck history yh(:, j), j=1..Lmax
    std::array<std::array<Real, VODE_LMAX>, N> yh{};

    // Working vectors
    std::array<Real, N> ewt{};
    std::array<Real, N> savf{};
    std::array<Real, N> acor{};
    Real acnrm_last{0.0};

    // Linear algebra storage
    std::array<std::array<Real, N>, N> jacobian{}; // P matrix factored
    std::array<int, N> pivot{};

    // Helper accessors for 1-based arrays
    inline Real& EL(int i) { return el[static_cast<size_type>(i-1)]; }
    inline Real& TAU(int i) { return tau[static_cast<size_type>(i-1)]; }
    inline Real& TQ(int i) { return tq[static_cast<size_type>(i-1)]; }
    inline Real& YH(int i, int j) { return yh[static_cast<size_type>(i-1)][static_cast<size_type>(j-1)]; }
};

template<typename Problem>
class VODE {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = VODEState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;

private:
    // Evaluate RHS f(t, y) into out
    static inline void rhs(Real t, const std::array<Real, N>& y, std::array<Real, N>& out) {
        Problem::rhs(t, y, out);
    }

    // Evaluate analytic Jacobian if available
    static inline void jacobian(Real t, const std::array<Real, N>& y, std::array<std::array<Real, N>, N>& J) {
        Problem::jacobian(t, y, J);
    }

    static std::optional<SteadyStateSnapOutcome> try_steady_state_snap(
        State& s,
        [[maybe_unused]] bool allow_fallback,
        const typename ProblemTraits<Problem>::state_type& base_state,
        Real time_point,
        Real hydro_dt) {
        if constexpr (detail::has_steady_state_generator<Problem>::value) {
            if (!s.steady_state_snap_enabled) {
                return std::nullopt;
            }
            if (!(hydro_dt > Real{0})) {
                return std::nullopt;
            }

            using StateVec = typename ProblemTraits<Problem>::state_type;
            StateVec candidate = base_state;
            StateVec atol_vec{};
            StateVec rtol_vec{};
            for (size_type i = 0; i < N; ++i) {
                atol_vec[i] = s.atol;
                rtol_vec[i] = s.rtol;
            }

            SteadyStateSnapConfig cfg{};
            cfg.steady_state.atol = s.atol;
            cfg.steady_state.rtol = s.rtol;
            cfg.balance_tolerance = s.steady_state_snap_balance_tolerance;
            cfg.balance_floor = s.steady_state_snap_balance_floor;
            cfg.timescale_safety = s.steady_state_snap_timescale_safety;
            cfg.min_diagonal = s.steady_state_snap_min_diagonal;
            cfg.max_departure_tolerance = s.steady_state_snap_max_departure_tolerance;
            cfg.departure_floor = s.steady_state_snap_departure_floor;
            cfg.symmetrization_floor = s.steady_state_snap_symmetrization_floor;
            Real norm_target = std::accumulate(candidate.begin(), candidate.end(), Real{0});
            if (!(norm_target > Real{0})) {
                norm_target = Real{1};
            }
            cfg.steady_state.norm_target = norm_target;

            const auto outcome = attempt_steady_state_snap<Problem>(time_point,
                                                                    candidate,
                                                                    atol_vec,
                                                                    rtol_vec,
                                                                    hydro_dt,
                                                                    allow_fallback,
                                                                    cfg);
            if (outcome.result == SteadyStateSnapResult::Snapped) {
                for (size_type i = 0; i < N; ++i) {
                    s.y[i] = candidate[i];
                    s.YH(static_cast<int>(i+1), 1) = candidate[i];
                    for (int j = 2; j <= VODE_LMAX; ++j) {
                        s.YH(static_cast<int>(i+1), j) = 0.0;
                    }
                }
                s.t = s.tout;
                s.tn = s.tout;
                s.n_step = 0;
                s.err_fails = 0;
            }

            return outcome;
        } else {
            static_cast<void>(s);
            static_cast<void>(base_state);
            static_cast<void>(time_point);
            static_cast<void>(hydro_dt);
            return std::nullopt;
        }
    }

    // dvset: set integration coefficients
    static void dvset(State& s) {
        constexpr Real CORTES = 0.1;
        const Real FLOTL = static_cast<Real>(s.L);
        const int NQM1 = s.NQ - 1;
        const int NQM2 = s.NQ - 2;

        for (int i = 3; i <= s.L; ++i) s.EL(i) = 0.0;
        s.EL(1) = 1.0;
        s.EL(2) = 1.0;
        Real ALPH0 = -1.0;
        Real AHATN0 = -1.0;
        Real HSUM = s.H;
        Real RXI = 1.0;
        Real RXIS = 1.0;

        if (s.NQ != 1) {
            for (int j = 1; j <= NQM2; ++j) {
                HSUM += s.TAU(j);
                RXI = s.H / HSUM;
                ALPH0 -= 1.0 / static_cast<Real>(j+1);
                for (int iback = 1; iback <= j+1; ++iback) {
                    const int i = (j + 3) - iback;
                    s.EL(i) += s.EL(i-1) * RXI;
                }
            }
            ALPH0 -= 1.0 / static_cast<Real>(s.NQ);
            RXIS = -s.EL(2) - ALPH0;
            HSUM += s.TAU(NQM1);
            RXI = s.H / HSUM;
            AHATN0 = -s.EL(2) - RXI;
            for (int iback = 1; iback <= s.NQ; ++iback) {
                const int i = (s.NQ + 2) - iback;
                s.EL(i) += s.EL(i-1) * RXIS;
            }
        }

        const Real T1 = 1.0 - AHATN0 + ALPH0;
        const Real T2 = 1.0 + static_cast<Real>(s.NQ) * T1;
        s.TQ(2) = std::abs(ALPH0 * T2 / T1);
        s.TQ(5) = std::abs(T2 / (s.EL(s.L) * RXI / RXIS));

        if (s.NQWAIT == 1) {
            const Real CNQM1 = RXIS / s.EL(s.L);
            const Real T3 = ALPH0 + 1.0 / static_cast<Real>(s.NQ);
            const Real T4 = AHATN0 + RXI;
            Real ELP = T3 / (1.0 - T4 + T3);
            s.TQ(1) = std::abs(ELP / CNQM1);
            HSUM += s.TAU(s.NQ);
            RXI = s.H / HSUM;
            const Real T5 = ALPH0 - 1.0 / static_cast<Real>(s.NQ+1);
            const Real T6 = AHATN0 - RXI;
            ELP = T2 / (1.0 - T6 + T5);
            s.TQ(3) = std::abs(ELP * RXI * (FLOTL + 1.0) * T5);
        }

        s.TQ(4) = CORTES * s.TQ(2);
        VODE_DBG("dvset: NQ=" << s.NQ << " L=" << s.L << " el2=" << s.EL(2) << " tq2=" << s.TQ(2));
    }

    // Multiply yh by Pascal triangle matrix (advance prediction)
    static void advance_nordsieck(State& s) {
        for (int k = s.NQ; k >= 1; --k) {
            for (int j = k; j <= s.NQ; ++j) {
                for (size_type i = 1; i <= N; ++i) {
                    s.YH(static_cast<int>(i), j) += s.YH(static_cast<int>(i), j+1);
                }
            }
        }
    }

    // Undo Pascal multiplication (retract)
    static void retract_nordsieck(State& s) {
        for (int k = s.NQ; k >= 1; --k) {
            for (int j = k; j <= s.NQ; ++j) {
                for (size_type i = 1; i <= N; ++i) {
                    s.YH(static_cast<int>(i), j) -= s.YH(static_cast<int>(i), j+1);
                }
            }
        }
    }

    // dvjac: build and factor P = I - h*rl1*J
    static void dvjac(State& s) {
        // Build Jacobian J
        if (s.jacobian_analytic) {
            if (s.n_step == 1 && !s.debug_dump_done) {
                if constexpr (N >= 3) {
                    VODE_DBG("DUMP y for J (C++): " << s.y[0] << ", " << s.y[1] << ", " << s.y[2]);
                    VODE_DBG("DUMP YH(:,1) (C++): " << s.YH(1,1) << ", " << s.YH(2,1) << ", " << s.YH(3,1));
                }
            }
            jacobian(s.tn, s.y, s.jacobian);
        } else {
            // Numerical Jacobian using EWT-based step sizes
            Real fac = 0.0;
            for (size_type i = 0; i < N; ++i) fac += (s.savf[i] * s.ewt[i]) * (s.savf[i] * s.ewt[i]);
            fac = std::sqrt(fac / static_cast<Real>(N));
            Real R0 = 1000.0 * std::abs(s.H) * math::UROUND * static_cast<Real>(N) * fac;
            if (R0 == 0.0) R0 = 1.0;

            for (size_type j = 0; j < N; ++j) {
                const Real yj = s.y[j];
                const Real R = std::max(std::sqrt(math::UROUND) * std::abs(yj), R0 / s.ewt[j]);
                s.y[j] += R;
                const Real invR = 1.0 / R;
                std::array<Real, N> fpert{};
                rhs(s.tn, s.y, fpert);
                for (size_type i = 0; i < N; ++i) {
                    s.jacobian[i][j] = (fpert[i] - s.savf[i]) * invR;
                }
                s.y[j] = yj;
            }
            s.n_rhs += static_cast<int>(N);
        }

        // Form P = I - h*rl1*J and factor
        const Real hrl1 = s.H * s.RL1;
        const Real con = -hrl1;
        if (s.n_step == 1 && !s.debug_dump_done) {
            if constexpr (N >= 3) {
                VODE_DBG("DUMP J (C++):");
                for (size_type i = 0; i < N; ++i) {
                    VODE_DBG("J row " << i << ": " << s.jacobian[i][0] << ", " << s.jacobian[i][1] << ", " << s.jacobian[i][2]);
                }
            }
        }
        // Mirror DVODE arithmetic: scale entire matrix by con, then add identity
        for (size_type j = 0; j < N; ++j) {
            for (size_type i = 0; i < N; ++i) {
                s.jacobian[i][j] *= con;
            }
        }
        for (size_type i = 0; i < N; ++i) {
            s.jacobian[i][i] += 1.0;
        }
        if (s.n_step == 1 && !s.debug_dump_done) {
            if constexpr (N >= 3) {
                VODE_DBG("DUMP P (C++):");
                for (size_type i = 0; i < N; ++i) {
                    VODE_DBG("P row " << i << ": " << s.jacobian[i][0] << ", " << s.jacobian[i][1] << ", " << s.jacobian[i][2]);
                }
            }
        }
        int ier = linalg::lu_decomposition<N, true>(s.jacobian, s.pivot);
        s.JCUR = 1;
        if (s.n_step == 1 && !s.debug_dump_done) {
            if constexpr (N >= 3) {
                VODE_DBG("DUMP LU (C++), ipvt=" << s.pivot[0] << "," << s.pivot[1] << "," << s.pivot[2]);
                for (size_type i = 0; i < N; ++i) {
                    VODE_DBG("LU row " << i << ": " << s.jacobian[i][0] << ", " << s.jacobian[i][1] << ", " << s.jacobian[i][2]);
                }
            }
        }
        if (ier != 0) {
            // Mark as failure by setting ICF and leave factorization as-is
            s.ICF = 2;
        }
    }

    // dvnlsd: nonlinear solve for one step, returns ACNRM and sets NFLAG
    static Real dvnlsd(int& NFLAG, State& s) {
        constexpr Real CCMAX = 0.3;
        constexpr Real CRDOWN = 0.3;
        constexpr Real RDIV = 2.0;
        constexpr int MAXCOR = 3;
        constexpr int MSBP = 20;

        if (NFLAG == 0) s.ICF = 0;
        if (NFLAG == -2) s.IPUP = 1;

        // Check if we should update Jacobian
        s.DRC = std::abs(s.RC - 1.0);
        if (s.DRC > CCMAX || s.n_step >= s.NSLP + MSBP) s.IPUP = 1;

        bool converged = false;
        int M = 0;
        Real DELP = 0.0;

        // Initialize y from yh(:,1) and evaluate f
        for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), 1);
        rhs(s.tn, s.y, s.savf);
        s.n_rhs++;

        if (s.IPUP == 1) {
            dvjac(s);
            VODE_DBG("dvjac: updated P; JCUR=" << int(s.JCUR) << " ICF=" << int(s.ICF));
            s.IPUP = 0; s.RC = 1.0; s.DRC = 0.0; s.CRATE = 1.0; s.NSLP = s.n_step;
            // If factorization failed, force retry with smaller step
            if (s.ICF == 2) { NFLAG = -1; s.IPUP = 1; return 1e10; }
        }

        for (size_type i = 0; i < N; ++i) s.acor[i] = 0.0;

        Real ACNRM = 1e10;
        while (true) {
            // Build corrector RHS: (h*rl1)*f - (rl1*yh(:,2) + acor)
            std::array<Real, N> rhs_c{};
            for (size_type i = 0; i < N; ++i) rhs_c[i] = (s.RL1 * s.H) * s.savf[i] - (s.RL1 * s.YH(static_cast<int>(i+1), 2) + s.acor[i]);
            if (s.n_step == 1 && !s.debug_dump_done) {
                if constexpr (N >= 3) {
                    VODE_DBG("DUMP PRED YH2 (C++): " << s.YH(1,2) << ", " << s.YH(2,2) << ", " << s.YH(3,2));
                    VODE_DBG("DUMP SAVF (C++): " << s.savf[0] << ", " << s.savf[1] << ", " << s.savf[2]);
                }
            }
            // Instrument: norm of RHS before solve
            {
                Real DELrhs = 0.0;
                for (size_type i = 0; i < N; ++i) DELrhs += (rhs_c[i] * s.ewt[i]) * (rhs_c[i] * s.ewt[i]);
                DELrhs = std::sqrt(DELrhs / static_cast<Real>(N));
                VODE_DBG("dvnlsd: RHS_DEL=" << DELrhs);
            }
            // Solve P * delta = rhs_c
            auto delta = rhs_c;
            linalg::lu_solve<N, true>(s.jacobian, s.pivot, delta);
            if (s.RC != 1.0) {
                const Real CSCALE = 2.0 / (1.0 + s.RC);
                for (size_type i = 0; i < N; ++i) delta[i] *= CSCALE;
            }
            if (s.n_step == 1 && !s.debug_dump_done) {
                if constexpr (N >= 3) {
                    VODE_DBG("DUMP RHS (C++): " << rhs_c[0] << ", " << rhs_c[1] << ", " << rhs_c[2]);
                    VODE_DBG("DUMP SOL (C++): " << delta[0] << ", " << delta[1] << ", " << delta[2]);
                }
            }

            // Compute norm of correction
            Real DEL = 0.0;
            for (size_type i = 0; i < N; ++i) DEL += (delta[i] * s.ewt[i]) * (delta[i] * s.ewt[i]);
            DEL = std::sqrt(DEL / static_cast<Real>(N));
            VODE_DBG("dvnlsd: M=" << M << " DEL=" << DEL);

            for (size_type i = 0; i < N; ++i) s.acor[i] += delta[i];
            for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), 1) + s.acor[i];

            if (M != 0) s.CRATE = std::max(CRDOWN * s.CRATE, DEL / DELP);
            const Real DCON = DEL * std::min(1.0, s.CRATE) / s.TQ(4);
            VODE_DBG("dvnlsd: DCON=" << DCON << " CRATE=" << s.CRATE << " RL1=" << s.RL1 << " H=" << s.H);
            if (DCON <= 1.0) { converged = true; ACNRM = (M == 0 ? DEL : 0.0); break; }

            M += 1;
            if (M == MAXCOR) break;
            if (M >= 2 && DEL > RDIV * DELP) break;

            DELP = DEL;
            rhs(s.tn, s.y, s.savf);
            s.n_rhs++;
        }

        if (!converged) {
            VODE_DBG("dvnlsd: no convergence; JCUR=" << int(s.JCUR) << ", ICF set -> retry");
            if (s.JCUR == 1) { NFLAG = -1; s.ICF = 2; s.IPUP = 1; return 1e10; }
            s.ICF = 1; s.IPUP = 1; // retry with new Jacobian
            NFLAG = -2; // signal retry
            return 1e10;
        }

        // Success
        NFLAG = 0; s.JCUR = 0; s.ICF = 0;
        if (s.n_step == 1 && !s.debug_dump_done) { s.debug_dump_done = true; }
        if (M != 0) {
            ACNRM = 0.0;
            for (size_type i = 0; i < N; ++i) ACNRM += (s.acor[i] * s.ewt[i]) * (s.acor[i] * s.ewt[i]);
            ACNRM = std::sqrt(ACNRM / static_cast<Real>(N));
        }
        return ACNRM;
    }

    // dvjust: adjust YH on order change
    static void dvjust(int IORD, State& s) {
        if ((s.NQ == 2) && (IORD != 1)) return;
        const int NQM1 = s.NQ - 1; const int NQM2 = s.NQ - 2;
        if (IORD != 1) {
            for (int j = 1; j <= VODE_LMAX; ++j) s.EL(j) = 0.0;
            s.EL(3) = 1.0; Real HSUM = 0.0;
            for (int j = 1; j <= NQM2; ++j) {
                HSUM += s.TAU(j); Real XI = HSUM / s.HSCAL;
                for (int ib = 1; ib <= j+1; ++ib) { int i = (j+4)-ib; s.EL(i) = s.EL(i) * XI + s.EL(i-1); }
            }
            for (int j = 3; j <= s.NQ; ++j) {
                for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) -= s.YH(static_cast<int>(i), s.L) * s.EL(j);
            }
        } else {
            for (int j = 1; j <= VODE_LMAX; ++j) s.EL(j) = 0.0;
            s.EL(3) = 1.0; Real ALPH0 = -1.0, ALPH1 = 1.0, PROD = 1.0, XIOLD = 1.0; Real HSUM = s.HSCAL;
            if (s.NQ != 1) {
                for (int j = 1; j <= NQM1; ++j) {
                    HSUM += s.TAU(j+1); Real XI = HSUM / s.HSCAL; PROD *= XI; ALPH0 -= 1.0 / static_cast<Real>(j+1); ALPH1 += 1.0 / XI;
                    for (int ib = 1; ib <= j+1; ++ib) { int i = (j+4)-ib; s.EL(i) = s.EL(i) * XIOLD + s.EL(i-1); }
                    XIOLD = XI;
                }
            }
            Real T1 = (-ALPH0 - ALPH1) / PROD;
            for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), s.L+1) = T1 * s.YH(static_cast<int>(i), VODE_LMAX);
            for (int j = 3; j <= s.NQ + 1; ++j) {
                for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) += s.EL(j) * s.YH(static_cast<int>(i), s.L+1);
            }
        }
    }

    // dvhin: compute initial step size H0
    static void dvhin(State& s, Real& H0, int& NITER, int& IER) {
        constexpr Real PT1 = 0.1;
        NITER = 0; IER = 0; H0 = 0.0;
        const Real TDIST = std::abs(s.tout - s.t);
        const Real TROUND = math::UROUND * std::max(std::abs(s.t), std::abs(s.tout));
        if (TDIST < 2.0 * TROUND) { IER = -1; VODE_DBG("dvhin: TDIST too small"); return; }
        const Real HLB = 100.0 * TROUND;
        Real HUB = PT1 * TDIST;
        for (size_type i = 0; i < N; ++i) {
            const Real DELYI = PT1 * std::abs(s.YH(static_cast<int>(i+1),1)) + s.atol;
            const Real AFI = std::abs(s.YH(static_cast<int>(i+1),2));
            if (AFI * HUB > DELYI) HUB = DELYI / (AFI + 1e-300);
        }
        int iter = 0; Real HG = std::sqrt(HLB * HUB);
        if (HUB < HLB) { H0 = HG; NITER = iter; return; }
        Real hnew = HG;
        while (true) {
            const Real H = std::copysign(HG, s.tout - s.t);
            for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1) + H * s.YH(static_cast<int>(i+1),2);
            const Real t1 = s.t + H;
            std::array<Real,N> f{}; rhs(t1, s.y, f);
            for (size_type i = 0; i < N; ++i) s.acor[i] = (f[i] - s.YH(static_cast<int>(i+1),2)) / H;
            Real YDDNRM = 0.0; for (size_type i = 0; i < N; ++i) YDDNRM += (s.acor[i] * s.ewt[i]) * (s.acor[i] * s.ewt[i]);
            YDDNRM = std::sqrt(YDDNRM / static_cast<Real>(N));
            if (YDDNRM * HUB * HUB > 2.0) hnew = std::sqrt(2.0 / YDDNRM); else hnew = std::sqrt(HG * HUB);
            iter += 1;
            if (iter >= 4) break;
            const Real HRAT = hnew / HG; if ((HRAT > 0.5) && (HRAT < 2.0)) break;
            if ((iter >= 2) && (hnew > 2.0 * HG)) { hnew = HG; break; }
            HG = hnew;
        }
        H0 = std::copysign(std::clamp(hnew * 0.5, HLB, HUB), s.tout - s.t);
        VODE_DBG("dvhin: TDIST=" << TDIST << " HLB=" << HLB << " HUB=" << HUB << " H0=" << H0 << " iters=" << iter);
        NITER = iter; IER = 0;
    }

    // One dvstep; returns kflag (0 success, -1 dt underflow, -2 corrector failure)
    static int dvstep(State& s) {
        constexpr int MXNCF = 10;
        constexpr Real ADDON = 1.0e-6;
        constexpr Real BIAS1 = 6.0;
        constexpr Real BIAS2 = 6.0;
        constexpr Real BIAS3 = 10.0;
        constexpr Real ETACF = 0.25;
        constexpr Real ETAMIN = 0.1;
        [[maybe_unused]] constexpr Real ETAMXF = 0.2;
        constexpr Real ETAMX2 = 10.0;
        constexpr Real ETAMX3 = 10.0;
        constexpr Real ONEPSM = 1.00001;
        constexpr Real THRESH = 1.5;

        Real TOLD = s.tn;
        int NCF = 0;
        s.JCUR = 0; int NFLAG = 0;
        bool raised_this_step = false;

        // Save y before solve (not used for constraints here)
        while (true) {
            // Apply any pending order and/or step-size change before each attempt.
            // Order changes must apply even when ETA == 1 (no step-size change),
            // to match DVODE behavior. Step-size scaling applies only when NEWH != 0.
            const int prev_NQ = s.NQ;
            if (s.NEWH != 0 || s.NEWQ != s.NQ) {
                // Apply order change first if requested
                if (s.NEWQ < s.NQ) { dvjust(-1, s); s.NQ = s.NEWQ; s.L = static_cast<short>(s.NQ + 1); s.NQWAIT = s.L; }
                else if (s.NEWQ > s.NQ) { dvjust(1, s); s.NQ = s.NEWQ; s.L = static_cast<short>(s.NQ + 1); s.NQWAIT = s.L; }

                // Apply step-size change if scheduled
                if (s.NEWH != 0) {
                    // Rescale Nordsieck history by powers of ETA (Pascal transform)
                    Real Rpre = 1.0;
                    for (int j = 2; j <= s.L; ++j) {
                        Rpre *= s.ETA;
                        for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) *= Rpre;
                    }
                    // Apply the step-size change
                    s.H = s.H * s.ETA;
                    s.HSCAL = s.H;
                    s.RC *= s.ETA;
                    s.NEWH = 0; // consumed
                }
                raised_this_step = (s.NQ > prev_NQ);
            } else {
                raised_this_step = false;
            }

            s.tn += s.H;
            VODE_DBG("dvstep: predict t->" << s.tn << " H=" << s.H << " NQ=" << s.NQ);
            advance_nordsieck(s);
            dvset(s);
            s.RL1 = 1.0 / s.EL(2);
            s.RC *= (s.RL1 / s.PRL1);
            s.PRL1 = s.RL1;
            // No derivative refresh here; DVODE proceeds with predicted history

            VODE_DBG(
                "PRE tn=" << s.tn <<
                " H=" << s.H <<
                " NQ=" << int(s.NQ) <<
                " L=" << int(s.L) <<
                " RL1=" << s.RL1 <<
                " RC=" << s.RC <<
                " PRL1=" << s.PRL1 <<
                " NQWAIT=" << int(s.NQWAIT) <<
                " ETA=" << s.ETA <<
                " ETAMAX=" << s.ETAMAX <<
                " NEWH=" << int(s.NEWH) <<
                " NEWQ=" << int(s.NEWQ) <<
                " TQ2=" << s.TQ(2) <<
                " TQ3=" << s.TQ(3) <<
                " TQ4=" << s.TQ(4) <<
                " TQ5=" << s.TQ(5)
            );

            const Real ACNRM = dvnlsd(NFLAG, s);
            s.acnrm_last = ACNRM; // Save corrector norm for ORDER_DECIDE
            if (NFLAG != 0) {
                VODE_DBG("dvstep: corrector failed; ACNRM~" << ACNRM << " NCF=" << NCF);
                // Nonlinear solver failed; retract and cut H
                NCF += 1; s.ETAMAX = 1.0; s.tn = TOLD; retract_nordsieck(s);
                if (std::abs(s.H) <= HMIN * ONEPSM) return -2; // convergence failure
                if (NCF == MXNCF) return -2;
                s.ETA = ETACF; s.ETA = std::max(s.ETA, (HMIN > 0.0 ? HMIN / std::abs(s.H) : 0.0));
                Real R = 1.0; for (int j = 2; j <= s.L; ++j) { R *= s.ETA; for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) *= R; }
                s.H = s.HSCAL * s.ETA; s.HSCAL = s.H; s.RC *= s.ETA;
                // Refresh derivative history for new H to avoid large predictor defect
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), 1);
                rhs(s.tn, s.y, s.savf); s.n_rhs++;
                for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1), 2) = s.H * s.savf[i];
                VODE_DBG("dvstep: reduce H by ETA=" << s.ETA << " -> H=" << s.H);
                continue;
            }

            // Error test
            const Real DSM = ACNRM / s.TQ(2);
            if (s.n_step == 1) {
                if constexpr (N >= 3) {
                    VODE_DBG("ACCEPT_DEBUG ACOR=" << s.acor[0] << "," << s.acor[1] << "," << s.acor[2]
                             << " TQ2=" << s.TQ(2) << " DSM=" << DSM << " RC=" << s.RC << " CRATE=" << s.CRATE
                             << " NQWAIT=" << int(s.NQWAIT));
                }
            }
            VODE_DBG("POST ACNRM=" << ACNRM << " DSM=" << DSM << " tq2=" << s.TQ(2)
                << " JCUR=" << int(s.JCUR) << " ICF=" << int(s.ICF) << " CRATE=" << s.CRATE << " RC=" << s.RC);
            if (DSM <= 1.0) {
                VODE_DBG("dvstep: accept step; n_step=" << s.n_step+1);
                VODE_DBG("ACCEPT t=" << s.tn << " hu=" << s.H << " nq=" << int(s.NQ));
                // Successful step; update histories
                int L = s.L;
                s.n_step += 1; s.err_fails = 0;
                for (int iback = 1; iback <= s.NQ; ++iback) { int i = L - iback; s.TAU(i+1) = s.TAU(i); }
                s.TAU(1) = s.H;
                for (int j = 1; j <= s.L; ++j) {
                    for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) += s.EL(j) * s.acor[static_cast<size_type>(i-1)];
                }
                s.NQWAIT -= 1;
                VODE_DBG("ACCEPT_POST_PRE NQWAIT=" << int(s.NQWAIT) << " L=" << int(s.L)
                    << " TQ5=" << s.TQ(5) << " (pre-CONP update)");
                [[maybe_unused]] bool saved_lmax = false;
                if ((s.L != VODE_LMAX) && (s.NQWAIT == 1)) {
                    for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), VODE_LMAX) = s.acor[static_cast<size_type>(i-1)];
                    s.CONP = s.TQ(5);
                    saved_lmax = true;
                }
                VODE_DBG("ACCEPT_POST NQWAIT=" << int(s.NQWAIT) << " L=" << int(s.L)
                    << " saved_LMAX=" << (saved_lmax?1:0) << " CONP=" << s.CONP
                    << " TQ5=" << s.TQ(5));

                if (s.ETAMAX != 1.0) break;
                if (s.NQWAIT < 2) s.NQWAIT = 2;
                s.NEWQ = s.NQ; s.NEWH = 0; s.ETA = 1.0; s.ETAMAX = ETAMX3; if (s.n_step <= 10) s.ETAMAX = ETAMX2;
                const Real R = 1.0 / s.TQ(2); for (size_type i = 0; i < N; ++i) s.acor[i] *= R;
                return 0;
            }

            // Error test failed; retract and reduce H
            s.tn = TOLD; NFLAG = -2; retract_nordsieck(s);
            if (std::abs(s.H) <= HMIN * ONEPSM) return -1; // dt underflow
            s.ETAMAX = 1.0;
            s.err_fails += 1;
            Real ETANEW = 1.0 / (std::pow(BIAS2 * DSM, 1.0 / static_cast<Real>(s.L)) + ADDON);

            if (s.NQWAIT == 1) {
                s.NEWQ = s.NQ; s.NQWAIT = 2; Real ETAM = 1.0 / (std::pow(BIAS2 * DSM, 1.0 / static_cast<Real>(s.L)) + ADDON);
                if (ETAM < ETANEW) ETANEW = ETAM;
            }

            VODE_DBG("REJECT DSM=" << DSM << " ETANEW_init=" << ETANEW
                << " raised_this_step=" << (raised_this_step?1:0)
                << " NQWAIT=" << int(s.NQWAIT) << " NEWQ=" << int(s.NEWQ));

            // If we just raised order and failed, revert order by one on retry
            if (raised_this_step && s.NQ > 1) {
                VODE_DBG("dvstep: rejection after order increase; revert order to " << (s.NQ - 1));
                s.NEWQ = static_cast<short>(s.NQ - 1);
                s.NEWH = 1;
            }

            // DVODE: after 3 or more consecutive error test failures, drop order
            if (s.err_fails >= 3 && s.NQ > 1) {
                VODE_DBG("dvstep: repeated error test failures=" << s.err_fails << ", dropping order to " << (s.NQ - 1));
                s.NEWQ = static_cast<short>(s.NQ - 1);
                s.NEWH = 1;
                s.NQWAIT = s.L;
                // Force a modest cut to H
                s.ETA = std::max(ETAMIN, (HMIN > 0.0 ? HMIN / std::abs(s.H) : 0.0));
            }

            // DVODE: choose ETANEW subject to ETAMIN and HMIN
            Real eta_candidate = std::max(ETAMIN, ETANEW);
            // Honor HMIN
            eta_candidate = std::max((HMIN > 0.0 ? HMIN / std::abs(s.H) : 0.0), eta_candidate);
            s.ETA = eta_candidate;
            if (std::abs(s.H) * s.HMXI * s.ETA > 1.0) s.ETA = 1.0 / (std::abs(s.H) * s.HMXI);
            if (s.ETA == 1.0) s.ETA = ETAMIN;

            s.H = s.HSCAL * s.ETA; s.HSCAL = s.H; s.RC *= s.ETA;
            // Refresh derivative history for new H to avoid large predictor defect
            for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), 1);
            rhs(s.tn, s.y, s.savf); s.n_rhs++;
            for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1), 2) = s.H * s.savf[i];
            VODE_DBG("REJECT_APPLY ETA=" << s.ETA << " -> H=" << s.H << " RC=" << s.RC);
            continue;
        }

        // Consider order/timestep change (align with DVODE)
        bool already_set_eta = false;
        const Real FLOTL = static_cast<Real>(s.L);
        // Use the corrector's ACNRM from this step
        const Real DSM = s.acnrm_last / s.TQ(2);
        const Real ETAQ_eff = 1.0 / (std::pow(BIAS2 * DSM, 1.0 / FLOTL) + ADDON);

        if (s.NQWAIT != 0) {
            // Only allow same-order step change when NQWAIT != 0
            s.ETA = ETAQ_eff;
            s.NEWQ = s.NQ;
            already_set_eta = true;
        } else {
            // NQWAIT == 0: consider q-1 and q+1
            s.NQWAIT = 2;
            Real ETAQM1 = 0.0;
            if (s.NQ != 1) {
                Real DDN = 0.0; for (size_type i = 0; i < N; ++i) DDN += (s.YH(static_cast<int>(i+1), s.L) * s.ewt[i]) * (s.YH(static_cast<int>(i+1), s.L) * s.ewt[i]);
                DDN = std::sqrt(DDN / static_cast<Real>(N)) / s.TQ(1);
                ETAQM1 = 1.0 / (std::pow(BIAS1 * DDN, 1.0 / (FLOTL - 1.0)) + ADDON);
            }
            Real ETAQP1 = 0.0;
            if (s.L != VODE_LMAX) {
                Real CNQUOT = (s.TQ(5) / s.CONP) * std::pow(s.H / s.TAU(2), s.L);
                for (size_type i = 0; i < N; ++i) s.savf[i] = s.acor[i] - CNQUOT * s.YH(static_cast<int>(i+1), VODE_LMAX);
                Real DUP = 0.0; for (size_type i = 0; i < N; ++i) DUP += (s.savf[i] * s.ewt[i]) * (s.savf[i] * s.ewt[i]);
                DUP = std::sqrt(DUP / static_cast<Real>(N)) / s.TQ(3);
                ETAQP1 = 1.0 / (std::pow(BIAS3 * DUP, 1.0 / (FLOTL + 1.0)) + ADDON);
                VODE_DBG("ETAQP1 path: CONP=" << s.CONP << " TQ5=" << s.TQ(5)
                    << " H=" << s.H << " TAU2=" << s.TAU(2) << " L=" << int(s.L)
                    << " CNQUOT=" << CNQUOT << " DUP=" << DUP << " ETAQP1=" << ETAQP1);
            }
            VODE_DBG("ORDER_DECIDE DSM=" << DSM << " ETAQ_eff=" << ETAQ_eff
                << " ETAQM1=" << ETAQM1 << " ETAQP1=" << ETAQP1);
            if (ETAQ_eff < ETAQP1) {
                if (ETAQP1 > ETAQM1) { s.ETA = ETAQP1; s.NEWQ = static_cast<short>(s.NQ + 1); for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), VODE_LMAX) = s.acor[static_cast<size_type>(i-1)]; }
                else { s.ETA = ETAQM1; s.NEWQ = static_cast<short>(s.NQ - 1); }
                already_set_eta = true;
            }
            if (ETAQ_eff < ETAQM1 && !already_set_eta) { s.ETA = ETAQM1; s.NEWQ = static_cast<short>(s.NQ - 1); already_set_eta = true; }
        }
        if (!already_set_eta) { s.ETA = ETAQ_eff; s.NEWQ = s.NQ; }

        if (s.ETA >= THRESH && s.ETAMAX != 1.0) {
            s.ETA = std::min(s.ETA, s.ETAMAX);
            if (std::abs(s.H) * s.HMXI * s.ETA > 1.0) s.ETA = 1.0 / (std::abs(s.H) * s.HMXI * s.ETA);
            s.NEWH = 1; s.ETAMAX = ETAMX3; if (s.n_step <= 10) s.ETAMAX = ETAMX2;
            const Real R = 1.0 / s.TQ(2); for (size_type i = 0; i < N; ++i) s.acor[i] *= R; return 0;
        }
        // Keep selected NEWQ (if different from NQ) but do not change step size (ETA -> 1).
        VODE_DBG("ORDER_APPLY ETA=" << s.ETA << " NEWQ=" << int(s.NEWQ) << " THRESH=" << THRESH << " ETAMAX=" << s.ETAMAX);
        s.NEWH = 0; s.ETA = 1.0; s.ETAMAX = ETAMX3; if (s.n_step <= 10) s.ETAMAX = ETAMX2;
        { const Real R = 1.0 / s.TQ(2); for (size_type i = 0; i < N; ++i) s.acor[i] *= R; }
        return 0;
    }

public:
    IntegratorResult integrate(ProblemState& /*problem_state*/, State& s) {
        if (s.tout == s.t) return IntegratorResult::SUCCESS;

        using Vector = ProblemState;
        bool snap_attempted = false;
        auto attempt_snap_once = [&](bool allow_fallback,
                                     Vector base_state,
                                     Real time_point,
                                     Real hydro_dt) -> bool {
            if (snap_attempted) {
                return false;
            }
            if (!(hydro_dt > Real{0})) {
                return false;
            }
            auto outcome = try_steady_state_snap(s, allow_fallback, base_state, time_point, hydro_dt);
            if (!outcome.has_value()) {
                return false;
            }
            snap_attempted = true;
            return outcome->result == SteadyStateSnapResult::Snapped;
        };

        const Real initial_hydro_dt = std::abs(s.tout - s.t);
        if (attempt_snap_once(true, s.y, s.t, initial_hydro_dt)) {
            return IntegratorResult::SUCCESS;
        }

        // Initialize
        s.tn = s.t; s.n_step = 0; s.n_jac = 0; s.NSLJ = 0;

        // Initial RHS and load yh(:,2)
        rhs(s.t, s.y, s.savf); for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1),2) = s.savf[i]; s.n_rhs = 1;
        // Load initial values yh(:,1)
        for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1),1) = s.y[i];

        // Load and invert error weights; temporarily set H=1
        s.NQ = 1; s.H = 1.0; for (size_type i = 0; i < N; ++i) { s.ewt[i] = s.rtol * std::abs(s.YH(static_cast<int>(i+1),1)) + s.atol; s.ewt[i] = 1.0 / s.ewt[i]; }

        // Initial step size
        Real H0 = 0.0; int NITER = 0; int IER = 0; dvhin(s, H0, NITER, IER); s.n_rhs += NITER;
        if (IER != 0) {
            if (attempt_snap_once(false, s.y, s.t, std::abs(s.tout - s.t))) {
                return IntegratorResult::SUCCESS;
            }
            return IntegratorResult::DT_UNDERFLOW;
        }
        s.H = H0; for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1),2) *= s.H;

        // Initialize method/order and related vars (match DVODE semantics)
        s.NQ = 1; s.NEWQ = 1; s.L = 2; s.TAU(1) = s.H; s.PRL1 = 1.0; s.RC = 0.0;
        s.ETAMAX = 1.0e4; s.NQWAIT = 2; s.HSCAL = s.H; s.NEWH = 0; s.NSLP = 0; s.IPUP = 1;
        s.ETA = 0.0; // DVODE starts with ETA = 0 before the first step

        bool skip_loop_start = true;
        while (true) {
            if (!skip_loop_start) {
                if (s.n_step >= s.max_steps) {
                    Vector base{};
                    for (size_type i = 0; i < N; ++i) base[i] = s.YH(static_cast<int>(i+1),1);
                    if (attempt_snap_once(false, base, s.tn, std::abs(s.tout - s.tn))) {
                        return IntegratorResult::SUCCESS;
                    }
                    for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1);
                    s.t = s.tn; return IntegratorResult::TOO_MANY_STEPS;
                }
                for (size_type i = 0; i < N; ++i) { s.ewt[i] = s.rtol * std::abs(s.YH(static_cast<int>(i+1),1)) + s.atol; s.ewt[i] = 1.0 / s.ewt[i]; }
            } else { skip_loop_start = false; }

            // TOLSF: too much accuracy requested?
            Real TOLSF = 0.0; for (size_type i = 0; i < N; ++i) TOLSF += (s.YH(static_cast<int>(i+1),1)*s.ewt[i])*(s.YH(static_cast<int>(i+1),1)*s.ewt[i]);
            TOLSF = math::UROUND * std::sqrt(TOLSF / static_cast<Real>(N));
            if (TOLSF > 1.0) {
                if (s.n_step == 0) {
                    if (attempt_snap_once(false, s.y, s.tn, std::abs(s.tout - s.tn))) {
                        return IntegratorResult::SUCCESS;
                    }
                    return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
                }
                Vector base{};
                for (size_type i = 0; i < N; ++i) base[i] = s.YH(static_cast<int>(i+1),1);
                if (attempt_snap_once(false, base, s.tn, std::abs(s.tout - s.tn))) {
                    return IntegratorResult::SUCCESS;
                }
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1);
                s.t = s.tn; return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
            }

            int kflag = dvstep(s);
            if (kflag == -1) {
                Vector base{};
                for (size_type i = 0; i < N; ++i) base[i] = s.YH(static_cast<int>(i+1),1);
                if (attempt_snap_once(false, base, s.tn, std::abs(s.tout - s.tn))) {
                    return IntegratorResult::SUCCESS;
                }
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1);
                s.t = s.tn; return IntegratorResult::DT_UNDERFLOW;
            }
            if (kflag == -2) {
                Vector base{};
                for (size_type i = 0; i < N; ++i) base[i] = s.YH(static_cast<int>(i+1),1);
                if (attempt_snap_once(false, base, s.tn, std::abs(s.tout - s.tn))) {
                    return IntegratorResult::SUCCESS;
                }
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1);
                s.t = s.tn; return IntegratorResult::CORRECTOR_CONVERGENCE;
            }

            // stop criterion
            if ((s.tn - s.tout) * s.H < 0.0) continue;

            // Interpolate to tout
            for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), s.L);
            Real S = (s.tout - s.tn) / s.H;
            for (int jb = 1; jb <= s.NQ; ++jb) {
                const int j = s.NQ - jb;
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), j+1) + S * s.y[i];
            }
            s.t = s.tout; return IntegratorResult::SUCCESS;
        }
    }
};

} // namespace integrators

#endif // VODE_HPP
