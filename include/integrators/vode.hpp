// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: VODE integrator (BDF up to order 5) ported from Microphysics
#ifndef VODE_HPP
#define VODE_HPP

#include <array>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

// Debug logging macro
#if defined(VODE_DEBUG) && !defined(__CUDA_ARCH__)
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
    int n_decomp{0};
    int n_solve{0};
    int n_error_fails{0};
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

    // Optional narrow diagnostic trace for chemistry debugging.
    bool trace_deuterium_components{false};
    int trace_max_internal_steps{200000};

    // Helper accessors for 1-based arrays
    INTEGRATORS_HOST_DEVICE Real& EL(int i) { return el[static_cast<size_type>(i-1)]; }
    INTEGRATORS_HOST_DEVICE Real& TAU(int i) { return tau[static_cast<size_type>(i-1)]; }
    INTEGRATORS_HOST_DEVICE Real& TQ(int i) { return tq[static_cast<size_type>(i-1)]; }
    INTEGRATORS_HOST_DEVICE Real& YH(int i, int j) { return yh[static_cast<size_type>(i-1)][static_cast<size_type>(j-1)]; }
};

template<typename Problem>
class VODE {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = VODEState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;

private:
    // Evaluate RHS f(t, y) into out
    static INTEGRATORS_HOST_DEVICE void rhs(Real t, const std::array<Real, N>& y, std::array<Real, N>& out) {
        Problem::rhs(t, y, out);
    }

    static INTEGRATORS_HOST_DEVICE void trace_deuterium_state(const char* event,
                                                                     const State& s,
                                                                     const std::array<Real, N>& y,
                                                                     const std::array<Real, N>* f,
                                                                     Real aux) {
        if (!s.trace_deuterium_components || N <= 14 ||
            s.n_step > s.trace_max_internal_steps) {
            return;
        }
#if defined(__CUDA_ARCH__)
        printf("vode_trace,%s,%d,%.17e,%.17e,%d,%d,%d,%.17e,%.17e,%.17e,%.17e,%.17e",
               event, s.n_step, s.tn, s.H, static_cast<int>(s.NQ), s.n_rhs, s.n_jac,
               y[4], y[5], y[9], y[10], y[14]);
        if (f != nullptr) {
            printf(",%.17e,%.17e,%.17e,%.17e", (*f)[4], (*f)[5], (*f)[9], (*f)[10]);
        } else {
            printf(",nan,nan,nan,nan");
        }
        printf(",%.17e\n", aux);
#else
        std::cout << "vode_trace," << event
                  << "," << s.n_step
                  << "," << s.tn
                  << "," << s.H
                  << "," << static_cast<int>(s.NQ)
                  << "," << s.n_rhs
                  << "," << s.n_jac
                  << "," << y[4]
                  << "," << y[5]
                  << "," << y[9]
                  << "," << y[10]
                  << "," << y[14];
        if (f != nullptr) {
            std::cout << "," << (*f)[4]
                      << "," << (*f)[5]
                      << "," << (*f)[9]
                      << "," << (*f)[10];
        } else {
            std::cout << ",nan,nan,nan,nan";
        }
        std::cout << "," << aux << "\n";
#endif
    }

    static INTEGRATORS_HOST_DEVICE void clean_state_vector(State& s) {
        if (!s.clean_constrained_components) return;

        const int constrained_components = std::min<int>(s.constrained_components, static_cast<int>(N));
        for (int i = 0; i < constrained_components; ++i) {
            auto& value = s.y[static_cast<size_type>(i)];
            value = std::max(value, s.component_floor);
        }
    }

    static INTEGRATORS_HOST_DEVICE void rhs_state(Real t, State& s, std::array<Real, N>& out) {
        clean_state_vector(s);
        rhs(t, s.y, out);
    }

    // Evaluate analytic Jacobian if available
    static INTEGRATORS_HOST_DEVICE void jacobian(Real t, const std::array<Real, N>& y, std::array<std::array<Real, N>, N>& J) {
        Problem::jacobian(t, y, J);
    }

    static INTEGRATORS_HOST_DEVICE Real rtol_for(const State& s, size_type i) {
        return s.use_vector_tolerances ? s.rtol_vec[i] : s.rtol;
    }

    static INTEGRATORS_HOST_DEVICE Real atol_for(const State& s, size_type i) {
        return s.use_vector_tolerances ? s.atol_vec[i] : s.atol;
    }

    static INTEGRATORS_HOST_DEVICE void update_error_weights(State& s) {
        for (size_type i = 0; i < N; ++i) {
            s.ewt[i] = rtol_for(s, i) * std::abs(s.YH(static_cast<int>(i+1), 1)) + atol_for(s, i);
            s.ewt[i] = 1.0 / s.ewt[i];
        }
    }

    // dvset: set integration coefficients
    static INTEGRATORS_HOST_DEVICE void dvset(State& s) {
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
    static INTEGRATORS_HOST_DEVICE void advance_nordsieck(State& s) {
        for (int k = s.NQ; k >= 1; --k) {
            for (int j = k; j <= s.NQ; ++j) {
                for (size_type i = 1; i <= N; ++i) {
                    s.YH(static_cast<int>(i), j) += s.YH(static_cast<int>(i), j+1);
                }
            }
        }
    }

    // Undo Pascal multiplication (retract)
    static INTEGRATORS_HOST_DEVICE void retract_nordsieck(State& s) {
        for (int k = s.NQ; k >= 1; --k) {
            for (int j = k; j <= s.NQ; ++j) {
                for (size_type i = 1; i <= N; ++i) {
                    s.YH(static_cast<int>(i), j) -= s.YH(static_cast<int>(i), j+1);
                }
            }
        }
    }

    // dvjac: build and factor P = I - h*rl1*J
    static INTEGRATORS_HOST_DEVICE int dvjac(State& s) {
        // Build Jacobian J
        if (s.jacobian_analytic && ProblemTraits<Problem>::has_analytic_jacobian) {
            if constexpr (ProblemTraits<Problem>::has_analytic_jacobian) {
                jacobian(s.tn, s.y, s.jacobian);
            }
        } else {
            if (s.jacobian_analytic && !ProblemTraits<Problem>::has_analytic_jacobian) {
                s.jacobian_analytic = false;
            }
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
                rhs_state(s.tn, s, fpert);
                for (size_type i = 0; i < N; ++i) {
                    s.jacobian[i][j] = (fpert[i] - s.savf[i]) * invR;
                }
                s.y[j] = yj;
            }
            s.n_rhs += static_cast<int>(N);
        }
        s.n_jac += 1;
        s.NSLJ = s.n_step;

        // Form P = I - h*rl1*J and factor
        const Real hrl1 = s.H * s.RL1;
        const Real con = -hrl1;

        // Mirror DVODE arithmetic: scale entire matrix by con, then add identity
        for (size_type j = 0; j < N; ++j) {
            for (size_type i = 0; i < N; ++i) {
                s.jacobian[i][j] *= con;
            }
        }
        for (size_type i = 0; i < N; ++i) {
            s.jacobian[i][i] += 1.0;
        }
        int ier = linalg::lu_decomposition<N, true>(s.jacobian, s.pivot);
        if (ier == 0) {
            s.n_decomp += 1;
        }
        s.JCUR = 1;
        return ier;
    }

    // dvnlsd: nonlinear solve for one step, returns ACNRM and sets NFLAG
    static INTEGRATORS_HOST_DEVICE Real dvnlsd(int& NFLAG, State& s) {
        constexpr Real CCMAX = 0.3;
        constexpr Real CRDOWN = 0.3;
        constexpr Real RDIV = 2.0;
        constexpr int MAXCOR = 3;
        constexpr int MSBP = 20;

        Real ACNRM = 1.e10;
        bool converged = false;
        int M = 0;
        Real DEL = 0.0;

        while (true) {
            if (NFLAG == 0) s.ICF = 0;
            if (NFLAG == -2) s.IPUP = 1;

            s.DRC = std::abs(s.RC - 1.0);
            if (s.DRC > CCMAX || s.n_step >= s.NSLP + MSBP) s.IPUP = 1;

            M = 0;
            Real DELP = 0.0;

            for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), 1);
            rhs_state(s.tn, s, s.savf);
            s.n_rhs++;
            trace_deuterium_state("corrector_rhs", s, s.y, &s.savf, static_cast<Real>(NFLAG));

            if (s.IPUP == 1) {
                const int IERPJ = dvjac(s);
                VODE_DBG("dvjac: updated P; JCUR=" << int(s.JCUR) << " ICF=" << int(s.ICF));
                s.IPUP = 0;
                s.RC = 1.0;
                s.DRC = 0.0;
                s.CRATE = 1.0;
                s.NSLP = s.n_step;

                if (IERPJ != 0) {
                    NFLAG = -1;
                    s.ICF = 2;
                    s.IPUP = 1;
                    return ACNRM;
                }
            }

            for (size_type i = 0; i < N; ++i) s.acor[i] = 0.0;

            while (true) {
                std::array<Real, N> delta{};
                for (size_type i = 0; i < N; ++i) {
                    delta[i] = (s.RL1 * s.H) * s.savf[i] -
                               (s.RL1 * s.YH(static_cast<int>(i+1), 2) + s.acor[i]);
                }

                linalg::lu_solve<N, true>(s.jacobian, s.pivot, delta);
                s.n_solve += 1;

                if (s.RC != 1.0) {
                    const Real CSCALE = 2.0 / (1.0 + s.RC);
                    for (size_type i = 0; i < N; ++i) delta[i] *= CSCALE;
                }

                DEL = 0.0;
                for (size_type i = 0; i < N; ++i) {
                    DEL += (delta[i] * s.ewt[i]) * (delta[i] * s.ewt[i]);
                }
                DEL = std::sqrt(DEL / static_cast<Real>(N));
                VODE_DBG("dvnlsd: M=" << M << " DEL=" << DEL);

                for (size_type i = 0; i < N; ++i) s.acor[i] += delta[i];
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), 1) + s.acor[i];
                trace_deuterium_state("corrector_delta", s, s.y, &delta, DEL);

                if (M != 0) s.CRATE = std::max(CRDOWN * s.CRATE, DEL / DELP);

                const Real DCON = DEL * std::min(1.0, s.CRATE) / s.TQ(4);
                VODE_DBG("dvnlsd: DCON=" << DCON << " CRATE=" << s.CRATE << " RL1=" << s.RL1 << " H=" << s.H);
                if (DCON <= 1.0) {
                    converged = true;
                    break;
                }

                M += 1;
                if (M == MAXCOR) break;
                if (M >= 2 && DEL > RDIV * DELP) break;

                DELP = DEL;
                rhs_state(s.tn, s, s.savf);
                s.n_rhs++;
            }

            if (converged) break;

            VODE_DBG("dvnlsd: no convergence; JCUR=" << int(s.JCUR));
            if (s.JCUR == 1) {
                NFLAG = -1;
                s.ICF = 2;
                s.IPUP = 1;
                return ACNRM;
            }

            s.ICF = 1;
            s.IPUP = 1;
        }

        NFLAG = 0;
        s.JCUR = 0;
        s.ICF = 0;

        if (M == 0) {
            ACNRM = DEL;
        } else {
            ACNRM = 0.0;
            for (size_type i = 0; i < N; ++i) {
                ACNRM += (s.acor[i] * s.ewt[i]) * (s.acor[i] * s.ewt[i]);
            }
            ACNRM = std::sqrt(ACNRM / static_cast<Real>(N));
        }

        return ACNRM;
    }

    // dvjust: adjust YH on order change
    static INTEGRATORS_HOST_DEVICE void dvjust(int IORD, State& s) {
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
    static INTEGRATORS_HOST_DEVICE void dvhin(State& s, Real& H0, int& NITER, int& IER) {
        constexpr Real PT1 = 0.1;
        NITER = 0; IER = 0; H0 = 0.0;
        const Real TDIST = std::abs(s.tout - s.t);
        const Real TROUND = math::UROUND * std::max(std::abs(s.t), std::abs(s.tout));
        if (TDIST < 2.0 * TROUND) { IER = -1; VODE_DBG("dvhin: TDIST too small"); return; }
        const Real HLB = 100.0 * TROUND;
        Real HUB = PT1 * TDIST;
        for (size_type i = 0; i < N; ++i) {
            const Real DELYI = PT1 * std::abs(s.YH(static_cast<int>(i+1),1)) + atol_for(s, i);
            const Real AFI = std::abs(s.YH(static_cast<int>(i+1),2));
            if (AFI * HUB > DELYI) {
                HUB = DELYI / (AFI + 1e-300);
            }
        }
        int iter = 0; Real HG = std::sqrt(HLB * HUB);
        if (HUB < HLB) { H0 = HG; NITER = iter; return; }
        Real hnew = HG;
        while (true) {
            const Real H = std::copysign(HG, s.tout - s.t);
            for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1) + H * s.YH(static_cast<int>(i+1),2);
            const Real t1 = s.t + H;
            std::array<Real,N> f{}; rhs_state(t1, s, f);
            for (size_type i = 0; i < N; ++i) s.acor[i] = (f[i] - s.YH(static_cast<int>(i+1),2)) / H;
            Real YDDNRM = 0.0;
            for (size_type i = 0; i < N; ++i) {
                const Real weighted = s.acor[i] * s.ewt[i];
                YDDNRM += weighted * weighted;
            }
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
    static INTEGRATORS_HOST_DEVICE int dvstep(State& s) {
        constexpr int KFC = -3;
        constexpr int KFH = -7;
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

        Real DSM = 0.0;
        int kflag = 0;
        const Real TOLD = s.tn;
        int NCF = 0;
        s.JCUR = 0;
        int NFLAG = 0;

        if (s.NEWH != 0) {
            if (s.NEWQ < s.NQ) {
                dvjust(-1, s);
                s.NQ = s.NEWQ;
                s.L = static_cast<short>(s.NQ + 1);
                s.NQWAIT = s.L;
            } else if (s.NEWQ > s.NQ) {
                dvjust(1, s);
                s.NQ = s.NEWQ;
                s.L = static_cast<short>(s.NQ + 1);
                s.NQWAIT = s.L;
            }

            Real R = 1.0;
            for (int j = 2; j <= s.L; ++j) {
                R *= s.ETA;
                for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) *= R;
            }

            s.H = s.HSCAL * s.ETA;
            s.HSCAL = s.H;
            s.RC *= s.ETA;
        }

        std::array<Real, N> y_save{};
        for (size_type i = 0; i < N; ++i) y_save[i] = s.y[i];

        while (true) {
            s.tn += s.H;
            VODE_DBG("dvstep: predict t->" << s.tn << " H=" << s.H << " NQ=" << s.NQ);
            advance_nordsieck(s);
            dvset(s);
            s.RL1 = 1.0 / s.EL(2);
            s.RC *= (s.RL1 / s.PRL1);
            s.PRL1 = s.RL1;

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
                NCF += 1;
                s.ETAMAX = 1.0;
                s.tn = TOLD;
                retract_nordsieck(s);
                if (std::abs(s.H) <= HMIN * ONEPSM) return -2;
                if (NCF == MXNCF) return -2;

                s.ETA = ETACF;
                s.ETA = std::max(s.ETA, (HMIN > 0.0 ? HMIN / std::abs(s.H) : 0.0));
                Real R = 1.0;
                for (int j = 2; j <= s.L; ++j) {
                    R *= s.ETA;
                    for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) *= R;
                }
                s.H = s.HSCAL * s.ETA;
                s.HSCAL = s.H;
                s.RC *= s.ETA;
                VODE_DBG("dvstep: reduce H by ETA=" << s.ETA << " -> H=" << s.H);
                continue;
            }

            bool valid_update = true;
            const int constrained_components = std::min<int>(s.constrained_components, static_cast<int>(N));
            for (int i = 0; i < constrained_components; ++i) {
                const auto idx = static_cast<size_type>(i);
                const Real reject_threshold = s.reject_change_buffer * atol_for(s, idx);
                if (std::abs(y_save[idx]) > reject_threshold &&
                    std::abs(s.y[idx]) > reject_threshold &&
                    (std::abs(s.y[idx]) > s.increase_change_factor * std::abs(y_save[idx]) ||
                     std::abs(s.y[idx]) < s.decrease_change_factor * std::abs(y_save[idx]))) {
                    valid_update = false;
                    break;
                }

                if (s.y[idx] < -s.species_failure_tolerance) {
                    valid_update = false;
                    break;
                }

                if (s.enforce_component_ceiling &&
                    s.y[idx] > s.component_ceiling + s.species_failure_tolerance) {
                    valid_update = false;
                    break;
                }
            }

            DSM = ACNRM / s.TQ(2);
            VODE_DBG("POST ACNRM=" << ACNRM << " DSM=" << DSM << " tq2=" << s.TQ(2)
                << " JCUR=" << int(s.JCUR) << " ICF=" << int(s.ICF) << " CRATE=" << s.CRATE << " RC=" << s.RC);
            if (DSM <= 1.0 && valid_update) {
                VODE_DBG("dvstep: accept step; n_step=" << s.n_step+1);
                VODE_DBG("ACCEPT t=" << s.tn << " hu=" << s.H << " nq=" << int(s.NQ));
                kflag = 0;
                s.n_step += 1; s.err_fails = 0;
                for (int iback = 1; iback <= s.NQ; ++iback) {
                    const int i = s.L - iback;
                    s.TAU(i+1) = s.TAU(i);
                }
                s.TAU(1) = s.H;
                for (int j = 1; j <= s.L; ++j) {
                    for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) += s.EL(j) * s.acor[static_cast<size_type>(i-1)];
                }
                if (s.trace_deuterium_components) {
                    std::array<Real, N> accepted_y{};
                    for (size_type i = 0; i < N; ++i) {
                        accepted_y[i] = s.YH(static_cast<int>(i + 1), 1);
                    }
                    trace_deuterium_state("accept", s, accepted_y, &s.savf, ACNRM);
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
                s.NEWQ = s.NQ;
                s.NEWH = 0;
                s.ETA = 1.0;
                s.ETAMAX = ETAMX3;
                if (s.n_step <= 10) s.ETAMAX = ETAMX2;
                const Real R = 1.0 / s.TQ(2); for (size_type i = 0; i < N; ++i) s.acor[i] *= R;
                return kflag;
            }

            kflag -= 1;
            s.n_error_fails += 1;
            NFLAG = -2;
            s.tn = TOLD;
            retract_nordsieck(s);
            if (std::abs(s.H) <= HMIN * ONEPSM) return -1;
            s.ETAMAX = 1.0;

            if (kflag > KFC) {
                const Real FLOTL = static_cast<Real>(s.L);
                s.ETA = 1.0 / (std::pow(BIAS2 * DSM, 1.0 / FLOTL) + ADDON);
                s.ETA = std::max({s.ETA, (HMIN > 0.0 ? HMIN / std::abs(s.H) : 0.0), ETAMIN});
                if ((kflag <= -2) && (s.ETA > ETAMXF)) s.ETA = ETAMXF;

                Real R = 1.0;
                for (int j = 2; j <= s.L; ++j) {
                    R *= s.ETA;
                    for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) *= R;
                }

                s.H = s.HSCAL * s.ETA;
                s.HSCAL = s.H;
                s.RC *= s.ETA;
                VODE_DBG("REJECT_APPLY ETA=" << s.ETA << " -> H=" << s.H << " RC=" << s.RC);
                continue;
            }

            if (kflag == KFH) return -1;

            if (s.NQ != 1) {
                s.ETA = std::max(ETAMIN, (HMIN > 0.0 ? HMIN / std::abs(s.H) : 0.0));
                dvjust(-1, s);
                s.L = s.NQ;
                s.NQ -= 1;
                s.NQWAIT = s.L;

                Real R = 1.0;
                for (int j = 2; j <= s.L; ++j) {
                    R *= s.ETA;
                    for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), j) *= R;
                }

                s.H = s.HSCAL * s.ETA;
                s.HSCAL = s.H;
                s.RC *= s.ETA;
                continue;
            }

            s.ETA = std::max(ETAMIN, (HMIN > 0.0 ? HMIN / std::abs(s.H) : 0.0));
            s.H *= s.ETA;
            s.HSCAL = s.H;
            s.TAU(1) = s.H;
            rhs_state(s.tn, s, s.savf);
            s.n_rhs++;
            for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1), 2) = s.H * s.savf[i];
            s.NQWAIT = 10;
        }

        // Consider order/timestep change (align with DVODE)
        bool already_set_eta = false;
        const Real FLOTL = static_cast<Real>(s.L);
        const Real ETAQ = 1.0 / (std::pow(BIAS2 * DSM, 1.0 / FLOTL) + ADDON);

        if (s.NQWAIT == 0) {
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
            VODE_DBG("ORDER_DECIDE DSM=" << DSM << " ETAQ=" << ETAQ
                << " ETAQM1=" << ETAQM1 << " ETAQP1=" << ETAQP1);
            if (ETAQ < ETAQP1) {
                if (ETAQP1 > ETAQM1) { s.ETA = ETAQP1; s.NEWQ = static_cast<short>(s.NQ + 1); for (size_type i = 1; i <= N; ++i) s.YH(static_cast<int>(i), VODE_LMAX) = s.acor[static_cast<size_type>(i-1)]; }
                else { s.ETA = ETAQM1; s.NEWQ = static_cast<short>(s.NQ - 1); }
                already_set_eta = true;
            }
            if (ETAQ < ETAQM1 && !already_set_eta) { s.ETA = ETAQM1; s.NEWQ = static_cast<short>(s.NQ - 1); already_set_eta = true; }
        }
        if (!already_set_eta) { s.ETA = ETAQ; s.NEWQ = s.NQ; }

        if (s.ETA >= THRESH && s.ETAMAX != 1.0) {
            s.ETA = std::min(s.ETA, s.ETAMAX);
            s.ETA = s.ETA / std::max(1.0, std::abs(s.H) * s.HMXI * s.ETA);
            s.NEWH = 1; s.ETAMAX = ETAMX3; if (s.n_step <= 10) s.ETAMAX = ETAMX2;
            const Real R = 1.0 / s.TQ(2); for (size_type i = 0; i < N; ++i) s.acor[i] *= R; return kflag;
        }
        VODE_DBG("ORDER_APPLY ETA=" << s.ETA << " NEWQ=" << int(s.NEWQ) << " THRESH=" << THRESH << " ETAMAX=" << s.ETAMAX);
        s.NEWQ = s.NQ;
        s.NEWH = 0; s.ETA = 1.0; s.ETAMAX = ETAMX3; if (s.n_step <= 10) s.ETAMAX = ETAMX2;
        { const Real R = 1.0 / s.TQ(2); for (size_type i = 0; i < N; ++i) s.acor[i] *= R; }
        return kflag;
    }

public:
    INTEGRATORS_HOST_DEVICE IntegratorResult integrate(ProblemState& /*problem_state*/, State& s) {
        if (s.tout == s.t) return IntegratorResult::SUCCESS;

        // Initialize
        s.tn = s.t; s.n_step = 0; s.n_jac = 0; s.n_decomp = 0; s.n_solve = 0;
        s.n_error_fails = 0; s.NSLJ = 0;

        // Initial RHS and load yh(:,2)
        rhs_state(s.t, s, s.savf);
        for (size_type i = 0; i < N; ++i) {
            s.YH(static_cast<int>(i+1),2) = s.savf[i];
        }
        s.n_rhs = 1;
        // Load initial values yh(:,1)
        for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1),1) = s.y[i];
        trace_deuterium_state("init_rhs", s, s.y, &s.savf, 0.0);

        // Load and invert error weights; temporarily set H=1
        s.NQ = 1; s.H = 1.0; update_error_weights(s);

        // Initial step size
        Real H0 = 0.0; int NITER = 0; int IER = 0; dvhin(s, H0, NITER, IER); s.n_rhs += NITER;
        if (IER != 0) return IntegratorResult::DT_UNDERFLOW;
        s.H = H0; for (size_type i = 0; i < N; ++i) s.YH(static_cast<int>(i+1),2) *= s.H;
        trace_deuterium_state("initial_h", s, s.y, nullptr, H0);

        // Initialize method/order and related vars (match DVODE semantics)
        s.NQ = 1; s.NEWQ = 1; s.L = 2; s.TAU(1) = s.H; s.PRL1 = 1.0; s.RC = 0.0;
        s.ETAMAX = 1.0e4; s.NQWAIT = 2; s.HSCAL = s.H; s.NEWH = 0; s.NSLP = 0; s.IPUP = 1;
        s.ETA = 0.0; // DVODE starts with ETA = 0 before the first step

        bool skip_loop_start = true;
        while (true) {
            if (!skip_loop_start) {
                if (s.n_step >= s.max_steps) {
                    // too many steps
                    for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1);
                    s.t = s.tn; return IntegratorResult::TOO_MANY_STEPS;
                }
                update_error_weights(s);
            } else { skip_loop_start = false; }

            // TOLSF: too much accuracy requested?
            Real TOLSF = 0.0; for (size_type i = 0; i < N; ++i) TOLSF += (s.YH(static_cast<int>(i+1),1)*s.ewt[i])*(s.YH(static_cast<int>(i+1),1)*s.ewt[i]);
            TOLSF = math::UROUND * std::sqrt(TOLSF / static_cast<Real>(N));
            if (TOLSF > 1.0) {
                if (s.n_step == 0) return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1);
                s.t = s.tn; return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
            }

            int kflag = dvstep(s);
            if (kflag == -1) { for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1); s.t = s.tn; return IntegratorResult::DT_UNDERFLOW; }
            if (kflag == -2) { for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1),1); s.t = s.tn; return IntegratorResult::CORRECTOR_CONVERGENCE; }

            // stop criterion
            if ((s.tn - s.tout) * s.H < 0.0) continue;

            // Interpolate to tout
            for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), s.L);
            Real S = (s.tout - s.tn) / s.H;
            for (int jb = 1; jb <= s.NQ; ++jb) {
                const int j = s.NQ - jb;
                for (size_type i = 0; i < N; ++i) s.y[i] = s.YH(static_cast<int>(i+1), j+1) + S * s.y[i];
            }
            trace_deuterium_state("final_interp", s, s.y, nullptr, S);
            s.t = s.tout; return IntegratorResult::SUCCESS;
        }
    }
};

} // namespace integrators

#endif // VODE_HPP
