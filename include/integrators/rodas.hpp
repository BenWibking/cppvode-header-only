// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: RODAS Rosenbrock integrator following Hairer/Wanner rodas.f
#ifndef RODAS_HPP
#define RODAS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include "integrator_types.hpp"
#include "linear_algebra.hpp"

namespace integrators {

template<size_type N>
struct RODASState : public IntegratorState<N> {
    int max_steps{100000};
    int method{1};
    bool predictive_controller{true};
    bool autonomous{true};
    bool jacobian_analytic{false};
    Real uround{1.e-16};
    Real hmax{0.0};
    Real fac_min{0.2};
    Real fac_max{6.0};
    Real safe{0.9};
    int n_accept{0};
    int n_reject{0};
    int n_decomp{0};
    int n_solve{0};
    Real last_error{0.0};
    Real min_error{std::numeric_limits<Real>::max()};
    Real max_error{0.0};
    Real min_abs_h{std::numeric_limits<Real>::max()};
    Real max_abs_h{0.0};
    int step_limited_by_fac_min{0};
    int step_limited_by_fac_max{0};

    std::array<Real, N> ynew{};
    std::array<Real, N> dy1{};
    std::array<Real, N> dy{};
    std::array<Real, N> ak1{};
    std::array<Real, N> ak2{};
    std::array<Real, N> ak3{};
    std::array<Real, N> ak4{};
    std::array<Real, N> ak5{};
    std::array<Real, N> ak6{};
    std::array<Real, N> fx{};
    std::array<Real, N> work{};
    std::array<std::array<Real, N>, N> fjac{};
    std::array<std::array<Real, N>, N> e{};
    std::array<int, N> ip{};
};

namespace detail {

struct RODASCoefficients {
    Real a21{}, a31{}, a32{}, a41{}, a42{}, a43{}, a51{}, a52{}, a53{}, a54{};
    Real c21{}, c31{}, c32{}, c41{}, c42{}, c43{}, c51{}, c52{}, c53{}, c54{};
    Real c61{}, c62{}, c63{}, c64{}, c65{};
    Real gamma{}, c2{}, c3{}, c4{}, d1{}, d2{}, d3{}, d4{};
};

INTEGRATORS_HOST_DEVICE inline RODASCoefficients rodas_coefficients(int method) {
    RODASCoefficients c{};
    if (method == 2) {
        c.c2 = 0.3507221;
        c.c3 = 0.2557041;
        c.c4 = 0.6817790;
        c.d1 = 0.2500000000000000e+00;
        c.d2 = -0.6902209999999998e-01;
        c.d3 = -0.9671999999999459e-03;
        c.d4 = -0.8797900000000025e-01;
        c.a21 = 0.1402888400000000e+01;
        c.a31 = 0.6581212688557198e+00;
        c.a32 = -0.1320936088384301e+01;
        c.a41 = 0.7131197445744498e+01;
        c.a42 = 0.1602964143958207e+02;
        c.a43 = -0.5561572550509766e+01;
        c.a51 = 0.2273885722420363e+02;
        c.a52 = 0.6738147284535289e+02;
        c.a53 = -0.3121877493038560e+02;
        c.a54 = 0.7285641833203814e+00;
        c.c21 = -0.5104353600000000e+01;
        c.c31 = -0.2899967805418783e+01;
        c.c32 = 0.4040399359702244e+01;
        c.c41 = -0.3264449927841361e+02;
        c.c42 = -0.9935311008728094e+02;
        c.c43 = 0.4999119122405989e+02;
        c.c51 = -0.7646023087151691e+02;
        c.c52 = -0.2785942120829058e+03;
        c.c53 = 0.1539294840910643e+03;
        c.c54 = 0.1097101866258358e+02;
        c.c61 = -0.7629701586804983e+02;
        c.c62 = -0.2942795630511232e+03;
        c.c63 = 0.1620029695867566e+03;
        c.c64 = 0.2365166903095270e+02;
        c.c65 = -0.7652977706771382e+01;
        c.gamma = 0.2500000000000000e+00;
    } else if (method == 3) {
        c.gamma = 0.25;
        c.c2 = 3.0 * c.gamma;
        c.c3 = 0.21;
        c.c4 = 0.63;
        c.d1 = 0.2500000000000000e+00;
        c.d2 = -0.5000000000000000e+00;
        c.d3 = -0.2350400000000000e-01;
        c.d4 = -0.3620000000000000e-01;
        c.a21 = 0.3000000000000000e+01;
        c.a31 = 0.1831036793486759e+01;
        c.a32 = 0.4955183967433795e+00;
        c.a41 = 0.2304376582692669e+01;
        c.a42 = -0.5249275245743001e-01;
        c.a43 = -0.1176798761832782e+01;
        c.a51 = -0.7170454962423024e+01;
        c.a52 = -0.4741636671481785e+01;
        c.a53 = -0.1631002631330971e+02;
        c.a54 = -0.1062004044111401e+01;
        c.c21 = -0.1200000000000000e+02;
        c.c31 = -0.8791795173947035e+01;
        c.c32 = -0.2207865586973518e+01;
        c.c41 = 0.1081793056857153e+02;
        c.c42 = 0.6780270611428266e+01;
        c.c43 = 0.1953485944642410e+02;
        c.c51 = 0.3419095006749676e+02;
        c.c52 = 0.1549671153725963e+02;
        c.c53 = 0.5474760875964130e+02;
        c.c54 = 0.1416005392148534e+02;
        c.c61 = 0.3462605830930532e+02;
        c.c62 = 0.1530084976114473e+02;
        c.c63 = 0.5699955578662667e+02;
        c.c64 = 0.1840807009793095e+02;
        c.c65 = -0.5714285714285717e+01;
    } else {
        c.c2 = 0.386;
        c.c3 = 0.21;
        c.c4 = 0.63;
        c.d1 = 0.2500000000000000e+00;
        c.d2 = -0.1043000000000000e+00;
        c.d3 = 0.1035000000000000e+00;
        c.d4 = -0.3620000000000023e-01;
        c.a21 = 0.1544000000000000e+01;
        c.a31 = 0.9466785280815826e+00;
        c.a32 = 0.2557011698983284e+00;
        c.a41 = 0.3314825187068521e+01;
        c.a42 = 0.2896124015972201e+01;
        c.a43 = 0.9986419139977817e+00;
        c.a51 = 0.1221224509226641e+01;
        c.a52 = 0.6019134481288629e+01;
        c.a53 = 0.1253708332932087e+02;
        c.a54 = -0.6878860361058950e+00;
        c.c21 = -0.5668800000000000e+01;
        c.c31 = -0.2430093356833875e+01;
        c.c32 = -0.2063599157091915e+00;
        c.c41 = -0.1073529058151375e+00;
        c.c42 = -0.9594562251023355e+01;
        c.c43 = -0.2047028614809616e+02;
        c.c51 = 0.7496443313967647e+01;
        c.c52 = -0.1024680431464352e+02;
        c.c53 = -0.3399990352819905e+02;
        c.c54 = 0.1170890893206160e+02;
        c.c61 = 0.8083246795921522e+01;
        c.c62 = -0.7981132988064893e+01;
        c.c63 = -0.3152159432874371e+02;
        c.c64 = 0.1631930543123136e+02;
        c.c65 = -0.6058818238834054e+01;
        c.gamma = 0.2500000000000000e+00;
    }
    return c;
}

} // namespace detail

template<typename Problem>
class RODAS {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = RODASState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;

private:
    static INTEGRATORS_HOST_DEVICE inline Real rtol_for(const State& s, size_type i) {
        return s.use_vector_tolerances ? s.rtol_vec[i] : s.rtol;
    }

    static INTEGRATORS_HOST_DEVICE inline Real atol_for(const State& s, size_type i) {
        return s.use_vector_tolerances ? s.atol_vec[i] : s.atol;
    }

    static INTEGRATORS_HOST_DEVICE inline void rhs(Real t, const std::array<Real, N>& y,
                                                  std::array<Real, N>& out) {
        Problem::rhs(t, y, out);
    }

    static INTEGRATORS_HOST_DEVICE inline void eval_jacobian(State& s, Real x) {
        if (s.jacobian_analytic && ProblemTraits<Problem>::has_analytic_jacobian) {
            if constexpr (ProblemTraits<Problem>::has_analytic_jacobian) {
                Problem::jacobian(x, s.y, s.fjac);
            }
        } else {
            for (size_type i = 0; i < N; ++i) {
                const Real ysafe = s.y[i];
                const Real delt = std::sqrt(s.uround * std::max(1.e-5, std::abs(ysafe)));
                s.y[i] = ysafe + delt;
                rhs(x, s.y, s.work);
                for (size_type j = 0; j < N; ++j) {
                    s.fjac[j][i] = (s.work[j] - s.dy1[j]) / delt;
                }
                s.y[i] = ysafe;
            }
        }
        s.n_jac += 1;
    }

    static INTEGRATORS_HOST_DEVICE inline int decompose(State& s, Real fac) {
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                s.e[i][j] = -s.fjac[i][j];
            }
            s.e[i][i] += fac;
        }
        const int info = linalg::lu_decomposition<N>(s.e, s.ip);
        if (info == 0) {
            s.n_decomp += 1;
        }
        return info;
    }

    static INTEGRATORS_HOST_DEVICE inline void solve_stage(State& s, const std::array<Real, N>& f,
                                                           std::array<Real, N>& ak,
                                                           const std::array<Real, N>& add,
                                                           Real hd) {
        for (size_type i = 0; i < N; ++i) {
            ak[i] = f[i] + add[i] + hd * s.fx[i];
        }
        linalg::lu_solve<N>(s.e, s.ip, ak);
        s.n_solve += 1;
    }

    static INTEGRATORS_HOST_DEVICE inline Real error_norm(const State& s) {
        Real err = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real sk = atol_for(s, i) + rtol_for(s, i) * std::max(std::abs(s.y[i]), std::abs(s.ynew[i]));
            const Real term = s.ak6[i] / sk;
            err += term * term;
        }
        return std::sqrt(err / static_cast<Real>(N));
    }

public:
    INTEGRATORS_HOST_DEVICE IntegratorResult integrate(ProblemState& problem_state, State& s) {
        if (s.tout == s.t) {
            return IntegratorResult::SUCCESS;
        }
        if (s.method < 1 || s.method > 3 || s.safe <= 0.001 || s.safe >= 1.0 ||
            s.fac_min <= 0.0 || s.fac_max < 1.0) {
            return IntegratorResult::BAD_INPUTS;
        }
        for (size_type i = 0; i < N; ++i) {
            if (atol_for(s, i) <= 0.0 || rtol_for(s, i) <= 10.0 * s.uround) {
                return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
            }
        }

        const auto coeff = detail::rodas_coefficients(s.method);
        const Real posneg = (s.tout >= s.t) ? 1.0 : -1.0;
        const Real hmaxn = std::min(s.hmax == 0.0 ? std::abs(s.tout - s.t) : std::abs(s.hmax),
                                    std::abs(s.tout - s.t));
        Real h = s.dt;
        if (std::abs(h) <= 10.0 * s.uround) {
            h = 1.e-6;
        }
        h = std::min(std::abs(h), hmaxn) * posneg;

        bool reject = false;
        bool last = false;
        int nsing = 0;
        Real hacc = 0.0;
        Real erracc = 1.0;
        Real hopt = h;
        Real x = s.t;

        for (;;) {
            if (s.n_step > s.max_steps) {
                s.t = x;
                s.dt = h;
                return IntegratorResult::TOO_MANY_STEPS;
            }
            if (0.1 * std::abs(h) <= std::abs(x) * s.uround) {
                s.t = x;
                s.dt = h;
                return IntegratorResult::DT_UNDERFLOW;
            }
            if (last) {
                s.t = x;
                s.dt = hopt;
                problem_state = s.y;
                return IntegratorResult::SUCCESS;
            }

            hopt = h;
            if ((x + h * 1.0001 - s.tout) * posneg >= 0.0) {
                h = s.tout - x;
                last = true;
            }

            rhs(x, s.y, s.dy1);
            s.n_rhs += 1;
            eval_jacobian(s, x);
            if (!s.autonomous) {
                const Real delt = std::sqrt(s.uround * std::max(1.e-5, std::abs(x)));
                rhs(x + delt, s.y, s.work);
                for (size_type i = 0; i < N; ++i) {
                    s.fx[i] = (s.work[i] - s.dy1[i]) / delt;
                }
            } else {
                s.fx.fill(0.0);
            }

            for (;;) {
                const Real fac = 1.0 / (h * coeff.gamma);
                if (decompose(s, fac) != 0) {
                    nsing += 1;
                    if (nsing >= 5) {
                        s.t = x;
                        s.dt = h;
                        return IntegratorResult::LU_DECOMPOSITION_ERROR;
                    }
                    h *= 0.5;
                    reject = true;
                    last = false;
                    continue;
                }

                const Real hc21 = coeff.c21 / h;
                const Real hc31 = coeff.c31 / h;
                const Real hc32 = coeff.c32 / h;
                const Real hc41 = coeff.c41 / h;
                const Real hc42 = coeff.c42 / h;
                const Real hc43 = coeff.c43 / h;
                const Real hc51 = coeff.c51 / h;
                const Real hc52 = coeff.c52 / h;
                const Real hc53 = coeff.c53 / h;
                const Real hc54 = coeff.c54 / h;
                const Real hc61 = coeff.c61 / h;
                const Real hc62 = coeff.c62 / h;
                const Real hc63 = coeff.c63 / h;
                const Real hc64 = coeff.c64 / h;
                const Real hc65 = coeff.c65 / h;
                const Real hd1 = s.autonomous ? 0.0 : h * coeff.d1;
                const Real hd2 = s.autonomous ? 0.0 : h * coeff.d2;
                const Real hd3 = s.autonomous ? 0.0 : h * coeff.d3;
                const Real hd4 = s.autonomous ? 0.0 : h * coeff.d4;

                s.work.fill(0.0);
                solve_stage(s, s.dy1, s.ak1, s.work, hd1);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] = s.y[i] + coeff.a21 * s.ak1[i];
                }
                rhs(x + coeff.c2 * h, s.ynew, s.dy);
                for (size_type i = 0; i < N; ++i) {
                    s.work[i] = hc21 * s.ak1[i];
                }
                solve_stage(s, s.dy, s.ak2, s.work, hd2);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] = s.y[i] + coeff.a31 * s.ak1[i] + coeff.a32 * s.ak2[i];
                }
                rhs(x + coeff.c3 * h, s.ynew, s.dy);
                for (size_type i = 0; i < N; ++i) {
                    s.work[i] = hc31 * s.ak1[i] + hc32 * s.ak2[i];
                }
                solve_stage(s, s.dy, s.ak3, s.work, hd3);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] = s.y[i] + coeff.a41 * s.ak1[i] + coeff.a42 * s.ak2[i] +
                                coeff.a43 * s.ak3[i];
                }
                rhs(x + coeff.c4 * h, s.ynew, s.dy);
                for (size_type i = 0; i < N; ++i) {
                    s.work[i] = hc41 * s.ak1[i] + hc42 * s.ak2[i] + hc43 * s.ak3[i];
                }
                solve_stage(s, s.dy, s.ak4, s.work, hd4);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] = s.y[i] + coeff.a51 * s.ak1[i] + coeff.a52 * s.ak2[i] +
                                coeff.a53 * s.ak3[i] + coeff.a54 * s.ak4[i];
                }
                rhs(x + h, s.ynew, s.dy);
                for (size_type i = 0; i < N; ++i) {
                    s.ak6[i] = hc52 * s.ak2[i] + hc54 * s.ak4[i] + hc51 * s.ak1[i] +
                               hc53 * s.ak3[i];
                }
                solve_stage(s, s.dy, s.ak5, s.ak6, 0.0);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] += s.ak5[i];
                }
                rhs(x + h, s.ynew, s.dy);
                for (size_type i = 0; i < N; ++i) {
                    s.work[i] = hc61 * s.ak1[i] + hc62 * s.ak2[i] + hc65 * s.ak5[i] +
                                hc64 * s.ak4[i] + hc63 * s.ak3[i];
                }
                solve_stage(s, s.dy, s.ak6, s.work, 0.0);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] += s.ak6[i];
                }
                s.n_rhs += 5;
                s.n_step += 1;

                const Real err = error_norm(s);
                s.last_error = err;
                s.min_error = std::min(s.min_error, err);
                s.max_error = std::max(s.max_error, err);
                s.min_abs_h = std::min(s.min_abs_h, std::abs(h));
                s.max_abs_h = std::max(s.max_abs_h, std::abs(h));
                const Real raw_fac = std::pow(err, 0.25) / s.safe;
                const Real lower_fac = 1.0 / s.fac_max;
                const Real upper_fac = 1.0 / s.fac_min;
                if (raw_fac < lower_fac) {
                    s.step_limited_by_fac_max += 1;
                } else if (raw_fac > upper_fac) {
                    s.step_limited_by_fac_min += 1;
                }
                const Real fac_step = std::max(lower_fac, std::min(upper_fac, raw_fac));
                Real hnew = h / fac_step;
                if (err <= 1.0) {
                    s.n_accept += 1;
                    if (s.predictive_controller) {
                        if (s.n_accept > 1) {
                            const Real facgus = std::max(1.0 / s.fac_max,
                                std::min(1.0 / s.fac_min,
                                         (hacc / h) * std::pow((err * err) / erracc, 0.25) / s.safe));
                            hnew = h / std::max(fac_step, facgus);
                        }
                        hacc = h;
                        erracc = std::max(1.e-2, err);
                    }
                    s.y = s.ynew;
                    x += h;
                    if (std::abs(hnew) > hmaxn) {
                        hnew = posneg * hmaxn;
                    }
                    if (reject) {
                        hnew = posneg * std::min(std::abs(hnew), std::abs(h));
                    }
                    reject = false;
                    h = hnew;
                    break;
                }

                reject = true;
                last = false;
                h = hnew;
                if (s.n_accept >= 1) {
                    s.n_reject += 1;
                }
            }
        }
    }
};

} // namespace integrators

#endif // RODAS_HPP
