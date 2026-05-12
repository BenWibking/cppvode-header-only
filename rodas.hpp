// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Autonomous ROS2S Rosenbrock integrator
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
struct RODASState {
    Real t{0.0};
    Real tout{0.0};
    Real dt{0.0};
    std::array<Real, N> y{};

    Real rtol{1.e-6};
    Real atol{1.e-12};
    bool use_vector_tolerances{false};
    std::array<Real, N> rtol_vec{};
    std::array<Real, N> atol_vec{};

    int n_step{0};
    int n_rhs{0};
    int n_jac{0};
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

    int max_steps{100000};
    bool predictive_controller{true};
    bool autonomous{true};
    bool jacobian_analytic{false};
    Real uround{1.e-16};
    Real hmax{0.0};
    Real fac_min{0.2};
    Real fac_max{6.0};
    Real safe{0.9};

    std::array<Real, N> ynew{};
    std::array<Real, N> ak1{};
    std::array<Real, N> ak2{};
    std::array<Real, N> work{};
    std::array<std::array<Real, N>, N> fjac{};
    std::array<std::array<Real, N>, N> e{};
    std::array<Real, N> dy{};
    std::array<int, N> ip{};

    INTEGRATORS_HOST_DEVICE std::array<Real, N>& rhs_scratch(std::array<Real, N>&) {
        return dy;
    }

    INTEGRATORS_HOST_DEVICE std::array<std::array<Real, N>, N>& matrix() {
        return e;
    }
};

namespace detail {

struct ROS2SCoefficients {
    static constexpr Real gamma = 0.292893218813452;
    static constexpr Real ct2 = 0.585786437626905;
    static constexpr Real a21 = 2.0000000000000036;
    static constexpr Real a31 = 6.828427124746214;
    static constexpr Real a32 = 3.4142135623731007;
    static constexpr Real c21 = -6.828427124746214;
    static constexpr Real c31 = -10.949747468305889;
    static constexpr Real c32 = -7.535533905932761;
    static constexpr Real b1 = 6.828427124746214;
    static constexpr Real b2 = 3.414213562373101;
    static constexpr Real b3 = 1.0;
    static constexpr Real e1 = -0.23570226039551292;
    static constexpr Real e2 = -0.23570226039551567;
    static constexpr Real e3 = -0.13807118745769906;
};

} // namespace detail

template<typename Problem>
class RODAS {
public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    using State = RODASState<N>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;

private:
    static INTEGRATORS_HOST_DEVICE Real rtol_for(const State& s, size_type i) {
        return s.use_vector_tolerances ? s.rtol_vec[i] : s.rtol;
    }

    static INTEGRATORS_HOST_DEVICE Real atol_for(const State& s, size_type i) {
        return s.use_vector_tolerances ? s.atol_vec[i] : s.atol;
    }

    static INTEGRATORS_HOST_DEVICE void rhs(Real t, const std::array<Real, N>& y,
                                            std::array<Real, N>& out) {
        Problem::rhs(t, y, out);
    }

    static INTEGRATORS_HOST_DEVICE void eval_jacobian(State& s, Real x) {
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
                    s.fjac[j][i] = (s.work[j] - s.ak1[j]) / delt;
                }
                s.y[i] = ysafe;
            }
        }
        s.n_jac += 1;
    }

    static INTEGRATORS_HOST_DEVICE int decompose(State& s, Real fac) {
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                s.matrix()[i][j] = -s.fjac[i][j];
            }
            s.matrix()[i][i] += fac;
        }
        const int info = linalg::lu_decomposition<N>(s.matrix(), s.ip);
        if (info == 0) {
            s.n_decomp += 1;
        }
        return info;
    }

    static INTEGRATORS_HOST_DEVICE void solve(State& s, std::array<Real, N>& ak) {
        linalg::lu_solve<N>(s.matrix(), s.ip, ak);
        s.n_solve += 1;
    }

    static INTEGRATORS_HOST_DEVICE void record_error_stats(State& s, Real err, Real h,
                                                           Real raw_fac, Real lower_fac,
                                                           Real upper_fac) {
        s.last_error = err;
        s.min_error = std::min(s.min_error, err);
        s.max_error = std::max(s.max_error, err);
        s.min_abs_h = std::min(s.min_abs_h, std::abs(h));
        s.max_abs_h = std::max(s.max_abs_h, std::abs(h));
        if (raw_fac < lower_fac) {
            s.step_limited_by_fac_max += 1;
        } else if (raw_fac > upper_fac) {
            s.step_limited_by_fac_min += 1;
        }
    }

    static INTEGRATORS_HOST_DEVICE void record_rhs(State& s, int count = 1) {
        s.n_rhs += count;
    }

    static INTEGRATORS_HOST_DEVICE void record_step(State& s, int n_step) {
        s.n_step = n_step;
    }

    static INTEGRATORS_HOST_DEVICE void record_accept(State& s, int n_accept) {
        s.n_accept = n_accept;
    }

    static INTEGRATORS_HOST_DEVICE void record_reject(State& s) {
        s.n_reject += 1;
    }

    static INTEGRATORS_HOST_DEVICE Real error_norm(const State& s) {
        Real err = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real sk = atol_for(s, i) + rtol_for(s, i) * std::max(std::abs(s.y[i]), std::abs(s.ynew[i]));
            const Real term = s.work[i] / sk;
            err += term * term;
        }
        return std::sqrt(err / static_cast<Real>(N));
    }

public:
    INTEGRATORS_HOST_DEVICE IntegratorResult integrate(ProblemState& problem_state, State& s) {
        using C = detail::ROS2SCoefficients;

        if (s.tout == s.t) {
            return IntegratorResult::SUCCESS;
        }
        if (!s.autonomous || s.safe <= 0.001 || s.safe >= 1.0 ||
            s.fac_min <= 0.0 || s.fac_max < 1.0) {
            return IntegratorResult::BAD_INPUTS;
        }
        for (size_type i = 0; i < N; ++i) {
            if (atol_for(s, i) <= 0.0 || rtol_for(s, i) <= 10.0 * s.uround) {
                return IntegratorResult::TOO_MUCH_ACCURACY_REQUESTED;
            }
        }

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
        int n_step = 0;
        int n_accept = 0;
        n_step = s.n_step;
        n_accept = s.n_accept;

        for (;;) {
            if (n_step > s.max_steps) {
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

            if (!(s.jacobian_analytic && ProblemTraits<Problem>::has_analytic_jacobian)) {
                rhs(x, s.y, s.ak1);
                record_rhs(s);
            }
            eval_jacobian(s, x);

            for (;;) {
                const Real fac = 1.0 / (h * C::gamma);
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

                rhs(x, s.y, s.ak1);
                solve(s, s.ak1);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] = s.y[i] + C::a21 * s.ak1[i];
                    s.ak2[i] = (C::c21 / h) * s.ak1[i];
                }
                auto& rhs_tmp = s.rhs_scratch(s.ynew);
                rhs(x + C::ct2 * h, s.ynew, rhs_tmp);
                for (size_type i = 0; i < N; ++i) {
                    s.ak2[i] += rhs_tmp[i];
                }
                solve(s, s.ak2);

                for (size_type i = 0; i < N; ++i) {
                    s.ynew[i] = s.y[i] + C::a31 * s.ak1[i] + C::a32 * s.ak2[i];
                    s.work[i] = (C::c31 * s.ak1[i] + C::c32 * s.ak2[i]) / h;
                }
                auto& rhs_tmp_stage3 = s.rhs_scratch(s.ynew);
                rhs(x + h, s.ynew, rhs_tmp_stage3);
                for (size_type i = 0; i < N; ++i) {
                    s.work[i] += rhs_tmp_stage3[i];
                }
                solve(s, s.work);

                for (size_type i = 0; i < N; ++i) {
                    const Real ak3i = s.work[i];
                    s.ynew[i] = s.y[i] + C::b1 * s.ak1[i] + C::b2 * s.ak2[i] + C::b3 * ak3i;
                    s.work[i] = C::e1 * s.ak1[i] + C::e2 * s.ak2[i] + C::e3 * ak3i;
                }
                record_rhs(s, 3);
                n_step += 1;
                record_step(s, n_step);

                const Real err = error_norm(s);
                const Real raw_fac = std::cbrt(err) / s.safe;
                const Real lower_fac = 1.0 / s.fac_max;
                const Real upper_fac = 1.0 / s.fac_min;
                record_error_stats(s, err, h, raw_fac, lower_fac, upper_fac);
                const Real fac_step = std::max(lower_fac, std::min(upper_fac, raw_fac));
                Real hnew = h / fac_step;
                if (err <= 1.0) {
                    n_accept += 1;
                    record_accept(s, n_accept);
                    if (s.predictive_controller) {
                        if (n_accept > 1) {
                            const Real facgus = std::max(1.0 / s.fac_max,
                                std::min(1.0 / s.fac_min,
                                         (hacc / h) *
                                         std::cbrt((err * err) / erracc) /
                                         s.safe));
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
                if (n_accept >= 1) {
                    record_reject(s);
                }
            }
        }
    }
};

template<typename Problem>
using ROS2S = RODAS<Problem>;

} // namespace integrators

#endif // RODAS_HPP
