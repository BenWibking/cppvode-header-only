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

template <size_type N, bool AnalyticJacobianOnly> struct RODASJacobianStorage {
    std::array<std::array<Real, N>, N> fjac{};
};

template <size_type N> struct RODASJacobianStorage<N, true> {};

template <size_type N, bool IncludeRhsScratch> struct RODASRhsScratchStorage {
    std::array<Real, N> dy{};

    INTEGRATORS_HOST_DEVICE std::array<Real, N> &rhs_scratch(std::array<Real, N> &) { return dy; }
};

template <size_type N> struct RODASRhsScratchStorage<N, false> {
    INTEGRATORS_HOST_DEVICE std::array<Real, N> &rhs_scratch(std::array<Real, N> &alias) {
        return alias;
    }
};

template <size_type N, bool ExternalMatrixStorage> struct RODASMatrixStorage {
    std::array<std::array<Real, N>, N> e{};

    INTEGRATORS_HOST_DEVICE std::array<std::array<Real, N>, N> &matrix() { return e; }
};

template <size_type N> struct RODASMatrixStorage<N, true> {
    std::array<std::array<Real, N>, N> *external_e{nullptr};

    INTEGRATORS_HOST_DEVICE std::array<std::array<Real, N>, N> &matrix() { return *external_e; }
};

template <size_type N, bool StaticTolerances> struct RODASToleranceStorage {
    Real rtol{1.e-6};
    Real atol{1.e-12};
    bool use_vector_tolerances{false};
    std::array<Real, N> rtol_vec{};
    std::array<Real, N> atol_vec{};
};

template <size_type N> struct RODASToleranceStorage<N, true> {};

template <bool CollectStats> struct RODASStatsStorage {
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
};

template <> struct RODASStatsStorage<false> {};

template <size_type N, bool AnalyticJacobianOnly = false, bool ExternalMatrixStorage = false,
          bool IncludeRhsScratch = true, bool StaticTolerances = false, bool CollectStats = true>
struct RODASState : public RODASToleranceStorage<N, StaticTolerances>,
                    public RODASStatsStorage<CollectStats>,
                    public RODASJacobianStorage<N, AnalyticJacobianOnly>,
                    public RODASMatrixStorage<N, ExternalMatrixStorage>,
                    public RODASRhsScratchStorage<N, IncludeRhsScratch> {
    Real t{0.0};
    Real tout{0.0};
    Real dt{0.0};
    std::array<Real, N> y{};

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
    std::array<Real, N> ak3{};
    std::array<Real, N> work{};
    std::array<int, N> ip{};
};

namespace detail {

enum class RosenbrockMethod { ROS2S, SanduA, SanduB, SanduD };

enum class RosenbrockErrorEstimator { StageWeights, FirstStage, EmbeddedWeights };

template <RosenbrockMethod Method> struct RosenbrockCoefficients;

template <> struct RosenbrockCoefficients<RosenbrockMethod::ROS2S> {
    static constexpr int stages = 3;
    static constexpr RosenbrockErrorEstimator error_estimator =
        RosenbrockErrorEstimator::StageWeights;
    static constexpr Real gamma = 0.292893218813452;
    static constexpr std::array<Real, 4> alpha{0.0, 0.585786437626905, 1.0, 0.0};
    static constexpr std::array<std::array<Real, 4>, 4> a{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{2.0000000000000036, 0.0, 0.0, 0.0}},
        {{6.828427124746214, 3.4142135623731007, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
    }};
    static constexpr std::array<std::array<Real, 4>, 4> c{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{-6.828427124746214, 0.0, 0.0, 0.0}},
        {{-10.949747468305889, -7.535533905932761, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
    }};
    static constexpr std::array<Real, 4> m{6.828427124746214, 3.414213562373101, 1.0, 0.0};
    static constexpr std::array<Real, 4> err{-0.23570226039551292, -0.23570226039551567,
                                             -0.13807118745769906, 0.0};
    static constexpr Real first_stage_weight = 0.0;
};

template <> struct RosenbrockCoefficients<RosenbrockMethod::SanduA> {
    static constexpr int stages = 3;
    static constexpr RosenbrockErrorEstimator error_estimator =
        RosenbrockErrorEstimator::FirstStage;
    static constexpr Real gamma = 0.7886751345948129;              // (3 + sqrt(3)) / 6
    static constexpr Real first_stage_weight = 1.2679491924311228; // 1 / gamma
    static constexpr std::array<Real, 4> alpha{0.0, 1.0, 1.0, 0.0};
    static constexpr std::array<std::array<Real, 4>, 4> a{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{1.2679491924311228, 0.0, 0.0, 0.0}},
        {{1.2679491924311228, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
    }};
    static constexpr std::array<std::array<Real, 4>, 4> c{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.9282032302755092, 0.0, 0.0, 0.0}},
        {{-0.4641016151377544, -0.4641016151377544, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
    }};
    static constexpr std::array<Real, 4> m{1.2679491924311228, 0.0, 1.0, 0.0};
    static constexpr std::array<Real, 4> err{0.0, 0.0, 0.0, 0.0};
};

template <> struct RosenbrockCoefficients<RosenbrockMethod::SanduB> {
    static constexpr int stages = 3;
    static constexpr RosenbrockErrorEstimator error_estimator =
        RosenbrockErrorEstimator::FirstStage;
    static constexpr Real gamma = 0.7886751345948129;              // (3 + sqrt(3)) / 6
    static constexpr Real first_stage_weight = 1.2679491924311228; // 1 / gamma
    static constexpr std::array<Real, 4> alpha{0.0, 1.0, 1.0, 0.0};
    static constexpr std::array<std::array<Real, 4>, 4> a{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{1.2679491924311228, 0.0, 0.0, 0.0}},
        {{1.2679491924311228, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
    }};
    static constexpr std::array<std::array<Real, 4>, 4> c{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
        {{-0.5358983848622456, -0.7320508075688772, 0.0, 0.0}},
        {{0.0, 0.0, 0.0, 0.0}},
    }};
    static constexpr std::array<Real, 4> m{1.2679491924311228, 0.0, 1.0, 0.0};
    static constexpr std::array<Real, 4> err{0.0, 0.0, 0.0, 0.0};
};

template <> struct RosenbrockCoefficients<RosenbrockMethod::SanduD> {
    static constexpr int stages = 4;
    static constexpr RosenbrockErrorEstimator error_estimator =
        RosenbrockErrorEstimator::EmbeddedWeights;
    static constexpr Real gamma = 0.5;
    static constexpr Real first_stage_weight = 0.0;
    static constexpr std::array<Real, 4> alpha{0.0, 1.0, 1.0, 1.0};
    static constexpr std::array<std::array<Real, 4>, 4> a{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{2.0, 0.0, 0.0, 0.0}},
        {{2.0, 0.0, 0.0, 0.0}},
        {{2.0, 0.0, 0.0, 0.0}},
    }};
    static constexpr std::array<std::array<Real, 4>, 4> c{{
        {{0.0, 0.0, 0.0, 0.0}},
        {{-1.3333333333333333, 0.0, 0.0, 0.0}},
        {{-3.3333333333333333, -2.0, 0.0, 0.0}},
        {{-0.5, 0.0, 1.5, 0.0}},
    }};
    static constexpr std::array<Real, 4> m{2.0, 0.0, 0.0, 1.0};
    static constexpr std::array<Real, 4> err{-0.6666666666666666, -1.0, -1.0, 1.3333333333333333};
};

} // namespace detail

template <typename Problem, detail::RosenbrockMethod Method = detail::RosenbrockMethod::ROS2S,
          bool AnalyticJacobianOnly = false, bool CollectStats = true, bool AllowPivoting = true,
          bool UsePrimordialGiftFactorization = false, bool ExternalMatrixStorage = false,
          bool CompactRhsScratch = false, bool StaticTolerances = false>
class RosenbrockIntegrator {
  public:
    static constexpr size_type N = ProblemTraits<Problem>::neqs;
    static_assert(!AnalyticJacobianOnly || ProblemTraits<Problem>::has_analytic_jacobian,
                  "Analytic-only ROS2S requires Problem::jacobian");
    static_assert(!StaticTolerances || ProblemTraits<Problem>::has_ros2s_static_tolerances,
                  "Static-tolerance ROS2S requires Problem::ros2s_rtol and "
                  "Problem::ros2s_atol");
    static constexpr bool include_rhs_scratch =
        !(CompactRhsScratch && ProblemTraits<Problem>::rhs_allows_input_output_alias);
    using State = RODASState<N, AnalyticJacobianOnly, ExternalMatrixStorage, include_rhs_scratch,
                             StaticTolerances, CollectStats>;
    using ProblemState = typename ProblemTraits<Problem>::state_type;

  private:
    static INTEGRATORS_HOST_DEVICE Real rtol_for(const State &s, size_type i) {
        if constexpr (StaticTolerances) {
            (void)s;
            return Problem::ros2s_rtol(i);
        } else {
            return s.use_vector_tolerances ? s.rtol_vec[i] : s.rtol;
        }
    }

    static INTEGRATORS_HOST_DEVICE Real atol_for(const State &s, size_type i) {
        if constexpr (StaticTolerances) {
            (void)s;
            return Problem::ros2s_atol(i);
        } else {
            return s.use_vector_tolerances ? s.atol_vec[i] : s.atol;
        }
    }

    static INTEGRATORS_HOST_DEVICE void rhs(Real t, const std::array<Real, N> &y,
                                            std::array<Real, N> &out) {
        Problem::rhs(t, y, out);
    }

    static constexpr bool uses_analytic_jacobian() {
        return AnalyticJacobianOnly || ProblemTraits<Problem>::has_analytic_jacobian;
    }

    static INTEGRATORS_HOST_DEVICE void eval_jacobian(State &s, Real x) {
        if constexpr (AnalyticJacobianOnly) {
            Problem::jacobian(x, s.y, s.matrix());
        } else if (s.jacobian_analytic && uses_analytic_jacobian()) {
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
        if constexpr (CollectStats) {
            s.n_jac += 1;
        }
    }

    static INTEGRATORS_HOST_DEVICE int decompose(State &s, Real fac) {
        static_assert(!UsePrimordialGiftFactorization || N == 15,
                      "GIFT factorization path is specialized for the primordial "
                      "15x15 system");
        for (size_type i = 0; i < N; ++i) {
            for (size_type j = 0; j < N; ++j) {
                if constexpr (AnalyticJacobianOnly) {
                    s.matrix()[i][j] = -s.matrix()[i][j];
                } else {
                    s.matrix()[i][j] = -s.fjac[i][j];
                }
            }
            s.matrix()[i][i] += fac;
        }
        int info = 0;
        if constexpr (UsePrimordialGiftFactorization) {
            info = linalg::primordial_gift_lu_decomposition<N>(s.matrix(), s.ip);
        } else {
            info = linalg::lu_decomposition<N, AllowPivoting>(s.matrix(), s.ip);
        }
        if constexpr (CollectStats) {
            if (info == 0) {
                s.n_decomp += 1;
            }
        }
        return info;
    }

    static INTEGRATORS_HOST_DEVICE void solve(State &s, std::array<Real, N> &ak) {
        if constexpr (UsePrimordialGiftFactorization) {
            linalg::primordial_gift_lu_solve<N>(s.matrix(), s.ip, ak);
        } else {
            linalg::lu_solve<N, AllowPivoting>(s.matrix(), s.ip, ak);
        }
        if constexpr (CollectStats) {
            s.n_solve += 1;
        }
    }

    static INTEGRATORS_HOST_DEVICE std::array<Real, N> &stage_vector(State &s, int stage) {
        switch (stage) {
        case 0:
            return s.ak1;
        case 1:
            return s.ak2;
        case 2:
            return s.ak3;
        default:
            return s.work;
        }
    }

    template <typename C>
    static INTEGRATORS_HOST_DEVICE void compute_stage(State &s, Real x, Real h, int stage) {
        auto &ak = stage_vector(s, stage);
        for (size_type i = 0; i < N; ++i) {
            Real yi = s.y[i];
            for (int j = 0; j < stage; ++j) {
                yi += C::a[stage][j] * stage_vector(s, j)[i];
            }
            s.ynew[i] = yi;
        }

        auto &rhs_tmp = s.rhs_scratch(s.ynew);
        rhs(x + C::alpha[stage] * h, s.ynew, rhs_tmp);
        for (size_type i = 0; i < N; ++i) {
            Real rhs_i = rhs_tmp[i];
            for (int j = 0; j < stage; ++j) {
                rhs_i += (C::c[stage][j] / h) * stage_vector(s, j)[i];
            }
            ak[i] = rhs_i;
        }
        solve(s, ak);
    }

    template <typename C> static INTEGRATORS_HOST_DEVICE void form_solution_and_error(State &s) {
        for (size_type i = 0; i < N; ++i) {
            Real solution_i = s.y[i];
            for (int j = 0; j < C::stages; ++j) {
                solution_i += C::m[j] * stage_vector(s, j)[i];
            }

            Real error_i = 0.0;
            if constexpr (C::error_estimator == detail::RosenbrockErrorEstimator::FirstStage) {
                error_i = solution_i - (s.y[i] + C::first_stage_weight * s.ak1[i]);
            } else {
                for (int j = 0; j < C::stages; ++j) {
                    error_i += C::err[j] * stage_vector(s, j)[i];
                }
            }

            s.ynew[i] = solution_i;
            s.work[i] = error_i;
        }
    }

    template <typename C> static INTEGRATORS_HOST_DEVICE Real controller_factor(Real err) {
        if constexpr (C::error_estimator == detail::RosenbrockErrorEstimator::FirstStage) {
            return std::sqrt(err);
        } else {
            return std::cbrt(err);
        }
    }

    static INTEGRATORS_HOST_DEVICE void record_error_stats(State &s, Real err, Real h, Real raw_fac,
                                                           Real lower_fac, Real upper_fac) {
        if constexpr (CollectStats) {
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
    }

    static INTEGRATORS_HOST_DEVICE void record_rhs(State &s, int count = 1) {
        if constexpr (CollectStats) {
            s.n_rhs += count;
        }
    }

    static INTEGRATORS_HOST_DEVICE void record_step(State &s, int n_step) {
        if constexpr (CollectStats) {
            s.n_step = n_step;
        }
    }

    static INTEGRATORS_HOST_DEVICE void record_accept(State &s, int n_accept) {
        if constexpr (CollectStats) {
            s.n_accept = n_accept;
        }
    }

    static INTEGRATORS_HOST_DEVICE void record_reject(State &s) {
        if constexpr (CollectStats) {
            s.n_reject += 1;
        }
    }

    static INTEGRATORS_HOST_DEVICE Real error_norm(const State &s) {
        Real err = 0.0;
        for (size_type i = 0; i < N; ++i) {
            const Real sk =
                atol_for(s, i) + rtol_for(s, i) * std::max(std::abs(s.y[i]), std::abs(s.ynew[i]));
            const Real term = s.work[i] / sk;
            err += term * term;
        }
        return std::sqrt(err / static_cast<Real>(N));
    }

  public:
    INTEGRATORS_HOST_DEVICE IntegratorResult integrate(ProblemState &problem_state, State &s) {
        using C = detail::RosenbrockCoefficients<Method>;

        if (s.tout == s.t) {
            return IntegratorResult::SUCCESS;
        }
        if (!s.autonomous || s.safe <= 0.001 || s.safe >= 1.0 || s.fac_min <= 0.0 ||
            s.fac_max < 1.0) {
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
        if constexpr (CollectStats) {
            n_step = s.n_step;
            n_accept = s.n_accept;
        }

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

            if (!AnalyticJacobianOnly && !(s.jacobian_analytic && uses_analytic_jacobian())) {
                rhs(x, s.y, s.ak1);
                record_rhs(s);
            }
            if constexpr (!AnalyticJacobianOnly) {
                eval_jacobian(s, x);
            }

            for (;;) {
                const Real fac = 1.0 / (h * C::gamma);
                if constexpr (AnalyticJacobianOnly) {
                    eval_jacobian(s, x);
                }
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

                for (int stage = 0; stage < C::stages; ++stage) {
                    compute_stage<C>(s, x, h, stage);
                }
                form_solution_and_error<C>(s);
                record_rhs(s, C::stages);
                n_step += 1;
                record_step(s, n_step);

                const Real err = error_norm(s);
                const Real raw_fac = controller_factor<C>(err) / s.safe;
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
                            const Real facgus = std::max(
                                1.0 / s.fac_max,
                                std::min(1.0 / s.fac_min,
                                         (hacc / h) * controller_factor<C>((err * err) / erracc) /
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

template <typename Problem, bool AnalyticJacobianOnly = false, bool CollectStats = true,
          bool AllowPivoting = true, bool UsePrimordialGiftFactorization = false,
          bool ExternalMatrixStorage = false, bool CompactRhsScratch = false,
          bool StaticTolerances = false>
using ROS2S = RosenbrockIntegrator<Problem, detail::RosenbrockMethod::ROS2S, AnalyticJacobianOnly,
                                   CollectStats, AllowPivoting, UsePrimordialGiftFactorization,
                                   ExternalMatrixStorage, CompactRhsScratch, StaticTolerances>;

template <typename Problem, bool AnalyticJacobianOnly = false, bool CollectStats = true,
          bool AllowPivoting = true, bool UsePrimordialGiftFactorization = false,
          bool ExternalMatrixStorage = false, bool CompactRhsScratch = false,
          bool StaticTolerances = false>
using RosenbrockSanduA =
    RosenbrockIntegrator<Problem, detail::RosenbrockMethod::SanduA, AnalyticJacobianOnly,
                         CollectStats, AllowPivoting, UsePrimordialGiftFactorization,
                         ExternalMatrixStorage, CompactRhsScratch, StaticTolerances>;

template <typename Problem, bool AnalyticJacobianOnly = false, bool CollectStats = true,
          bool AllowPivoting = true, bool UsePrimordialGiftFactorization = false,
          bool ExternalMatrixStorage = false, bool CompactRhsScratch = false,
          bool StaticTolerances = false>
using RosenbrockSanduB =
    RosenbrockIntegrator<Problem, detail::RosenbrockMethod::SanduB, AnalyticJacobianOnly,
                         CollectStats, AllowPivoting, UsePrimordialGiftFactorization,
                         ExternalMatrixStorage, CompactRhsScratch, StaticTolerances>;

template <typename Problem, bool AnalyticJacobianOnly = false, bool CollectStats = true,
          bool AllowPivoting = true, bool UsePrimordialGiftFactorization = false,
          bool ExternalMatrixStorage = false, bool CompactRhsScratch = false,
          bool StaticTolerances = false>
using RosenbrockSanduD =
    RosenbrockIntegrator<Problem, detail::RosenbrockMethod::SanduD, AnalyticJacobianOnly,
                         CollectStats, AllowPivoting, UsePrimordialGiftFactorization,
                         ExternalMatrixStorage, CompactRhsScratch, StaticTolerances>;

template <typename Problem, bool AnalyticJacobianOnly = false, bool CollectStats = true,
          bool AllowPivoting = true, bool UsePrimordialGiftFactorization = false,
          bool ExternalMatrixStorage = false, bool CompactRhsScratch = false,
          bool StaticTolerances = false>
using RODAS = ROS2S<Problem, AnalyticJacobianOnly, CollectStats, AllowPivoting,
                    UsePrimordialGiftFactorization, ExternalMatrixStorage, CompactRhsScratch,
                    StaticTolerances>;

template <typename Problem> using ROS2SAnalytic = ROS2S<Problem, true>;

} // namespace integrators

#endif // RODAS_HPP
