// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Primordial chemistry one-zone collapse test ported from AMReX Microphysics
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <integrators/rodas.hpp>
#include <integrators/primordial_chem.hpp>
#include <integrators/vode.hpp>

#if defined(__CUDACC__)
#define PRIMORDIAL_CHEM_PROBE_HOST_DEVICE __host__ __device__
#else
#define PRIMORDIAL_CHEM_PROBE_HOST_DEVICE
#endif

#ifndef PRIMORDIAL_CHEM_ROS2S_COMPACT_ANALYTIC_JACOBIAN
#define PRIMORDIAL_CHEM_ROS2S_COMPACT_ANALYTIC_JACOBIAN 0
#endif

#ifndef PRIMORDIAL_CHEM_ROS2S_SPECIALIZE_STATS
#define PRIMORDIAL_CHEM_ROS2S_SPECIALIZE_STATS 0
#endif

#ifndef PRIMORDIAL_CHEM_ROS2S_ALLOW_PIVOTING
#define PRIMORDIAL_CHEM_ROS2S_ALLOW_PIVOTING 1
#endif

#ifndef PRIMORDIAL_CHEM_ROS2S_GIFT_FACTORIZATION
#define PRIMORDIAL_CHEM_ROS2S_GIFT_FACTORIZATION 0
#endif

#ifndef PRIMORDIAL_CHEM_ROS2S_EXTERNAL_MATRIX
#define PRIMORDIAL_CHEM_ROS2S_EXTERNAL_MATRIX 0
#endif

#ifndef PRIMORDIAL_CHEM_ROS2S_COMPACT_RHS_SCRATCH
#define PRIMORDIAL_CHEM_ROS2S_COMPACT_RHS_SCRATCH 0
#endif

#ifndef PRIMORDIAL_CHEM_ROS2S_STATIC_TOLERANCES
#define PRIMORDIAL_CHEM_ROS2S_STATIC_TOLERANCES 0
#endif

#ifndef PRIMORDIAL_CHEM_ACTIVE_KERNELS_ONLY
#define PRIMORDIAL_CHEM_ACTIVE_KERNELS_ONLY 0
#endif

#ifndef PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_THREADS
#define PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_THREADS 0
#endif

#ifndef PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_MIN_BLOCKS
#define PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_MIN_BLOCKS 0
#endif

#if defined(__CUDACC__) && PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_THREADS > 0 &&                 \
    PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_MIN_BLOCKS > 0
#define PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS                                                  \
    __launch_bounds__(PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_THREADS,                           \
                      PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_MIN_BLOCKS)
#elif defined(__CUDACC__) && PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_THREADS > 0
#define PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS                                                  \
    __launch_bounds__(PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS_THREADS)
#else
#define PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS
#endif

namespace pc = integrators::primordial_chem;

namespace {

using RODASMatrix = std::array<std::array<integrators::Real, pc::neqs>, pc::neqs>;

constexpr integrators::Real tff_reduc = 1.0e-1;
constexpr int nsteps = 1000;
constexpr integrators::Real temperature = 1.0e2;
constexpr integrators::Real rtol_spec = 1.0e-4;
constexpr integrators::Real atol_spec = 1.0e-4;
constexpr integrators::Real rtol_enuc = 1.0e-6;
constexpr integrators::Real atol_enuc = 1.0e-6;
constexpr integrators::Real reference_thermodynamic_rtol = 1.0e-4;
constexpr int default_grid_dim = 1;
constexpr int default_cuda_threads_per_block = 128;
constexpr int perturbation_interval = 20;
constexpr integrators::Real perturbation_amplitude = 0.1;
constexpr integrators::Real state_validity_floor_slop = 10.0;
constexpr bool compact_ros2s_analytic_jacobian =
    PRIMORDIAL_CHEM_ROS2S_COMPACT_ANALYTIC_JACOBIAN != 0;
constexpr bool specialize_ros2s_stats = PRIMORDIAL_CHEM_ROS2S_SPECIALIZE_STATS != 0;
constexpr bool ros2s_allow_pivoting = PRIMORDIAL_CHEM_ROS2S_ALLOW_PIVOTING != 0;
constexpr bool ros2s_gift_factorization = PRIMORDIAL_CHEM_ROS2S_GIFT_FACTORIZATION != 0;
constexpr bool ros2s_external_matrix = PRIMORDIAL_CHEM_ROS2S_EXTERNAL_MATRIX != 0;
constexpr bool ros2s_compact_rhs_scratch = PRIMORDIAL_CHEM_ROS2S_COMPACT_RHS_SCRATCH != 0;
constexpr bool ros2s_static_tolerances = PRIMORDIAL_CHEM_ROS2S_STATIC_TOLERANCES != 0;
constexpr bool active_kernels_only = PRIMORDIAL_CHEM_ACTIVE_KERNELS_ONLY != 0;

struct ToleranceConfig {
    integrators::Real rtol_spec_value{rtol_spec};
    integrators::Real atol_spec_value{atol_spec};
    integrators::Real rtol_enuc_value{rtol_enuc};
    integrators::Real atol_enuc_value{atol_enuc};
};

ToleranceConfig runtime_tolerances{};

#if defined(__CUDACC__)
__device__ __constant__ ToleranceConfig device_tolerances;
#endif

PRIMORDIAL_CHEM_PROBE_HOST_DEVICE integrators::Real active_rtol_spec() {
#if defined(__CUDA_ARCH__)
    return device_tolerances.rtol_spec_value;
#else
    return runtime_tolerances.rtol_spec_value;
#endif
}

PRIMORDIAL_CHEM_PROBE_HOST_DEVICE integrators::Real active_atol_spec() {
#if defined(__CUDA_ARCH__)
    return device_tolerances.atol_spec_value;
#else
    return runtime_tolerances.atol_spec_value;
#endif
}

PRIMORDIAL_CHEM_PROBE_HOST_DEVICE integrators::Real active_rtol_enuc() {
#if defined(__CUDA_ARCH__)
    return device_tolerances.rtol_enuc_value;
#else
    return runtime_tolerances.rtol_enuc_value;
#endif
}

PRIMORDIAL_CHEM_PROBE_HOST_DEVICE integrators::Real active_atol_enuc() {
#if defined(__CUDA_ARCH__)
    return device_tolerances.atol_enuc_value;
#else
    return runtime_tolerances.atol_enuc_value;
#endif
}

enum class IntegratorChoice {
    VODE,
    ROS2S,
    ROSENBROCK_SANDU_A,
    ROSENBROCK_SANDU_B,
    ROSENBROCK_SANDU_C,
    ROSENBROCK_SANDU_D
};

constexpr std::array<integrators::Real, pc::NumSpec> initial_number_densities{
    1.0e-4, 1.0e-4, 1.0e0,  1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40,
    1.0e-40, 1.0e-6, 1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40, 0.0775};

// Reference generated with ROS2S, analytic Jacobian, --rtol 1e-9 --atol 1e-10.
constexpr std::array<integrators::Real, pc::NumSpec> reference_number_densities{
    19898.111435054092,    19883.725397617425,    1.6186350992258595e17,
    1.7789352113556338,   6.3053138754138041e-12, 33.348889152618248,
    16.164972648007762,   1.0000000000015343e-100, 3.379071096392231e17,
    5.3122299092018002e-19, 319.268056479172,     2.7107590622924508e-60,
    6.3498970553404515e-12, 6.4913402847062456e16};

// Reference comparison tolerances. The deuterium channels are especially
// sensitive to fused multiply-add contraction in the generated chemistry rates.
constexpr std::array<integrators::Real, pc::NumSpec> reference_species_rtol{
    1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-4, 10.0, 1.0e-3,
    1.0e-4, 1.0e-4, 1.0e-4, 10.0, 1.0e-4, 1.0e-4, 1.0e-4};

constexpr std::array<bool, pc::NumSpec> deuterium_bearing_species{
    false, false, false, false, true, true, false,
    true,  false, true,  true,  false, false, false};

constexpr integrators::Real reference_temperature = 3032.8601488545046;
constexpr integrators::Real reference_eint = 2.7216856652474249e11;
constexpr integrators::Real reference_rho = 1.836285633166796e-6;

struct CollapseResult {
    pc::burn_t initial_state{};
    pc::burn_t final_state{};
    integrators::Real time{};
    int completed_steps{};
    int failed_step{-1};
    integrators::IntegratorResult result{integrators::IntegratorResult::SUCCESS};
};

struct IntegratorWorkStats {
    std::uint64_t internal_steps{};
    std::uint64_t rhs_calls{};
    std::uint64_t jacobian_calls{};
    std::uint64_t decompositions{};
    std::uint64_t linear_solves{};
    std::uint64_t accepted_steps{};
    std::uint64_t rejected_steps{};
    std::uint64_t negative_rejected_steps{};
    std::uint64_t error_failures{};
};

struct NonpositivityStats {
    std::array<std::uint64_t, pc::neqs> cumulative{};
    std::array<std::uint64_t, pc::neqs> last_step{};
};

struct PrimordialChemRos2sStaticTolerances : pc::PrimordialChem {
    INTEGRATORS_HOST_DEVICE static constexpr integrators::Real ros2s_rtol(integrators::size_type i) {
        return i == pc::NumSpec ? rtol_enuc : rtol_spec;
    }

    INTEGRATORS_HOST_DEVICE static constexpr integrators::Real ros2s_atol(integrators::size_type i) {
        return i == pc::NumSpec ? atol_enuc : atol_spec;
    }
};

using Ros2sProblem = std::conditional_t<ros2s_static_tolerances,
                                        PrimordialChemRos2sStaticTolerances,
                                        pc::PrimordialChem>;

struct CollapseState {
    pc::burn_t initial_state{};
    pc::burn_t state{};
    integrators::Real time{};
    integrators::Real density_driver{};
    int completed_steps{};
    IntegratorWorkStats integrator_work{};
    NonpositivityStats nonpositivity{};
};

struct CollapseStepResult {
    bool stop{};
    int failed_step{-1};
    integrators::IntegratorResult result{integrators::IntegratorResult::SUCCESS};
};

struct Diagnostics {
    bool trace_pre_burn{};
    bool trace_vode{};
    bool probe_deuterium_terms{};
    bool dump_history{};
    bool dump_final_state{};
    bool integrator_stats{};
    bool positivity_report{};
    bool reject_negative_substeps{};
    int trace_cell_id{-1};
    int trace_step{-1};
};

constexpr int deuterium_probe_term_count = 32;

struct DeuteriumProbeTerms {
    std::array<integrators::Real, deuterium_probe_term_count> value{};
};

constexpr std::array<std::string_view, deuterium_probe_term_count> deuterium_probe_term_names{
    "x0_T^-0.75",
    "x1_Dp_recomb",
    "x2_D_ion",
    "x20_Hm_Dm_rate",
    "x39_H_Dm",
    "x47_log10_rate",
    "x51_Hp_HD",
    "x55_Dp_H2_sink_rate",
    "x65_H_HDp",
    "x67_T^-0.5",
    "x69_Hp_Dm",
    "x74_Hp_D",
    "x76_H_Dp",
    "x77_x74_minus_x76",
    "x97_H_Dp_log10",
    "x101_D_H2_to_HD_H",
    "x102_HD_H_to_D_H2",
    "x103_x101_minus_x102",
    "x104_Hm_exchange_rate",
    "x105_Hm_D",
    "x106_H_Dm",
    "x107_x105_minus_x106",
    "x111_Dp_Dm",
    "x112_neg_x51",
    "x113_neg_x55",
    "x114_HDp_source",
    "x115_HD_source",
    "x116x55_Dp_H2",
    "ydot_Dp",
    "ydot_D",
    "ydot_Dm",
    "ydot_HD"};

PRIMORDIAL_CHEM_PROBE_HOST_DEVICE DeuteriumProbeTerms compute_deuterium_probe_terms(const pc::burn_t& state) {
    using integrators::Real;

    pc::Array1D<Real, 0, pc::NumSpec - 1> X;
    for (int i = 0; i < pc::NumSpec; ++i) {
        X(i) = state.xn[static_cast<std::size_t>(i)];
    }

    const Real T = state.T;
    const Real x0 = std::exp((-0.75) * std::log(std::abs(T)));
    const Real x1 = 2.5950363272655348e-10 * X(0) * X(4) * x0;
    const Real x2 = 1.3300135414628029e-18 *
                    std::exp((0.94999999999999996) * std::log(std::abs(T))) *
                    X(0) * X(5) * std::exp(-0.00010729613733905579 * T);
    const Real x3 = T * T;
    const Real x6 = std::log(T);
    const Real x28 = std::sqrt(T);
    const Real x29 = 1.0 / x28;
    const Real x34 = 1.0 / T;
    const Real x38 = X(2) * X(7);
    const Real x44 = pc::ln10;
    const Real x45 = 1.0 / x44;
    const Real x46 = x45 * x6;
    const Real x47 = std::exp((-0.12690000000000001 *
                                    std::exp((-3.0) * std::log(std::abs(x44))) *
                                    (x6 * x6 * x6) +
                                1.1180000000000001 *
                                    std::exp((-2.0) * std::log(std::abs(x44))) *
                                    (x6 * x6) -
                                1.5229999999999999 * x46 - 19.379999999999999) *
                               std::log(std::abs(10.0)));
    const Real x48 = X(1) * X(5);
    const Real x51 = 1.0000000000000001e-9 * X(1) * X(10) * std::exp(-457.0 * x34);
    const Real x52 = x6 * x6;
    const Real x53 = std::exp((-2.0) * std::log(std::abs(x44)));
    const Real x54 = x52 * x53;
    const Real x55 = 8.4600000000000008e-10 * x46 -
                     1.3700000000000002e-10 * x54 + 4.1700000000000001e-10;
    const Real x56 = x6 * x6 * x6;
    const Real x57 = x56 / (x44 * x44 * x44);
    const Real x60 = (x6 * x6) * (x6 * x6);
    const Real x61 = x6 * x6 * x6 * x6 * x6;
    const Real x20 =
        2.6534040307116387e-9 * std::exp((-0.10000000000000001) * std::log(std::abs(T)));
    const Real x21 = X(3) * X(5);
    const Real x22 = x20 * x21;
    const Real x37 = 7.1999999999999996e-8 * X(9) * X(0) * x29;
    const Real x39 = x20 * x38;
    const Real x65 = 6.3999999999999996e-10 * X(2) * X(9);
    const Real x67 = std::exp((-0.5) * std::log(std::abs(T)));
    const Real x68 = X(7) * x67;
    const Real x69 = 7.9674337148168363e-7 * X(1) * x68;
    const bool x73 = T >= 50.0;
    const Real x74 = x48 *
                     (x73 ? (2.0000000000000001e-10 *
                                 std::exp((0.40200000000000002) * std::log(std::abs(T))) *
                                 std::exp(-37.100000000000001 * x34) -
                             3.3099999999999998e-17 *
                                 std::exp((1.48) * std::log(std::abs(T))))
                          : 0.0);
    const Real x75 = X(2) * X(4);
    const Real x76 = x75 *
                     (x73 ? (2.0299999999999998e-9 *
                                 std::exp((-0.33200000000000002) * std::log(std::abs(T))) +
                             2.0600000000000001e-10 *
                                 std::exp((0.39600000000000002) * std::log(std::abs(T))) *
                                 std::exp(-33.0 * x34))
                          : 0.0);
    const Real x77 = x74 - x76;
    const Real x97 = x47 * x75;
    const Real x101 =
        X(5) * X(8) *
        ((T <= 1167.4796423742259)
             ? std::exp((5.8888600000000002 * x46 + 7.1969200000000004 * x54 +
                         2.2506900000000001 * x57 - 56.473700000000001 -
                         2.1690299999999998 * x60 / ((x44 * x44) * (x44 * x44)) +
                         0.31788699999999998 * x61 / (x44 * x44 * x44 * x44 * x44)) *
                        std::log(std::abs(10.0)))
             : 3.1699999999999999e-10 * std::exp(-5207.0 * x34));
    const Real x102 =
        X(10) * X(2) *
        ((T > 200.0) ? 5.25e-11 * std::exp(-4430.0 * x34 + 173900.0 / x3) : 0.0);
    const Real x103 = x101 - x102;
    const Real x104 =
        6.1739095063118665e-10 * std::exp((0.40999999999999998) * std::log(std::abs(T)));
    const Real x105 = x104 * x21;
    const Real x106 = x104 * x38;
    const Real x107 = x105 - x106;
    const Real x111 = 9.8726896031426014e-7 * X(4) * x68;
    const Real x112 = -x51;
    const Real x113 = -x55;
    const Real x114 = -x37 + x47 * x48;
    const Real x115 = x103 + x22 + 1.0e-25 * X(2) * X(5);
    const Real x116 = X(4) * X(8);
    const Real x116x55 = x116 * x55;

    DeuteriumProbeTerms terms{};
    terms.value = {x0,
                   x1,
                   x2,
                   x20,
                   x39,
                   x47,
                   x51,
                   x55,
                   x65,
                   x67,
                   x69,
                   x74,
                   x76,
                   x77,
                   x97,
                   x101,
                   x102,
                   x103,
                   x104,
                   x105,
                   x106,
                   x107,
                   x111,
                   x112,
                   x113,
                   x114,
                   x115,
                   x116x55,
                   x116 * x113 - x1 - x111 - x112 + x74 - x76 - x97,
                   1.9745379206285203e-6 * X(4) * X(7) * x67 + x1 - x107 -
                       x114 - x115 - x2 + x69 - x77,
                   x105 - x106 - x111 + x2 - x39 - x69,
                   x112 + x115 + x116x55 + x39 + x65};
    return terms;
}

pc::burn_t cpu_step_368_probe_state() {
    pc::burn_t state{};
    state.rho = 8.0225621235527647e-09;
    state.T = 2073.7308848817029;
    state.e = 181614478772.05142;
    state.xn = {7360.6148584126731,
                7359.7221359245577,
                208450877171531.84,
                0.0098553897639302168,
                5.1648268819482538e-14,
                0.0008831534396278695,
                0.90257787787854005,
                4.1454094118597464e-20,
                1725642829363144.5,
                2.2352532353821725e-21,
                0.029147185068568383,
                2.0901377622181475e-61,
                4.191666669164281e-20,
                283600654265914.0};
    return state;
}

pc::burn_t gpu_step_368_probe_state() {
    pc::burn_t state{};
    state.rho = 8.0225621235527697e-09;
    state.T = 2073.7307514362606;
    state.e = 181614464702.20938;
    state.xn = {7361.4944869083301,
                7360.6016574652031,
                208450612022688.44,
                0.0098565661361487231,
                3.7570491836910345e-14,
                0.00064235454350635552,
                0.90268600926302534,
                3.0154941040398443e-20,
                1725642961937542.2,
                1.6259892831234718e-21,
                0.021199997186172825,
                2.0952356627443784e-61,
                4.1921692967711609e-20,
                283600654265926.75};
    return state;
}

struct BatchStatus {
    int completed_global_steps{};
    int failed_cell{-1};
    CollapseStepResult failed_step{};
};

constexpr std::string_view integrator_name(IntegratorChoice choice) {
    switch (choice) {
    case IntegratorChoice::VODE:
        return "VODE";
    case IntegratorChoice::ROS2S:
        return "ROS2S";
    case IntegratorChoice::ROSENBROCK_SANDU_A:
        return "RosenbrockSanduA";
    case IntegratorChoice::ROSENBROCK_SANDU_B:
        return "RosenbrockSanduB";
    case IntegratorChoice::ROSENBROCK_SANDU_C:
        return "RosenbrockSanduC";
    case IntegratorChoice::ROSENBROCK_SANDU_D:
    default:
        return "RosenbrockSanduD";
    }
}

INTEGRATORS_HOST_DEVICE void configure_microphysics_tolerances(integrators::VODEState<pc::neqs>& state) {
    state.use_vector_tolerances = true;
    for (int n = 0; n < pc::NumSpec; ++n) {
        state.rtol_vec[static_cast<std::size_t>(n)] = active_rtol_spec();
        state.atol_vec[static_cast<std::size_t>(n)] = active_atol_spec();
    }
    state.rtol_vec[pc::NumSpec] = active_rtol_enuc();
    state.atol_vec[pc::NumSpec] = active_atol_enuc();
    state.rtol = active_rtol_spec();
    state.atol = active_atol_spec();
    state.max_steps = 10000000;
    state.HMXI = 1.0e-30;
    state.constrained_components = pc::NumSpec;
    state.reject_change_buffer = 1.0e100;
    state.species_failure_tolerance = 1.0e-2;
    state.clean_constrained_components = true;
    state.component_floor = pc::small_number_density_floor();
}

template<bool AnalyticJacobianOnly, bool ExternalMatrixStorage, bool IncludeRhsScratch,
         bool StaticTolerances, bool CollectStats>
INTEGRATORS_HOST_DEVICE void configure_microphysics_tolerances(
    integrators::RODASState<pc::neqs, AnalyticJacobianOnly, ExternalMatrixStorage,
                            IncludeRhsScratch, StaticTolerances, CollectStats>& state) {
    if constexpr (!StaticTolerances) {
        state.use_vector_tolerances = true;
        for (int n = 0; n < pc::NumSpec; ++n) {
            state.rtol_vec[static_cast<std::size_t>(n)] = active_rtol_spec();
            state.atol_vec[static_cast<std::size_t>(n)] = active_atol_spec();
        }
        state.rtol_vec[pc::NumSpec] = active_rtol_enuc();
        state.atol_vec[pc::NumSpec] = active_atol_enuc();
        state.rtol = active_rtol_spec();
        state.atol = active_atol_spec();
    }
    state.max_steps = 10000000;
}

INTEGRATORS_HOST_DEVICE integrators::IntegratorResult burn_once_vode(pc::burn_t& state, integrators::Real dt,
                                                                      bool finite_difference_jacobian,
                                                                      bool trace_vode,
                                                                      IntegratorWorkStats* stats) {
    pc::eos_rt(state);

    auto integrator = integrators::VODE<pc::PrimordialChem>{};
    auto vode_state = integrators::VODEState<pc::neqs>{};
    configure_microphysics_tolerances(vode_state);

    vode_state.t = 0.0;
    vode_state.tout = dt;
    vode_state.jacobian_analytic = !finite_difference_jacobian;
    vode_state.trace_deuterium_components = trace_vode;
    for (int n = 0; n < pc::NumSpec; ++n) {
        vode_state.y[static_cast<std::size_t>(n)] = state.xn[static_cast<std::size_t>(n)];
    }
    vode_state.y[pc::NumSpec] = state.e;

    pc::PrimordialChem::state_type problem_state{};
    const auto result = integrator.integrate(problem_state, vode_state);
    if (stats != nullptr) {
        stats->internal_steps += static_cast<std::uint64_t>(std::max(0, vode_state.n_step));
        stats->rhs_calls += static_cast<std::uint64_t>(std::max(0, vode_state.n_rhs));
        stats->jacobian_calls += static_cast<std::uint64_t>(std::max(0, vode_state.n_jac));
        stats->decompositions += static_cast<std::uint64_t>(std::max(0, vode_state.n_decomp));
        stats->linear_solves += static_cast<std::uint64_t>(std::max(0, vode_state.n_solve));
        stats->error_failures += static_cast<std::uint64_t>(std::max(0, vode_state.n_error_fails));
    }

    if (result == integrators::IntegratorResult::SUCCESS) {
        for (int n = 0; n < pc::NumSpec; ++n) {
            state.xn[static_cast<std::size_t>(n)] = vode_state.y[static_cast<std::size_t>(n)];
        }
        state.e = vode_state.y[pc::NumSpec];
        state.success = true;
    } else {
#if defined(__CUDA_ARCH__)
        printf("VODE internal failure: result=%d n_step=%d n_rhs=%d n_jac=%d err_fails=%d H=%.17e tn=%.17e NQ=%d L=%d acnrm=%.17e\n",
               static_cast<int>(result), vode_state.n_step, vode_state.n_rhs,
               vode_state.n_jac, vode_state.err_fails, vode_state.H, vode_state.tn,
               static_cast<int>(vode_state.NQ), static_cast<int>(vode_state.L),
               vode_state.acnrm_last);
#else
        std::cout << "VODE internal failure: result=" << static_cast<int>(result)
                  << " n_step=" << vode_state.n_step
                  << " n_rhs=" << vode_state.n_rhs
                  << " n_jac=" << vode_state.n_jac
                  << " err_fails=" << vode_state.err_fails
                  << " H=" << vode_state.H
                  << " tn=" << vode_state.tn
                  << " NQ=" << static_cast<int>(vode_state.NQ)
                  << " L=" << static_cast<int>(vode_state.L)
                  << " acnrm=" << vode_state.acnrm_last << "\n";
#endif
        state.success = false;
    }

    return result;
}

template<IntegratorChoice Choice, bool FiniteDifferenceJacobian, bool CollectStats>
INTEGRATORS_HOST_DEVICE integrators::IntegratorResult burn_once_ros2s(pc::burn_t& state, integrators::Real dt,
                                                                       IntegratorWorkStats* stats,
                                                                       RODASMatrix* external_matrix,
                                                                       bool reject_negative_substeps) {
    pc::eos_rt(state);

    constexpr bool analytic_jacobian_only =
        !FiniteDifferenceJacobian && compact_ros2s_analytic_jacobian;
    constexpr bool external_matrix_storage =
        !FiniteDifferenceJacobian && ros2s_external_matrix;
    using Integrator =
        std::conditional_t<Choice == IntegratorChoice::ROS2S,
                           integrators::RODAS<Ros2sProblem, analytic_jacobian_only, CollectStats,
                                              ros2s_allow_pivoting, ros2s_gift_factorization,
                                              external_matrix_storage, ros2s_compact_rhs_scratch,
                                              ros2s_static_tolerances>,
        std::conditional_t<Choice == IntegratorChoice::ROSENBROCK_SANDU_A,
                           integrators::RosenbrockSanduA<
                               Ros2sProblem, analytic_jacobian_only, CollectStats,
                               ros2s_allow_pivoting, ros2s_gift_factorization,
                               external_matrix_storage, ros2s_compact_rhs_scratch,
                               ros2s_static_tolerances>,
        std::conditional_t<Choice == IntegratorChoice::ROSENBROCK_SANDU_B,
                           integrators::RosenbrockSanduB<
                               Ros2sProblem, analytic_jacobian_only, CollectStats,
                               ros2s_allow_pivoting, ros2s_gift_factorization,
                               external_matrix_storage, ros2s_compact_rhs_scratch,
                               ros2s_static_tolerances>,
        std::conditional_t<Choice == IntegratorChoice::ROSENBROCK_SANDU_C,
                           integrators::RosenbrockSanduC<
                               Ros2sProblem, analytic_jacobian_only, CollectStats,
                               ros2s_allow_pivoting, ros2s_gift_factorization,
                               external_matrix_storage, ros2s_compact_rhs_scratch,
                               ros2s_static_tolerances>,
                           integrators::RosenbrockSanduD<
                               Ros2sProblem, analytic_jacobian_only, CollectStats,
                               ros2s_allow_pivoting, ros2s_gift_factorization,
                               external_matrix_storage, ros2s_compact_rhs_scratch,
                               ros2s_static_tolerances>>>>>;
    auto integrator = Integrator{};
    auto ros2s_state =
        typename decltype(integrator)::State{};
    if constexpr (external_matrix_storage) {
        ros2s_state.external_e = external_matrix;
    } else {
        (void)external_matrix;
    }
    configure_microphysics_tolerances(ros2s_state);

    ros2s_state.t = 0.0;
    ros2s_state.tout = dt;
    ros2s_state.dt = dt;
    ros2s_state.jacobian_analytic = !FiniteDifferenceJacobian;
    ros2s_state.autonomous = true;
    ros2s_state.reject_negative_states = reject_negative_substeps;
    for (int n = 0; n < pc::NumSpec; ++n) {
        ros2s_state.y[static_cast<std::size_t>(n)] = state.xn[static_cast<std::size_t>(n)];
    }
    ros2s_state.y[pc::NumSpec] = state.e;

    typename Ros2sProblem::state_type problem_state{};
    const auto result = integrator.integrate(problem_state, ros2s_state);
    if constexpr (CollectStats) {
        if (stats != nullptr) {
            stats->internal_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_step));
            stats->rhs_calls += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_rhs));
            stats->jacobian_calls += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_jac));
            stats->decompositions += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_decomp));
            stats->linear_solves += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_solve));
            stats->accepted_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_accept));
            stats->rejected_steps += static_cast<std::uint64_t>(std::max(0, ros2s_state.n_reject));
            stats->negative_rejected_steps +=
                static_cast<std::uint64_t>(std::max(0, ros2s_state.n_negative_reject));
        }
    }

    if (result == integrators::IntegratorResult::SUCCESS) {
        for (int n = 0; n < pc::NumSpec; ++n) {
            state.xn[static_cast<std::size_t>(n)] = ros2s_state.y[static_cast<std::size_t>(n)];
        }
        state.e = ros2s_state.y[pc::NumSpec];
        state.success = true;
    } else {
        state.success = false;
    }

    return result;
}

INTEGRATORS_HOST_DEVICE integrators::IntegratorResult burn_once_ros2s(pc::burn_t& state, integrators::Real dt,
                                                                       bool finite_difference_jacobian,
                                                                       IntegratorChoice integrator_choice,
                                                                       IntegratorWorkStats* stats,
                                                                       RODASMatrix* external_matrix,
                                                                       bool reject_negative_substeps) {
    switch (integrator_choice) {
    case IntegratorChoice::ROS2S:
        if (finite_difference_jacobian) {
            if (stats != nullptr) {
                return burn_once_ros2s<IntegratorChoice::ROS2S, true, true>(
                    state, dt, stats, external_matrix, reject_negative_substeps);
            }
            return burn_once_ros2s<IntegratorChoice::ROS2S, true, false>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        if (stats != nullptr) {
            return burn_once_ros2s<IntegratorChoice::ROS2S, false, true>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        return burn_once_ros2s<IntegratorChoice::ROS2S, false, false>(
            state, dt, stats, external_matrix, reject_negative_substeps);
    case IntegratorChoice::ROSENBROCK_SANDU_A:
        if (finite_difference_jacobian) {
            if (stats != nullptr) {
                return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_A, true, true>(
                    state, dt, stats, external_matrix, reject_negative_substeps);
            }
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_A, true, false>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        if (stats != nullptr) {
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_A, false, true>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_A, false, false>(
            state, dt, stats, external_matrix, reject_negative_substeps);
    case IntegratorChoice::ROSENBROCK_SANDU_B:
        if (finite_difference_jacobian) {
            if (stats != nullptr) {
                return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_B, true, true>(
                    state, dt, stats, external_matrix, reject_negative_substeps);
            }
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_B, true, false>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        if (stats != nullptr) {
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_B, false, true>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_B, false, false>(
            state, dt, stats, external_matrix, reject_negative_substeps);
    case IntegratorChoice::ROSENBROCK_SANDU_C:
        if (finite_difference_jacobian) {
            if (stats != nullptr) {
                return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_C, true, true>(
                    state, dt, stats, external_matrix, reject_negative_substeps);
            }
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_C, true, false>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        if (stats != nullptr) {
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_C, false, true>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_C, false, false>(
            state, dt, stats, external_matrix, reject_negative_substeps);
    case IntegratorChoice::ROSENBROCK_SANDU_D:
    default:
        if (finite_difference_jacobian) {
            if (stats != nullptr) {
                return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_D, true, true>(
                    state, dt, stats, external_matrix, reject_negative_substeps);
            }
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_D, true, false>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        if (stats != nullptr) {
            return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_D, false, true>(
                state, dt, stats, external_matrix, reject_negative_substeps);
        }
        return burn_once_ros2s<IntegratorChoice::ROSENBROCK_SANDU_D, false, false>(
            state, dt, stats, external_matrix, reject_negative_substeps);
    }
}

INTEGRATORS_HOST_DEVICE integrators::IntegratorResult burn(pc::burn_t& state, integrators::Real dt,
                                                           bool finite_difference_jacobian,
                                                           bool trace_vode,
                                                           IntegratorChoice integrator_choice,
                                                           IntegratorWorkStats* stats,
                                                           RODASMatrix* external_matrix,
                                                           bool reject_negative_substeps) {
    switch (integrator_choice) {
    case IntegratorChoice::VODE:
        (void)external_matrix;
        (void)reject_negative_substeps;
        return burn_once_vode(state, dt, finite_difference_jacobian, trace_vode, stats);
    case IntegratorChoice::ROS2S:
    default:
        return burn_once_ros2s(state, dt, finite_difference_jacobian, integrator_choice, stats,
                               external_matrix, reject_negative_substeps);
    }
}

template<IntegratorChoice Choice, bool FiniteDifferenceJacobian, bool CollectStats>
INTEGRATORS_HOST_DEVICE integrators::IntegratorResult burn(pc::burn_t& state, integrators::Real dt,
                                                           bool trace_vode,
                                                           IntegratorWorkStats* stats,
                                                           RODASMatrix* external_matrix,
                                                           bool reject_negative_substeps) {
    if constexpr (Choice == IntegratorChoice::VODE) {
        (void)external_matrix;
        (void)reject_negative_substeps;
        return burn_once_vode(state, dt, FiniteDifferenceJacobian, trace_vode, stats);
    } else if constexpr (Choice == IntegratorChoice::ROS2S ||
                         Choice == IntegratorChoice::ROSENBROCK_SANDU_A ||
                         Choice == IntegratorChoice::ROSENBROCK_SANDU_B ||
                         Choice == IntegratorChoice::ROSENBROCK_SANDU_C ||
                         Choice == IntegratorChoice::ROSENBROCK_SANDU_D) {
        (void)trace_vode;
        return burn_once_ros2s<Choice, FiniteDifferenceJacobian, CollectStats>(
            state, dt, stats, external_matrix, reject_negative_substeps);
    }
}

INTEGRATORS_HOST_DEVICE pc::burn_t make_initial_state() {
    pc::burn_t state;
    state.T = temperature;
    state.xn = initial_number_densities;
    state.rho = pc::density(state.xn);
    pc::normalize_number_densities_to_density(state);
    pc::eos_rt(state);
    return state;
}

INTEGRATORS_HOST_DEVICE CollapseState make_collapse_state() {
    pc::burn_t state = make_initial_state();
    return {state, state, 0.0, state.rho, 0};
}

INTEGRATORS_HOST_DEVICE std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

INTEGRATORS_HOST_DEVICE integrators::Real perturbation_factor(int cell, int step, int component) {
    const auto seed = (static_cast<std::uint64_t>(static_cast<unsigned int>(cell)) << 32U) ^
                      (static_cast<std::uint64_t>(static_cast<unsigned int>(step)) << 16U) ^
                      static_cast<std::uint64_t>(static_cast<unsigned int>(component));
    const auto bits = splitmix64(seed) >> 11U;
    const auto unit = static_cast<integrators::Real>(bits) *
                      (1.0 / static_cast<integrators::Real>(1ULL << 53U));
    return 1.0 + perturbation_amplitude * (2.0 * unit - 1.0);
}

INTEGRATORS_HOST_DEVICE void apply_cell_perturbation(CollapseState& collapse, int step, int cell,
                                                     bool perturb_cells) {
    if (!perturb_cells || step == 0 || step % perturbation_interval != 0) {
        return;
    }

    const auto factor = perturbation_factor(cell, step, 0);
    collapse.density_driver *= factor;
    collapse.state.rho *= factor;

    for (int n = 0; n < pc::NumSpec; ++n) {
        collapse.state.xn[static_cast<std::size_t>(n)] *= factor;
    }

    pc::floor_and_normalize_number_densities(collapse.state);
    pc::balance_charge(collapse.state);
    pc::floor_and_normalize_number_densities(collapse.state);
    pc::eos_re(collapse.state);
}

PRIMORDIAL_CHEM_PROBE_HOST_DEVICE void add_integrator_work(IntegratorWorkStats& total,
                                                           const IntegratorWorkStats& step) {
    total.internal_steps += step.internal_steps;
    total.rhs_calls += step.rhs_calls;
    total.jacobian_calls += step.jacobian_calls;
    total.decompositions += step.decompositions;
    total.linear_solves += step.linear_solves;
    total.accepted_steps += step.accepted_steps;
    total.rejected_steps += step.rejected_steps;
    total.negative_rejected_steps += step.negative_rejected_steps;
    total.error_failures += step.error_failures;
}

INTEGRATORS_HOST_DEVICE void record_nonpositivity(CollapseState& collapse) {
    collapse.nonpositivity.last_step.fill(0);
    for (int n = 0; n < pc::NumSpec; ++n) {
        const auto value = collapse.state.xn[static_cast<std::size_t>(n)];
        if (!std::isfinite(value) || value <= 0.0) {
            collapse.nonpositivity.last_step[static_cast<std::size_t>(n)] += 1;
            collapse.nonpositivity.cumulative[static_cast<std::size_t>(n)] += 1;
        }
    }
    const auto energy = collapse.state.e;
    if (!std::isfinite(energy) || energy <= 0.0) {
        collapse.nonpositivity.last_step[pc::NumSpec] += 1;
        collapse.nonpositivity.cumulative[pc::NumSpec] += 1;
    }
}

INTEGRATORS_HOST_DEVICE CollapseStepResult advance_collapse_step(CollapseState& collapse, int step, int cell,
                                                                 bool perturb_cells,
                                                                 bool finite_difference_jacobian,
                                                                 Diagnostics diagnostics,
                                                                 IntegratorChoice integrator_choice,
                                                                 RODASMatrix* external_matrix) {
    apply_cell_perturbation(collapse, step, cell, perturb_cells);

    const integrators::Real dd1 = collapse.density_driver;
    const integrators::Real rhotmp = pc::density(collapse.state.xn);
    const integrators::Real tff = std::sqrt(pc::pi * 3.0 / (32.0 * rhotmp * pc::grav_constant));
    const integrators::Real dt = tff_reduc * tff;

    collapse.density_driver += dt * (collapse.density_driver / tff);

    if (dt < 10.0 || collapse.density_driver > 2.0e-6) {
        return {true, -1, integrators::IntegratorResult::SUCCESS};
    }

    const integrators::Real density_ratio = collapse.density_driver / dd1;
    for (auto& xn : collapse.state.xn) {
        xn *= density_ratio;
    }
    collapse.state.rho *= density_ratio;

    if (diagnostics.trace_pre_burn && cell == diagnostics.trace_cell_id &&
        step == diagnostics.trace_step) {
#if !defined(__CUDA_ARCH__)
        std::cout << "trace pre-burn state: cell-id " << cell << " collapse step " << step << "\n";
        std::cout << "completed steps: " << collapse.completed_steps << "\n";
        std::cout << "time: " << collapse.time << "\n";
        std::cout << "dt: " << dt << "\n";
        std::cout << "density driver: " << collapse.density_driver << "\n";
        std::cout << "rho: " << collapse.state.rho << "\n";
        std::cout << "T: " << collapse.state.T << "\n";
        std::cout << "Eint: " << collapse.state.e << "\n";
        for (int n = 0; n < pc::NumSpec; ++n) {
            std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                      << collapse.state.xn[static_cast<std::size_t>(n)] << "\n";
        }
#endif
    }

    const bool trace_vode = diagnostics.trace_vode && cell == diagnostics.trace_cell_id &&
                            step == diagnostics.trace_step;
    IntegratorWorkStats step_stats{};
    const auto result = burn(collapse.state, dt, finite_difference_jacobian,
                             trace_vode, integrator_choice,
                             diagnostics.integrator_stats ? &step_stats : nullptr,
                             external_matrix, diagnostics.reject_negative_substeps);
    if (diagnostics.integrator_stats) {
        add_integrator_work(collapse.integrator_work, step_stats);
    }
    if (result != integrators::IntegratorResult::SUCCESS) {
        return {true, step, result};
    }
    if (diagnostics.positivity_report) {
        record_nonpositivity(collapse);
    }

    pc::floor_and_normalize_number_densities(collapse.state);
    pc::balance_charge(collapse.state);
    pc::floor_and_normalize_number_densities(collapse.state);
    pc::eos_re(collapse.state);

    collapse.time += dt;
    collapse.completed_steps += 1;
    return {false, -1, integrators::IntegratorResult::SUCCESS};
}

template<IntegratorChoice Choice, bool FiniteDifferenceJacobian, bool CollectStats>
INTEGRATORS_HOST_DEVICE CollapseStepResult advance_collapse_step(CollapseState& collapse, int step, int cell,
                                                                 bool perturb_cells,
                                                                 Diagnostics diagnostics,
                                                                 RODASMatrix* external_matrix) {
    apply_cell_perturbation(collapse, step, cell, perturb_cells);

    const integrators::Real dd1 = collapse.density_driver;
    const integrators::Real rhotmp = pc::density(collapse.state.xn);
    const integrators::Real tff = std::sqrt(pc::pi * 3.0 / (32.0 * rhotmp * pc::grav_constant));
    const integrators::Real dt = tff_reduc * tff;

    collapse.density_driver += dt * (collapse.density_driver / tff);

    if (dt < 10.0 || collapse.density_driver > 2.0e-6) {
        return {true, -1, integrators::IntegratorResult::SUCCESS};
    }

    const integrators::Real density_ratio = collapse.density_driver / dd1;
    for (auto& xn : collapse.state.xn) {
        xn *= density_ratio;
    }
    collapse.state.rho *= density_ratio;

    if (diagnostics.trace_pre_burn && cell == diagnostics.trace_cell_id &&
        step == diagnostics.trace_step) {
#if !defined(__CUDA_ARCH__)
        std::cout << "trace pre-burn state: cell-id " << cell << " collapse step " << step << "\n";
        std::cout << "completed steps: " << collapse.completed_steps << "\n";
        std::cout << "time: " << collapse.time << "\n";
        std::cout << "dt: " << dt << "\n";
        std::cout << "density driver: " << collapse.density_driver << "\n";
        std::cout << "rho: " << collapse.state.rho << "\n";
        std::cout << "T: " << collapse.state.T << "\n";
        std::cout << "Eint: " << collapse.state.e << "\n";
        for (int n = 0; n < pc::NumSpec; ++n) {
            std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                      << collapse.state.xn[static_cast<std::size_t>(n)] << "\n";
        }
#endif
    }

    const bool trace_vode = diagnostics.trace_vode && cell == diagnostics.trace_cell_id &&
                            step == diagnostics.trace_step;
    IntegratorWorkStats step_stats{};
    IntegratorWorkStats* step_stats_ptr = nullptr;
    if constexpr (CollectStats) {
        if (diagnostics.integrator_stats) {
            step_stats_ptr = &step_stats;
        }
    }
    const auto result = burn<Choice, FiniteDifferenceJacobian, CollectStats>(
        collapse.state, dt, trace_vode, step_stats_ptr, external_matrix,
        diagnostics.reject_negative_substeps);
    if constexpr (CollectStats) {
        if (diagnostics.integrator_stats) {
            add_integrator_work(collapse.integrator_work, step_stats);
        }
    }
    if (result != integrators::IntegratorResult::SUCCESS) {
        return {true, step, result};
    }
    if (diagnostics.positivity_report) {
        record_nonpositivity(collapse);
    }

    pc::floor_and_normalize_number_densities(collapse.state);
    pc::balance_charge(collapse.state);
    pc::floor_and_normalize_number_densities(collapse.state);
    pc::eos_re(collapse.state);

    collapse.time += dt;
    collapse.completed_steps += 1;
    return {false, -1, integrators::IntegratorResult::SUCCESS};
}

#if defined(__CUDACC__)
__global__ void initialize_collapse_kernel(CollapseState* cells,
                                           CollapseStepResult* cell_results,
                                           int num_cells) {
    const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= num_cells) {
        return;
    }

    cells[cell] = make_collapse_state();
    cell_results[cell] = {};
}

template<IntegratorChoice Choice, bool FiniteDifferenceJacobian, bool CollectStats>
PRIMORDIAL_CHEM_COLLAPSE_LAUNCH_BOUNDS __global__
void collapse_step_kernel(CollapseState* cells,
                          CollapseStepResult* cell_results,
                          RODASMatrix* ros2s_matrices,
                          int step,
                          int num_cells,
                          int cell_id_offset,
                          bool perturb_cells,
                          Diagnostics diagnostics) {
    const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= num_cells || cell_results[cell].stop) {
        return;
    }

    RODASMatrix* external_matrix = nullptr;
    if constexpr (Choice != IntegratorChoice::VODE && !FiniteDifferenceJacobian &&
                  ros2s_external_matrix) {
        external_matrix = &ros2s_matrices[cell];
    } else {
        (void)ros2s_matrices;
    }

    cell_results[cell] = advance_collapse_step<Choice, FiniteDifferenceJacobian, CollectStats>(
        cells[cell], step, cell + cell_id_offset, perturb_cells, diagnostics, external_matrix);
}

__global__ void deuterium_probe_kernel(const pc::burn_t* states,
                                       DeuteriumProbeTerms* terms,
                                       int num_states) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= num_states) {
        return;
    }
    terms[i] = compute_deuterium_probe_terms(states[i]);
}

#endif

bool nearly_equal(integrators::Real value, integrators::Real reference,
                  integrators::Real rtol, integrators::Real atol) {
    return std::abs(value - reference) <= atol + rtol * std::abs(reference);
}

bool parse_positive_int(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_nonnegative_int(const char* text, int& value) {
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed < 0 || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

bool parse_positive_real(const char* text, integrators::Real& value) {
    errno = 0;
    char* end = nullptr;
    const auto parsed = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || parsed <= 0.0 ||
        !std::isfinite(parsed)) {
        return false;
    }
    value = static_cast<integrators::Real>(parsed);
    return true;
}

void print_usage(const char* program) {
    std::cerr << "usage: " << program
              << " [--grid N] [--integrator vode|ros2s|sandu-a|sandu-b|sandu-c|sandu-d] [--perturb] [--fd-jacobian]"
              << " [--cell-id N] [--trace-vode] [--trace-step N] [--trace-cell-id N]"
              << " [--probe-deuterium-terms] [--dump-history] [--dump-final-state] [--integrator-stats]"
              << " [--positivity-report] [--reject-negative-substeps]"
              << " [--threads-per-block N] [--rtol X] [--atol X] [--energy-atol X]\n";
}

bool parse_integrator(std::string_view text, IntegratorChoice& integrator_choice) {
    if (text == "vode") {
        integrator_choice = IntegratorChoice::VODE;
        return true;
    }
    if (text == "ros2s" || text == "rodas") {
        integrator_choice = IntegratorChoice::ROS2S;
        return true;
    }
    if (text == "sandu-a" || text == "rosenbrock-sandu-a") {
        integrator_choice = IntegratorChoice::ROSENBROCK_SANDU_A;
        return true;
    }
    if (text == "sandu-b" || text == "rosenbrock-sandu-b") {
        integrator_choice = IntegratorChoice::ROSENBROCK_SANDU_B;
        return true;
    }
    if (text == "sandu-c" || text == "rosenbrock-sandu-c") {
        integrator_choice = IntegratorChoice::ROSENBROCK_SANDU_C;
        return true;
    }
    if (text == "sandu-d" || text == "rosenbrock-sandu-d") {
        integrator_choice = IntegratorChoice::ROSENBROCK_SANDU_D;
        return true;
    }
    return false;
}

bool parse_args(int argc, char** argv, int& grid_dim, IntegratorChoice& integrator_choice,
                bool& perturb_cells, bool& finite_difference_jacobian, int& cell_id_offset,
                Diagnostics& diagnostics, int& threads_per_block) {
    grid_dim = default_grid_dim;
    integrator_choice = IntegratorChoice::ROS2S;
    perturb_cells = false;
    finite_difference_jacobian = false;
    cell_id_offset = 0;
    diagnostics = {};
    threads_per_block = default_cuda_threads_per_block;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return false;
        }
        if (arg == "--grid") {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], grid_dim)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        if (arg == "--integrator") {
            if (i + 1 >= argc || !parse_integrator(argv[++i], integrator_choice)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        if (arg == "--perturb") {
            perturb_cells = true;
            continue;
        }
        if (arg == "--fd-jacobian") {
            finite_difference_jacobian = true;
            continue;
        }
        if (arg == "--cell-id") {
            if (i + 1 >= argc || !parse_nonnegative_int(argv[++i], cell_id_offset)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        if (arg == "--trace-pre-burn") {
            diagnostics.trace_pre_burn = true;
            continue;
        }
        if (arg == "--trace-vode") {
            diagnostics.trace_vode = true;
            continue;
        }
        if (arg == "--probe-deuterium-terms") {
            diagnostics.probe_deuterium_terms = true;
            continue;
        }
        if (arg == "--dump-history") {
            diagnostics.dump_history = true;
            continue;
        }
        if (arg == "--dump-final-state") {
            diagnostics.dump_final_state = true;
            continue;
        }
        if (arg == "--integrator-stats") {
            diagnostics.integrator_stats = true;
            continue;
        }
        if (arg == "--positivity-report") {
            diagnostics.positivity_report = true;
            continue;
        }
        if (arg == "--reject-negative-substeps") {
            diagnostics.reject_negative_substeps = true;
            continue;
        }
        if (arg == "--threads-per-block") {
            if (i + 1 >= argc || !parse_positive_int(argv[++i], threads_per_block)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        if (arg == "--rtol") {
            integrators::Real parsed = 0.0;
            if (i + 1 >= argc || !parse_positive_real(argv[++i], parsed)) {
                print_usage(argv[0]);
                return false;
            }
            runtime_tolerances.rtol_spec_value = parsed;
            runtime_tolerances.rtol_enuc_value = parsed * (rtol_enuc / rtol_spec);
            continue;
        }
        if (arg == "--atol") {
            integrators::Real parsed = 0.0;
            if (i + 1 >= argc || !parse_positive_real(argv[++i], parsed)) {
                print_usage(argv[0]);
                return false;
            }
            runtime_tolerances.atol_spec_value = parsed;
            runtime_tolerances.atol_enuc_value = parsed;
            continue;
        }
        if (arg == "--energy-atol") {
            integrators::Real parsed = 0.0;
            if (i + 1 >= argc || !parse_positive_real(argv[++i], parsed)) {
                print_usage(argv[0]);
                return false;
            }
            runtime_tolerances.atol_enuc_value = parsed;
            continue;
        }
        if (arg == "--trace-step") {
            if (i + 1 >= argc || !parse_nonnegative_int(argv[++i], diagnostics.trace_step)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        if (arg == "--trace-cell-id") {
            if (i + 1 >= argc || !parse_nonnegative_int(argv[++i], diagnostics.trace_cell_id)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        constexpr std::string_view cell_id_prefix = "--cell-id=";
        if (arg.rfind(cell_id_prefix, 0) == 0) {
            if (!parse_nonnegative_int(argv[i] + cell_id_prefix.size(), cell_id_offset)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        constexpr std::string_view integrator_prefix = "--integrator=";
        if (arg.rfind(integrator_prefix, 0) == 0) {
            if (!parse_integrator(arg.substr(integrator_prefix.size()), integrator_choice)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        constexpr std::string_view grid_prefix = "--grid=";
        if (arg.rfind(grid_prefix, 0) == 0) {
            if (!parse_positive_int(argv[i] + grid_prefix.size(), grid_dim)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        constexpr std::string_view threads_prefix = "--threads-per-block=";
        if (arg.rfind(threads_prefix, 0) == 0) {
            if (!parse_positive_int(argv[i] + threads_prefix.size(), threads_per_block)) {
                print_usage(argv[0]);
                return false;
            }
            continue;
        }
        constexpr std::string_view rtol_prefix = "--rtol=";
        if (arg.rfind(rtol_prefix, 0) == 0) {
            integrators::Real parsed = 0.0;
            if (!parse_positive_real(argv[i] + rtol_prefix.size(), parsed)) {
                print_usage(argv[0]);
                return false;
            }
            runtime_tolerances.rtol_spec_value = parsed;
            runtime_tolerances.rtol_enuc_value = parsed * (rtol_enuc / rtol_spec);
            continue;
        }
        constexpr std::string_view atol_prefix = "--atol=";
        if (arg.rfind(atol_prefix, 0) == 0) {
            integrators::Real parsed = 0.0;
            if (!parse_positive_real(argv[i] + atol_prefix.size(), parsed)) {
                print_usage(argv[0]);
                return false;
            }
            runtime_tolerances.atol_spec_value = parsed;
            runtime_tolerances.atol_enuc_value = parsed;
            continue;
        }
        constexpr std::string_view energy_atol_prefix = "--energy-atol=";
        if (arg.rfind(energy_atol_prefix, 0) == 0) {
            integrators::Real parsed = 0.0;
            if (!parse_positive_real(argv[i] + energy_atol_prefix.size(), parsed)) {
                print_usage(argv[0]);
                return false;
            }
            runtime_tolerances.atol_enuc_value = parsed;
            continue;
        }

        print_usage(argv[0]);
        return false;
    }
    return true;
}

void print_probe_comparison(std::string_view state_name,
                            const DeuteriumProbeTerms& cpu,
                            const DeuteriumProbeTerms* gpu) {
    std::cout << "\nDeuterium RHS probe: " << state_name << "\n";
    std::cout << std::setw(24) << "term"
              << std::setw(24) << "cpu";
    if (gpu != nullptr) {
        std::cout << std::setw(24) << "gpu"
                  << std::setw(24) << "abs diff"
                  << std::setw(18) << "rel diff";
    }
    std::cout << "\n";

    for (int i = 0; i < deuterium_probe_term_count; ++i) {
        const auto cpu_value = cpu.value[static_cast<std::size_t>(i)];
        std::cout << std::setw(24) << deuterium_probe_term_names[static_cast<std::size_t>(i)]
                  << std::setw(24) << cpu_value;
        if (gpu != nullptr) {
            const auto gpu_value = gpu->value[static_cast<std::size_t>(i)];
            const auto abs_diff = std::abs(gpu_value - cpu_value);
            const auto denom = std::max(std::abs(cpu_value), std::numeric_limits<integrators::Real>::min());
            std::cout << std::setw(24) << gpu_value
                      << std::setw(24) << abs_diff
                      << std::setw(18) << abs_diff / denom;
        }
        std::cout << "\n";
    }
}

int run_deuterium_probe() {
    std::array<pc::burn_t, 2> states{cpu_step_368_probe_state(), gpu_step_368_probe_state()};
    std::array<DeuteriumProbeTerms, 2> cpu_terms{
        compute_deuterium_probe_terms(states[0]),
        compute_deuterium_probe_terms(states[1])};

#if defined(__CUDACC__)
    pc::burn_t* device_states = nullptr;
    DeuteriumProbeTerms* device_terms = nullptr;
    std::array<DeuteriumProbeTerms, 2> gpu_terms{};

    auto err = cudaMalloc(&device_states, sizeof(pc::burn_t) * states.size());
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc probe states failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    err = cudaMalloc(&device_terms, sizeof(DeuteriumProbeTerms) * gpu_terms.size());
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc probe terms failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_states);
        return 1;
    }
    err = cudaMemcpy(device_states, states.data(), sizeof(pc::burn_t) * states.size(),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy probe states failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_terms);
        cudaFree(device_states);
        return 1;
    }
    deuterium_probe_kernel<<<1, 32>>>(device_states, device_terms, static_cast<int>(states.size()));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "deuterium_probe_kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_terms);
        cudaFree(device_states);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "deuterium_probe_kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_terms);
        cudaFree(device_states);
        return 1;
    }
    err = cudaMemcpy(gpu_terms.data(), device_terms, sizeof(DeuteriumProbeTerms) * gpu_terms.size(),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy probe terms failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_terms);
        cudaFree(device_states);
        return 1;
    }
    cudaFree(device_terms);
    cudaFree(device_states);

    print_probe_comparison("CPU pre-burn state at cell-id 1622 step 368", cpu_terms[0], &gpu_terms[0]);
    print_probe_comparison("GPU failed state at cell-id 1622 step 368", cpu_terms[1], &gpu_terms[1]);
#else
    print_probe_comparison("CPU pre-burn state at cell-id 1622 step 368", cpu_terms[0], nullptr);
    print_probe_comparison("GPU failed state at cell-id 1622 step 368", cpu_terms[1], nullptr);
#endif
    return 0;
}

void print_history_header() {
    std::cout << "history_step,stop,result,completed,time,density_driver,rho,T,Eint";
    for (int n = 0; n < pc::NumSpec; ++n) {
        std::cout << "," << pc::short_spec_names[static_cast<std::size_t>(n)];
    }
    std::cout << "\n";
}

void print_history_row(int step, const CollapseState& cell, const CollapseStepResult& result) {
    std::cout << step
              << "," << result.stop
              << "," << static_cast<int>(result.result)
              << "," << cell.completed_steps
              << "," << cell.time
              << "," << cell.density_driver
              << "," << cell.state.rho
              << "," << cell.state.T
              << "," << cell.state.e;
    for (int n = 0; n < pc::NumSpec; ++n) {
        std::cout << "," << cell.state.xn[static_cast<std::size_t>(n)];
    }
    std::cout << "\n";
}

std::string_view positivity_component_name(int component) {
    if (component == pc::NumSpec) {
        return "Eint";
    }
    return pc::short_spec_names[static_cast<std::size_t>(component)];
}

void print_positivity_header() {
    std::cout << "positivity_csv_begin\n";
    std::cout << "method,step,component,nonpositive_count,cumulative_count\n";
}

void print_positivity_rows(IntegratorChoice integrator_choice, int step, const CollapseState& cell) {
    for (int component = 0; component < pc::neqs; ++component) {
        const auto count = cell.nonpositivity.last_step[static_cast<std::size_t>(component)];
        const auto cumulative = cell.nonpositivity.cumulative[static_cast<std::size_t>(component)];
        if (count == 0 && cumulative == 0) {
            continue;
        }
        std::cout << integrator_name(integrator_choice) << ","
                  << step << ","
                  << positivity_component_name(component) << ","
                  << count << ","
                  << cumulative << "\n";
    }
}

void print_positivity_footer() {
    std::cout << "positivity_csv_end\n";
}

bool checked_cell_count(int grid_dim, int& num_cells) {
    const auto grid = static_cast<long long>(grid_dim);
    const auto cells = grid * grid * grid;
    if (cells <= 0 || cells > std::numeric_limits<int>::max()) {
        return false;
    }
    num_cells = static_cast<int>(cells);
    return true;
}

BatchStatus batch_status_from_results(const std::vector<CollapseStepResult>& cell_results,
                                      int completed_global_steps) {
    bool all_stopped = true;
    for (std::size_t cell = 0; cell < cell_results.size(); ++cell) {
        const auto& result = cell_results[cell];
        if (result.result != integrators::IntegratorResult::SUCCESS) {
            return {completed_global_steps, static_cast<int>(cell), result};
        }
        all_stopped = all_stopped && result.stop;
    }

    if (all_stopped) {
        return {completed_global_steps, -1, {true, -1, integrators::IntegratorResult::SUCCESS}};
    }
    return {completed_global_steps, -1, {false, -1, integrators::IntegratorResult::SUCCESS}};
}

} // namespace

int main(int argc, char** argv) {
    std::cout << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10);
    std::cout << "Primordial Chemistry One-Zone Collapse\n";
    std::cout << "======================================\n\n";

    int grid_dim = default_grid_dim;
    IntegratorChoice integrator_choice = IntegratorChoice::ROS2S;
    bool perturb_cells = false;
    bool finite_difference_jacobian = false;
    int cell_id_offset = 0;
    int threads_per_block = default_cuda_threads_per_block;
    Diagnostics diagnostics{};
    if (!parse_args(argc, argv, grid_dim, integrator_choice, perturb_cells,
                    finite_difference_jacobian, cell_id_offset, diagnostics,
                    threads_per_block)) {
        return 1;
    }
    if (diagnostics.probe_deuterium_terms) {
        return run_deuterium_probe();
    }
#if defined(__CUDACC__)
    if constexpr (active_kernels_only) {
        if (finite_difference_jacobian) {
            std::cerr << "This build only includes analytic CUDA kernels; --fd-jacobian is unavailable.\n";
            return 1;
        }
        if (diagnostics.integrator_stats) {
            std::cerr << "This build only includes the no-stats ROS2S CUDA kernel; "
                         "--integrator-stats is unavailable.\n";
            return 1;
        }
    }
#endif
    int num_cells = 0;
    if (!checked_cell_count(grid_dim, num_cells)) {
        std::cerr << "grid dimension is too large: " << grid_dim << "\n";
        return 1;
    }

    pc::set_redshift(30.0);

    std::vector<CollapseState> host_cells(static_cast<std::size_t>(num_cells));
    std::vector<CollapseStepResult> cell_results(static_cast<std::size_t>(num_cells));
    BatchStatus batch_status{};
    double collapse_loop_walltime_sec = 0.0;
#if defined(__CUDACC__)
    double cuda_step_kernel_walltime_sec = 0.0;
    double cuda_step_kernel_event_sec = 0.0;
    double cuda_result_copy_walltime_sec = 0.0;
    double cuda_history_copy_walltime_sec = 0.0;
    double host_status_walltime_sec = 0.0;
    double final_state_copy_walltime_sec = 0.0;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaErrorNoDevice || device_count == 0) {
        std::cout << "CUDA run skipped: no CUDA-capable device\n";
        return 77;
    }
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    err = cudaMemcpyToSymbol(device_tolerances, &runtime_tolerances, sizeof(ToleranceConfig));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpyToSymbol(device_tolerances) failed: "
                  << cudaGetErrorString(err) << "\n";
        return 1;
    }
    cudaDeviceProp device_prop{};
    err = cudaGetDeviceProperties(&device_prop, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    if (threads_per_block <= 0 || threads_per_block > device_prop.maxThreadsPerBlock) {
        std::cerr << "invalid --threads-per-block " << threads_per_block
                  << " for device max " << device_prop.maxThreadsPerBlock << "\n";
        return 1;
    }

    CollapseState* device_cells = nullptr;
    CollapseStepResult* device_cell_results = nullptr;
    RODASMatrix* device_ros2s_matrices = nullptr;
    cudaEvent_t step_kernel_event_start = nullptr;
    cudaEvent_t step_kernel_event_stop = nullptr;

    err = cudaEventCreate(&step_kernel_event_start);
    if (err != cudaSuccess) {
        std::cerr << "cudaEventCreate start failed: " << cudaGetErrorString(err) << "\n";
        return 1;
    }
    err = cudaEventCreate(&step_kernel_event_stop);
    if (err != cudaSuccess) {
        std::cerr << "cudaEventCreate stop failed: " << cudaGetErrorString(err) << "\n";
        cudaEventDestroy(step_kernel_event_start);
        return 1;
    }

    err = cudaMalloc(&device_cells, sizeof(CollapseState) * host_cells.size());
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        cudaEventDestroy(step_kernel_event_stop);
        cudaEventDestroy(step_kernel_event_start);
        return 1;
    }

    err = cudaMalloc(&device_cell_results, sizeof(CollapseStepResult) * cell_results.size());
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_cells);
        cudaEventDestroy(step_kernel_event_stop);
        cudaEventDestroy(step_kernel_event_start);
        return 1;
    }

    if constexpr (ros2s_external_matrix) {
        err = cudaMalloc(&device_ros2s_matrices, sizeof(RODASMatrix) * host_cells.size());
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc ROS2S matrix scratch failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            cudaEventDestroy(step_kernel_event_stop);
            cudaEventDestroy(step_kernel_event_start);
            return 1;
        }
    }

    const int blocks = (num_cells + threads_per_block - 1) / threads_per_block;
    initialize_collapse_kernel<<<blocks, threads_per_block>>>(
        device_cells, device_cell_results, num_cells);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "initialize_collapse_kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_ros2s_matrices);
        cudaFree(device_cell_results);
        cudaFree(device_cells);
        cudaEventDestroy(step_kernel_event_stop);
        cudaEventDestroy(step_kernel_event_start);
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "initialize_collapse_kernel execution failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_ros2s_matrices);
        cudaFree(device_cell_results);
        cudaFree(device_cells);
        cudaEventDestroy(step_kernel_event_stop);
        cudaEventDestroy(step_kernel_event_start);
        return 1;
    }

    if (diagnostics.dump_history) {
        print_history_header();
    }
    if (diagnostics.positivity_report) {
        print_positivity_header();
    }
    const auto collapse_loop_start = std::chrono::steady_clock::now();
    for (int n = 0; n < nsteps; ++n) {
        const auto step_kernel_start = std::chrono::steady_clock::now();
        err = cudaEventRecord(step_kernel_event_start);
        if (err != cudaSuccess) {
            std::cerr << "cudaEventRecord start failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            cudaEventDestroy(step_kernel_event_stop);
            cudaEventDestroy(step_kernel_event_start);
            return 1;
        }
        if constexpr (active_kernels_only) {
            switch (integrator_choice) {
            case IntegratorChoice::VODE:
                collapse_step_kernel<IntegratorChoice::VODE, false, true><<<blocks, threads_per_block>>>(
                    device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                    cell_id_offset, perturb_cells, diagnostics);
                break;
            case IntegratorChoice::ROS2S:
                collapse_step_kernel<IntegratorChoice::ROS2S, false, false><<<blocks, threads_per_block>>>(
                    device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                    cell_id_offset, perturb_cells, diagnostics);
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_A:
                collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_A, false, false><<<blocks, threads_per_block>>>(
                    device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                    cell_id_offset, perturb_cells, diagnostics);
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_B:
                collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_B, false, false><<<blocks, threads_per_block>>>(
                    device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                    cell_id_offset, perturb_cells, diagnostics);
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_C:
                collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_C, false, false><<<blocks, threads_per_block>>>(
                    device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                    cell_id_offset, perturb_cells, diagnostics);
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_D:
            default:
                collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_D, false, false><<<blocks, threads_per_block>>>(
                    device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                    cell_id_offset, perturb_cells, diagnostics);
                break;
            }
        } else {
            switch (integrator_choice) {
            case IntegratorChoice::VODE:
                if (finite_difference_jacobian) {
                    collapse_step_kernel<IntegratorChoice::VODE, true, true><<<blocks, threads_per_block>>>(
                        device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                        cell_id_offset, perturb_cells, diagnostics);
                } else {
                    collapse_step_kernel<IntegratorChoice::VODE, false, true><<<blocks, threads_per_block>>>(
                        device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                        cell_id_offset, perturb_cells, diagnostics);
                }
                break;
            case IntegratorChoice::ROS2S:
                if (finite_difference_jacobian) {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROS2S, true, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROS2S, true, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROS2S, true, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                } else {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROS2S, false, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROS2S, false, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROS2S, false, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                }
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_A:
                if (finite_difference_jacobian) {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_A, true, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_A, true, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_A, true, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                } else {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_A, false, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_A, false, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_A, false, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                }
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_B:
                if (finite_difference_jacobian) {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_B, true, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_B, true, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_B, true, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                } else {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_B, false, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_B, false, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_B, false, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                }
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_C:
                if (finite_difference_jacobian) {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_C, true, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_C, true, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_C, true, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                } else {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_C, false, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_C, false, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_C, false, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                }
                break;
            case IntegratorChoice::ROSENBROCK_SANDU_D:
            default:
                if (finite_difference_jacobian) {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_D, true, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_D, true, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_D, true, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                } else {
                    if constexpr (specialize_ros2s_stats) {
                        if (diagnostics.integrator_stats) {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_D, false, true><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        } else {
                            collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_D, false, false><<<blocks, threads_per_block>>>(
                                device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                                cell_id_offset, perturb_cells, diagnostics);
                        }
                    } else {
                        collapse_step_kernel<IntegratorChoice::ROSENBROCK_SANDU_D, false, true><<<blocks, threads_per_block>>>(
                            device_cells, device_cell_results, device_ros2s_matrices, n, num_cells,
                            cell_id_offset, perturb_cells, diagnostics);
                    }
                }
                break;
            }
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "collapse_step_kernel launch failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            cudaEventDestroy(step_kernel_event_stop);
            cudaEventDestroy(step_kernel_event_start);
            return 1;
        }

        err = cudaEventRecord(step_kernel_event_stop);
        if (err != cudaSuccess) {
            std::cerr << "cudaEventRecord stop failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            cudaEventDestroy(step_kernel_event_stop);
            cudaEventDestroy(step_kernel_event_start);
            return 1;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "collapse_step_kernel execution failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            cudaEventDestroy(step_kernel_event_stop);
            cudaEventDestroy(step_kernel_event_start);
            return 1;
        }
        float step_kernel_event_ms = 0.0F;
        err = cudaEventElapsedTime(&step_kernel_event_ms,
                                   step_kernel_event_start, step_kernel_event_stop);
        if (err != cudaSuccess) {
            std::cerr << "cudaEventElapsedTime failed on step " << n << ": "
                      << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            cudaEventDestroy(step_kernel_event_stop);
            cudaEventDestroy(step_kernel_event_start);
            return 1;
        }
        cuda_step_kernel_event_sec += static_cast<double>(step_kernel_event_ms) * 1.0e-3;
        const auto step_kernel_end = std::chrono::steady_clock::now();
        cuda_step_kernel_walltime_sec +=
            std::chrono::duration<double>(step_kernel_end - step_kernel_start).count();

        const auto result_copy_start = std::chrono::steady_clock::now();
        err = cudaMemcpy(cell_results.data(), device_cell_results,
                         sizeof(CollapseStepResult) * cell_results.size(),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
            cudaFree(device_cell_results);
            cudaFree(device_cells);
            return 1;
        }
        const auto result_copy_end = std::chrono::steady_clock::now();
        cuda_result_copy_walltime_sec +=
            std::chrono::duration<double>(result_copy_end - result_copy_start).count();

        if (diagnostics.dump_history) {
            const auto history_copy_start = std::chrono::steady_clock::now();
            err = cudaMemcpy(host_cells.data(), device_cells,
                             sizeof(CollapseState) * host_cells.size(),
                             cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                std::cerr << "cudaMemcpy history cell failed: " << cudaGetErrorString(err) << "\n";
                cudaFree(device_cell_results);
                cudaFree(device_cells);
                cudaEventDestroy(step_kernel_event_stop);
                cudaEventDestroy(step_kernel_event_start);
                return 1;
            }
            const auto history_copy_end = std::chrono::steady_clock::now();
            cuda_history_copy_walltime_sec +=
                std::chrono::duration<double>(history_copy_end - history_copy_start).count();
            print_history_row(n, host_cells.front(), cell_results.front());
        }

        const auto host_status_start = std::chrono::steady_clock::now();
        batch_status = batch_status_from_results(cell_results, n + 1);
        const auto host_status_end = std::chrono::steady_clock::now();
        host_status_walltime_sec +=
            std::chrono::duration<double>(host_status_end - host_status_start).count();
        if (batch_status.failed_cell >= 0 || batch_status.failed_step.stop) {
            break;
        }
    }
    const auto collapse_loop_end = std::chrono::steady_clock::now();
    collapse_loop_walltime_sec =
        std::chrono::duration<double>(collapse_loop_end - collapse_loop_start).count();

    const auto final_state_copy_start = std::chrono::steady_clock::now();
    err = cudaMemcpy(host_cells.data(), device_cells, sizeof(CollapseState) * host_cells.size(),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(device_cell_results);
        cudaFree(device_cells);
        cudaEventDestroy(step_kernel_event_stop);
        cudaEventDestroy(step_kernel_event_start);
        return 1;
    }
    const auto final_state_copy_end = std::chrono::steady_clock::now();
    final_state_copy_walltime_sec =
        std::chrono::duration<double>(final_state_copy_end - final_state_copy_start).count();
    cudaFree(device_ros2s_matrices);
    cudaFree(device_cell_results);
    cudaFree(device_cells);
    cudaEventDestroy(step_kernel_event_stop);
    cudaEventDestroy(step_kernel_event_start);

    std::cout << "integration backend: CUDA per-step kernels, one cell per thread\n";
#else
    for (auto& cell : host_cells) {
        cell = make_collapse_state();
    }

    if (diagnostics.dump_history) {
        print_history_header();
    }
    if (diagnostics.positivity_report) {
        print_positivity_header();
    }
    const auto collapse_loop_start = std::chrono::steady_clock::now();
    for (int n = 0; n < nsteps; ++n) {
        for (std::size_t cell = 0; cell < host_cells.size(); ++cell) {
            if (!cell_results[cell].stop) {
                cell_results[cell] =
                    advance_collapse_step(host_cells[cell], n,
                                          static_cast<int>(cell) + cell_id_offset,
                                          perturb_cells, finite_difference_jacobian,
                                          diagnostics, integrator_choice, nullptr);
            }
        }

        if (diagnostics.dump_history) {
            print_history_row(n, host_cells.front(), cell_results.front());
        }

        batch_status = batch_status_from_results(cell_results, n + 1);
        if (batch_status.failed_cell >= 0 || batch_status.failed_step.stop) {
            break;
        }
        if (diagnostics.positivity_report) {
            for (const auto& cell : host_cells) {
                print_positivity_rows(integrator_choice, n, cell);
            }
        }
    }
    const auto collapse_loop_end = std::chrono::steady_clock::now();
    collapse_loop_walltime_sec =
        std::chrono::duration<double>(collapse_loop_end - collapse_loop_start).count();
    if (diagnostics.positivity_report) {
        print_positivity_footer();
    }

    std::cout << "integration backend: CPU serialized cells\n";
#endif

    if (host_cells.empty()) {
        std::cerr << "no cells were initialized\n";
        return 1;
    }

    const int representative_cell = batch_status.failed_cell >= 0 ? batch_status.failed_cell : 0;
    const auto& representative_state = host_cells[static_cast<std::size_t>(representative_cell)];
    const auto representative_step =
        batch_status.failed_cell >= 0
            ? batch_status.failed_step
            : cell_results[static_cast<std::size_t>(representative_cell)];
    const CollapseResult collapse{
        representative_state.initial_state, representative_state.state,
        representative_state.time, representative_state.completed_steps,
        representative_step.failed_step, representative_step.result};

    const auto initial_state = collapse.initial_state;
    const auto state = collapse.final_state;
    const auto t = collapse.time;

    if (collapse.result != integrators::IntegratorResult::SUCCESS) {
        std::cout << integrator_name(integrator_choice) << " failed in cell " << representative_cell
                  << " on collapse step " << collapse.failed_step
                  << " with code " << static_cast<int>(collapse.result) << "\n";
        std::cout << "time: " << t << "\n";
        std::cout << "completed steps: " << representative_state.completed_steps << "\n";
        std::cout << "density driver: " << representative_state.density_driver << "\n";
        std::cout << "rho: " << state.rho << "\n";
        std::cout << "T: " << state.T << "\n";
        std::cout << "Eint: " << state.e << "\n";
        for (int k = 0; k < pc::NumSpec; ++k) {
            std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(k)] << ": "
                      << state.xn[static_cast<std::size_t>(k)] << "\n";
        }
        return 1;
    }

    bool reference_pass = true;
    bool physical_pass = true;
    integrators::Real max_non_deuterium_species_rel_error = 0.0;
    integrators::Real max_thermodynamic_rel_error = 0.0;
    int min_completed_steps = std::numeric_limits<int>::max();
    int max_completed_steps = 0;
    int first_invalid_cell = -1;
    std::uint64_t total_completed_steps = 0;
    IntegratorWorkStats total_integrator_work{};

    for (std::size_t cell_index = 0; cell_index < host_cells.size(); ++cell_index) {
        const auto& cell = host_cells[cell_index];
        min_completed_steps = std::min(min_completed_steps, cell.completed_steps);
        max_completed_steps = std::max(max_completed_steps, cell.completed_steps);
        total_completed_steps += static_cast<std::uint64_t>(std::max(0, cell.completed_steps));
        if (diagnostics.integrator_stats) {
            add_integrator_work(total_integrator_work, cell.integrator_work);
        }

        bool cell_physical_pass = cell.completed_steps > 0 &&
                                  std::isfinite(cell.state.T) && cell.state.T > 0.0 &&
                                  std::isfinite(cell.state.e) && cell.state.e > 0.0 &&
                                  std::isfinite(cell.state.rho) && cell.state.rho > 0.0;

        for (int n = 0; n < pc::NumSpec; ++n) {
            const auto value = cell.state.xn[static_cast<std::size_t>(n)];
            const auto reference = reference_number_densities[static_cast<std::size_t>(n)];
            const auto denom = std::max(std::abs(reference), atol_spec);
            const auto rel_error = std::abs(value - reference) / denom;
            cell_physical_pass = cell_physical_pass &&
                                 std::isfinite(value) &&
                                 value >= pc::small_number_density_floor() / state_validity_floor_slop;
            if (!deuterium_bearing_species[static_cast<std::size_t>(n)]) {
                max_non_deuterium_species_rel_error =
                    std::max(max_non_deuterium_species_rel_error, rel_error);
            }
            reference_pass = reference_pass && nearly_equal(
                                                  value, reference,
                                                  reference_species_rtol[static_cast<std::size_t>(n)], atol_spec);
        }

        reference_pass = reference_pass && nearly_equal(cell.state.T, reference_temperature,
                                                       reference_thermodynamic_rtol, atol_spec);
        reference_pass = reference_pass && nearly_equal(cell.state.e, reference_eint,
                                                       reference_thermodynamic_rtol, atol_enuc);
        reference_pass = reference_pass && nearly_equal(cell.state.rho, reference_rho, rtol_spec, atol_spec);

        const auto temperature_rel_error =
            std::abs(cell.state.T - reference_temperature) /
            std::max(std::abs(reference_temperature), atol_spec);
        const auto internal_energy_rel_error =
            std::abs(cell.state.e - reference_eint) /
            std::max(std::abs(reference_eint), atol_enuc);
        max_thermodynamic_rel_error =
            std::max(max_thermodynamic_rel_error,
                     std::max(temperature_rel_error, internal_energy_rel_error));
        if (!cell_physical_pass && first_invalid_cell < 0) {
            first_invalid_cell = static_cast<int>(cell_index);
        }
        physical_pass = physical_pass && cell_physical_pass;
    }

    const bool pass = perturb_cells ? physical_pass : reference_pass;

    std::cout << "grid: " << grid_dim << "^3 cells (" << num_cells << " total)\n";
    std::cout << "integrator: " << integrator_name(integrator_choice) << "\n";
#if defined(__CUDACC__)
    std::cout << "cuda threads per block: " << threads_per_block << "\n";
#endif
    std::cout << "rtol: species=" << active_rtol_spec()
              << " energy=" << active_rtol_enuc() << "\n";
    std::cout << "atol: species=" << active_atol_spec()
              << " energy=" << active_atol_enuc() << "\n";
    std::cout << "jacobian: "
              << (finite_difference_jacobian ? "finite-difference" : "analytic") << "\n";
    std::cout << "cell perturbations: "
              << (perturb_cells ? "+/-10% every 20 collapse steps" : "disabled") << "\n";
    std::cout << "completed global kernel/step launches: "
              << batch_status.completed_global_steps << "\n";
    std::cout << "collapse loop walltime: " << collapse_loop_walltime_sec << " s\n";
#if defined(__CUDACC__)
    std::cout << "cuda step kernel+synchronize walltime: "
              << cuda_step_kernel_walltime_sec << " s\n";
    std::cout << "cuda step kernel event time: "
              << cuda_step_kernel_event_sec << " s\n";
    std::cout << "cuda per-step result copy walltime: "
              << cuda_result_copy_walltime_sec << " s\n";
    std::cout << "cuda history copy walltime: "
              << cuda_history_copy_walltime_sec << " s\n";
    std::cout << "host status reduction walltime: "
              << host_status_walltime_sec << " s\n";
    std::cout << "final state copy walltime: "
              << final_state_copy_walltime_sec << " s\n";
#endif
    std::cout << "completed collapse steps per cell: "
              << min_completed_steps << "..." << max_completed_steps << "\n";
    if (diagnostics.integrator_stats) {
        const auto per_cell = [num_cells](std::uint64_t value) {
            return static_cast<double>(value) / static_cast<double>(num_cells);
        };
        const auto per_completed_step = [total_completed_steps](std::uint64_t value) {
            return total_completed_steps == 0
                       ? 0.0
                       : static_cast<double>(value) / static_cast<double>(total_completed_steps);
        };
        std::cout << "integrator work totals: internal_steps=" << total_integrator_work.internal_steps
                  << " rhs=" << total_integrator_work.rhs_calls
                  << " jacobian=" << total_integrator_work.jacobian_calls
                  << " decomp=" << total_integrator_work.decompositions
                  << " solve=" << total_integrator_work.linear_solves
                  << " accepted=" << total_integrator_work.accepted_steps
                  << " rejected=" << total_integrator_work.rejected_steps
                  << " negative_rejected=" << total_integrator_work.negative_rejected_steps
                  << " error_failures=" << total_integrator_work.error_failures << "\n";
        std::cout << "integrator work per cell: internal_steps="
                  << per_cell(total_integrator_work.internal_steps)
                  << " rhs=" << per_cell(total_integrator_work.rhs_calls)
                  << " jacobian=" << per_cell(total_integrator_work.jacobian_calls)
                  << " decomp=" << per_cell(total_integrator_work.decompositions)
                  << " solve=" << per_cell(total_integrator_work.linear_solves)
                  << " accepted=" << per_cell(total_integrator_work.accepted_steps)
                  << " rejected=" << per_cell(total_integrator_work.rejected_steps)
                  << " negative_rejected=" << per_cell(total_integrator_work.negative_rejected_steps)
                  << " error_failures=" << per_cell(total_integrator_work.error_failures) << "\n";
        std::cout << "integrator work per completed collapse step: internal_steps="
                  << per_completed_step(total_integrator_work.internal_steps)
                  << " rhs=" << per_completed_step(total_integrator_work.rhs_calls)
                  << " jacobian=" << per_completed_step(total_integrator_work.jacobian_calls)
                  << " decomp=" << per_completed_step(total_integrator_work.decompositions)
                  << " solve=" << per_completed_step(total_integrator_work.linear_solves)
                  << " accepted=" << per_completed_step(total_integrator_work.accepted_steps)
                  << " rejected=" << per_completed_step(total_integrator_work.rejected_steps)
                  << " negative_rejected="
                  << per_completed_step(total_integrator_work.negative_rejected_steps)
                  << " error_failures=" << per_completed_step(total_integrator_work.error_failures) << "\n";
    }
    std::cout << "representative cell: " << representative_cell << "\n";
    std::cout << "representative time: " << t << "\n";
    std::cout << "T initial: " << initial_state.T << "\n";
    std::cout << "T final:   " << state.T << "\n";
    std::cout << "Eint initial: " << initial_state.e << "\n";
    std::cout << "Eint final:   " << state.e << "\n";
    std::cout << "rho initial: " << initial_state.rho << "\n";
    std::cout << "rho final:   " << state.rho << "\n";
    std::cout << "max non-deuterium species relative error vs Microphysics reference: "
              << max_non_deuterium_species_rel_error << "\n";
    std::cout << "max T/Eint relative error vs Microphysics reference: "
              << max_thermodynamic_rel_error << "\n";
    std::cout << "state validity: " << (physical_pass ? "PASS" : "FAIL") << "\n";
    std::cout << "reference comparison: " << (reference_pass ? "PASS" : "FAIL")
              << (perturb_cells ? " (diagnostic for perturbed run)" : "") << "\n";

    if (!pass || diagnostics.dump_final_state) {
        if (first_invalid_cell >= 0) {
            const auto& invalid = host_cells[static_cast<std::size_t>(first_invalid_cell)];
            std::cout << "\nFirst invalid final state: cell " << first_invalid_cell << "\n";
            std::cout << "completed steps: " << invalid.completed_steps << "\n";
            std::cout << "time: " << invalid.time << "\n";
            std::cout << "density driver: " << invalid.density_driver << "\n";
            std::cout << "rho: " << invalid.state.rho << "\n";
            std::cout << "T: " << invalid.state.T << "\n";
            std::cout << "Eint: " << invalid.state.e << "\n";
            for (int n = 0; n < pc::NumSpec; ++n) {
                std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                          << invalid.state.xn[static_cast<std::size_t>(n)] << "\n";
            }
        }
        std::cout << "\nFinal number densities:\n";
        for (int n = 0; n < pc::NumSpec; ++n) {
            std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                      << state.xn[static_cast<std::size_t>(n)]
                      << " (reference " << reference_number_densities[static_cast<std::size_t>(n)] << ")\n";
        }
    }

    return pass ? 0 : 1;
}
