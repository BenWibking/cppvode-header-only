// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: Primordial chemistry one-zone collapse test ported from AMReX Microphysics
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include <integrators/primordial_chem.hpp>
#include <integrators/vode.hpp>

namespace pc = integrators::primordial_chem;

namespace {

constexpr integrators::Real tff_reduc = 1.0e-1;
constexpr int nsteps = 1000;
constexpr integrators::Real temperature = 1.0e2;
constexpr integrators::Real rtol_spec = 1.0e-4;
constexpr integrators::Real atol_spec = 1.0e-4;
constexpr integrators::Real rtol_enuc = 1.0e-6;
constexpr integrators::Real atol_enuc = 1.0e-6;

constexpr std::array<integrators::Real, pc::NumSpec> initial_number_densities{
    1.0e-4, 1.0e-4, 1.0e0,  1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40,
    1.0e-40, 1.0e-6, 1.0e-40, 1.0e-40, 1.0e-40, 1.0e-40, 0.0775};

constexpr std::array<integrators::Real, pc::NumSpec> reference_number_densities{
    19911.96049,      19897.56689,      1.61920532e17, 1.780427462, 1.0e-100,
    0.9313391106,     16.17402851,      1.0e-100,      3.378785985e17,
    1.0e-100,         8.912464329,      2.624158281e-60, 6.363655862e-12,
    6.491340291e16};

constexpr integrators::Real reference_temperature = 3032.992479;
constexpr integrators::Real reference_eint = 2.721837163e11;
constexpr integrators::Real reference_rho = 1.836285633e-6;

void configure_microphysics_tolerances(integrators::VODEState<pc::neqs>& state) {
    state.use_vector_tolerances = true;
    for (int n = 0; n < pc::NumSpec; ++n) {
        state.rtol_vec[static_cast<std::size_t>(n)] = rtol_spec;
        state.atol_vec[static_cast<std::size_t>(n)] = atol_spec;
    }
    state.rtol_vec[pc::NumSpec] = rtol_enuc;
    state.atol_vec[pc::NumSpec] = atol_enuc;
    state.rtol = rtol_spec;
    state.atol = atol_spec;
    state.max_steps = 150000;
    state.HMXI = 1.0e-30;
    state.constrained_components = pc::NumSpec;
    state.reject_change_buffer = 1.0e100;
    state.species_failure_tolerance = 1.0e-2;
    state.clean_constrained_components = true;
    state.component_floor = pc::small_x;
}

integrators::IntegratorResult burn_once(pc::burn_t& state, integrators::Real dt) {
    pc::eos_rt(state);

    auto integrator = integrators::VODE<pc::PrimordialChem>{};
    auto vode_state = integrators::VODEState<pc::neqs>{};
    configure_microphysics_tolerances(vode_state);

    vode_state.t = 0.0;
    vode_state.tout = dt;
    vode_state.jacobian_analytic = true;
    for (int n = 0; n < pc::NumSpec; ++n) {
        vode_state.y[static_cast<std::size_t>(n)] = state.xn[static_cast<std::size_t>(n)];
    }
    vode_state.y[pc::NumSpec] = state.e;

    pc::PrimordialChem::state_type problem_state{};
    const auto result = integrator.integrate(problem_state, vode_state);

    if (result == integrators::IntegratorResult::SUCCESS) {
        for (int n = 0; n < pc::NumSpec; ++n) {
            state.xn[static_cast<std::size_t>(n)] = vode_state.y[static_cast<std::size_t>(n)];
        }
        state.e = vode_state.y[pc::NumSpec];
        state.success = true;
    } else {
        state.success = false;
    }

    return result;
}

integrators::IntegratorResult burn(pc::burn_t& state, integrators::Real dt) {
    return burn_once(state, dt);
}

bool nearly_equal(integrators::Real value, integrators::Real reference,
                  integrators::Real rtol, integrators::Real atol) {
    return std::abs(value - reference) <= atol + rtol * std::abs(reference);
}

} // namespace

int main() {
    std::cout << std::setprecision(std::numeric_limits<integrators::Real>::max_digits10);
    std::cout << "Primordial Chemistry One-Zone Collapse\n";
    std::cout << "======================================\n\n";

    pc::set_redshift(30.0);

    pc::burn_t state;
    state.T = temperature;
    state.xn = initial_number_densities;
    state.rho = pc::density(state.xn);
    pc::normalize_number_densities_to_density(state);
    pc::eos_rt(state);

    const auto initial_state = state;
    integrators::Real t = 0.0;
    integrators::Real dd = state.rho;
    int completed_steps = 0;

    for (int n = 0; n < nsteps; ++n) {
        const integrators::Real dd1 = dd;
        const integrators::Real rhotmp = pc::density(state.xn);
        const integrators::Real tff = std::sqrt(pc::pi * 3.0 / (32.0 * rhotmp * pc::grav_constant));
        const integrators::Real dt = tff_reduc * tff;

        dd += dt * (dd / tff);

        if (dt < 10.0 || dd > 2.0e-6) {
            break;
        }

        const integrators::Real density_ratio = dd / dd1;
        for (auto& xn : state.xn) {
            xn *= density_ratio;
        }
        state.rho *= density_ratio;

        const auto result = burn(state, dt);
        if (result != integrators::IntegratorResult::SUCCESS) {
            std::cout << "VODE failed on collapse step " << n
                      << " with code " << static_cast<int>(result) << "\n";
            std::cout << "time: " << t << "\n";
            std::cout << "rho: " << state.rho << "\n";
            std::cout << "T: " << state.T << "\n";
            std::cout << "Eint: " << state.e << "\n";
            for (int k = 0; k < pc::NumSpec; ++k) {
                std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(k)] << ": "
                          << state.xn[static_cast<std::size_t>(k)] << "\n";
            }
            return 1;
        }

        pc::floor_and_normalize_number_densities(state);
        pc::balance_charge(state);
        pc::floor_and_normalize_number_densities(state);
        pc::eos_re(state);

        t += dt;
        completed_steps += 1;
    }

    bool pass = true;
    integrators::Real max_species_rel_error = 0.0;
    for (int n = 0; n < pc::NumSpec; ++n) {
        const auto value = state.xn[static_cast<std::size_t>(n)];
        const auto reference = reference_number_densities[static_cast<std::size_t>(n)];
        const auto denom = std::max(std::abs(reference), atol_spec);
        max_species_rel_error = std::max(max_species_rel_error, std::abs(value - reference) / denom);
        pass = pass && nearly_equal(value, reference, rtol_spec, atol_spec);
    }

    pass = pass && nearly_equal(state.T, reference_temperature, rtol_spec, atol_spec);
    pass = pass && nearly_equal(state.e, reference_eint, rtol_spec, atol_spec);
    pass = pass && nearly_equal(state.rho, reference_rho, rtol_spec, atol_spec);

    std::cout << "completed collapse steps: " << completed_steps << "\n";
    std::cout << "time: " << t << "\n";
    std::cout << "T initial: " << initial_state.T << "\n";
    std::cout << "T final:   " << state.T << "\n";
    std::cout << "Eint initial: " << initial_state.e << "\n";
    std::cout << "Eint final:   " << state.e << "\n";
    std::cout << "rho initial: " << initial_state.rho << "\n";
    std::cout << "rho final:   " << state.rho << "\n";
    std::cout << "max species relative error vs Microphysics reference: "
              << max_species_rel_error << "\n";
    std::cout << "reference comparison: " << (pass ? "PASS" : "FAIL") << "\n";

    if (!pass) {
        std::cout << "\nFinal number densities:\n";
        for (int n = 0; n < pc::NumSpec; ++n) {
            std::cout << "  " << pc::short_spec_names[static_cast<std::size_t>(n)] << ": "
                      << state.xn[static_cast<std::size_t>(n)]
                      << " (reference " << reference_number_densities[static_cast<std::size_t>(n)] << ")\n";
        }
    }

    return pass ? 0 : 1;
}
