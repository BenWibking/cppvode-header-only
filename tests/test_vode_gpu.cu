// SPDX-License-Identifier: BSD-3-Clause
// ABOUTME: GPU test that attempts to call VODE inside a CUDA kernel
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <integrators/integrators.hpp>

#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#endif

using namespace integrators;

#ifdef __CUDACC__
template <typename T>
__global__ void ParallelForKernelGPU(int N, T f) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        f(i);
    }
}
#endif

template <typename T>
void ParallelForKernelCPU(int N, T f) {
    for (int i = 0; i < N; i++) {
        f(i);
    }
}

template <typename T>
void ParallelFor(int N, T f) {
    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;
#ifdef __CUDACC__
    ParallelForKernelGPU<<<numBlocks, blockSize>>>(N, f);
#else
    ParallelForKernelCPU(N, f);
#endif
}

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// Simple exponential growth ODE: y' = k y, analytic J = k
struct ExpODE {
    static constexpr size_type neqs = 1;
    using state_type = std::array<Real, neqs>;
    using rhs_type = std::array<Real, neqs>;
    using jacobian_type = std::array<std::array<Real, neqs>, neqs>;
    static constexpr Real k = 2.5; // growth rate
    HOST_DEVICE static void rhs(Real /*t*/, const state_type& y, rhs_type& dydt) {
        dydt[0] = k * y[0];
    }
    HOST_DEVICE static void jacobian(Real /*t*/, const state_type& /*y*/, jacobian_type& J) {
        J[0][0] = k;
    }
};

int main() {
    // Intention: each GPU thread integrates a tiny ODE with VODE
    const int N = 256;
    const Real t0 = 0.0;
    const Real dt = 0.1;
    const Real t1 = t0 + dt;
    const Real y0 = 1.0;
    const Real exact = std::exp(ExpODE::k * dt);
    const Real tol = 1.e-6;

    printf("GPU VODE test: N=%d, dt=%.6f\n", N, dt);

    // Buffers for initial values and results
    Real* y_out = nullptr;
    int* status = nullptr;
#ifdef __CUDACC__
    cudaMalloc(&y_out, N * sizeof(Real));
    cudaMalloc(&status, N * sizeof(int));
#else
    y_out = (Real*)std::malloc(N * sizeof(Real));
    status = (int*)std::malloc(N * sizeof(int));
#endif

    // Launch per-element integrator inside kernel/lambda
    ParallelFor(N, [=] HOST_DEVICE(int i) {
        (void)i;
        // Initialize VODE state and integrate y' = k y from t0 to t1
        auto integ = VODE<ExpODE>{};
        auto s = VODEState<ExpODE::neqs>{};
        s.jacobian_analytic = true;
        s.t = t0;
        s.tout = t1;
        s.y[0] = y0;
        s.rtol = tol;
        s.atol = 1.e-12;
        auto ps = ExpODE::state_type{y0};
        auto res = integ.integrate(ps, s);
        y_out[i] = s.y[0];
        status[i] = static_cast<int>(res);
    });

#ifdef __CUDACC__
    cudaDeviceSynchronize();
#endif

    // Validate results on host
#ifdef __CUDACC__
    thrust::device_ptr<Real> y_ptr(y_out);
    thrust::device_ptr<int> s_ptr(status);
    // Check all statuses are SUCCESS (== 1)
    int min_status = *thrust::min_element(s_ptr, s_ptr + N);
    int max_status = *thrust::max_element(s_ptr, s_ptr + N);
    if (min_status != 1 || max_status != 1) {
        printf("Device integration failed: status range [%d, %d]\n", min_status, max_status);
        return 1;
    }
    // Compute max absolute error
    struct AbsErr {
        Real exact;
        __host__ __device__ Real operator()(const Real& y) const { return std::fabs(y - exact); }
    } abs_err{exact};
    // Transform-reduce via max_element on transformed iterator is clunky; copy to temp array of errors
    Real* err_buf = nullptr;
    cudaMalloc(&err_buf, N * sizeof(Real));
    // Simple transform kernel
    auto err_kernel = [=] __device__(int i) { err_buf[i] = fabs(y_out[i] - exact); };
    ParallelForKernelGPU<<<(N + 127) / 128, 128>>>(N, err_kernel);
    cudaDeviceSynchronize();
    thrust::device_ptr<Real> e_ptr(err_buf);
    Real max_err = *thrust::max_element(e_ptr, e_ptr + N);
    cudaFree(err_buf);
#else
    // CPU fallback path (no CUDA): just check the first element we wrote
    int ok = 1;
    for (int i = 0; i < N; ++i) ok = ok && (status[i] == 1);
    if (!ok) {
        printf("CPU path: integration failed in at least one element\n");
        return 1;
    }
    Real max_err = 0.0;
    for (int i = 0; i < N; ++i) max_err = std::max(max_err, std::fabs(y_out[i] - exact));
#endif

    printf("Max error vs exact exp(kt): %.3e (tol=%.1e)\n", max_err, tol);
    if (max_err > 1e3 * tol) {
        printf("Error too large\n");
        return 1;
    }

    // Cleanup
#ifdef __CUDACC__
    cudaFree(y_out);
    cudaFree(status);
#else
    std::free(y_out);
    std::free(status);
#endif
    printf("GPU VODE test: PASSED\n");
    return 0;
}

