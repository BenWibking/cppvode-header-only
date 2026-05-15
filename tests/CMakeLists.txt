# ABOUTME: CMake configuration for unit tests
# ABOUTME: Tests individual components of the integrator library
set(INTEGRATORS_BUILD_GPU_TESTS OFF)
if(BUILD_CUDA_TESTS AND CMAKE_CUDA_COMPILER)
    set(INTEGRATORS_BUILD_GPU_TESTS ON)
endif()

if(NOT INTEGRATORS_BUILD_GPU_TESTS)
add_executable(test_linear_algebra test_linear_algebra.cpp)
target_link_libraries(test_linear_algebra PRIVATE integrators)

add_executable(test_convergence test_convergence.cpp)  
target_link_libraries(test_convergence PRIVATE integrators)

# Add as tests
add_test(NAME linear_algebra_test COMMAND test_linear_algebra)
add_test(NAME convergence_test COMMAND test_convergence)

add_executable(test_vode_stiff_decay test_vode_stiff_decay.cpp)
target_link_libraries(test_vode_stiff_decay PRIVATE integrators)
add_test(NAME vode_stiff_decay_test COMMAND test_vode_stiff_decay)

# HIRES stiff benchmark (VODE)
add_executable(test_vode_hires test_vode_hires.cpp)
target_link_libraries(test_vode_hires PRIVATE integrators)
add_test(NAME vode_hires_test COMMAND test_vode_hires)

# Nelson astrochemistry L2 relative error test
add_executable(test_vode_nelson test_vode_nelson.cpp)
target_link_libraries(test_vode_nelson PRIVATE integrators)
add_test(NAME vode_nelson_test COMMAND test_vode_nelson)

# Strict Robertson convergence/self-consistency test
add_executable(test_vode_robertson_strict test_vode_robertson_strict.cpp)
target_link_libraries(test_vode_robertson_strict PRIVATE integrators)
add_test(NAME vode_robertson_strict_test COMMAND test_vode_robertson_strict)

# ROS2S Robertson regression
add_executable(test_rosenbrock_robertson test_rosenbrock_robertson.cpp)
target_link_libraries(test_rosenbrock_robertson PRIVATE integrators)
add_test(NAME rosenbrock_robertson_test COMMAND test_rosenbrock_robertson)

# ROS2S Rosenbrock tableau regression
add_executable(test_rosenbrock test_rosenbrock.cpp)
target_link_libraries(test_rosenbrock PRIVATE integrators)
add_test(NAME rosenbrock_test COMMAND test_rosenbrock)

# Sandu positivity-favorable Rosenbrock method regressions
add_executable(test_rosenbrock_sandu test_rosenbrock_sandu.cpp)
target_link_libraries(test_rosenbrock_sandu PRIVATE integrators)
add_test(NAME rosenbrock_sandu_test COMMAND test_rosenbrock_sandu)

# Linear invariants conservation with VODE
add_executable(test_vode_linear_invariants test_vode_linear_invariants.cpp)
target_link_libraries(test_vode_linear_invariants PRIVATE integrators)
add_test(NAME vode_linear_invariants_test COMMAND test_vode_linear_invariants)

# Numerical Jacobian variant
add_executable(test_vode_linear_invariants_numjac test_vode_linear_invariants_numjac.cpp)
target_link_libraries(test_vode_linear_invariants_numjac PRIVATE integrators)
add_test(NAME vode_linear_invariants_numjac_test COMMAND test_vode_linear_invariants_numjac)

# Backward Euler numerical correctness regressions
add_executable(test_backward_euler_regressions test_backward_euler_regressions.cpp)
target_link_libraries(test_backward_euler_regressions PRIVATE integrators)
add_test(NAME backward_euler_regressions_test COMMAND test_backward_euler_regressions)

# Khokhlov YASS integrator
add_executable(test_yass test_yass.cpp)
target_link_libraries(test_yass PRIVATE integrators)
add_test(NAME yass_test COMMAND test_yass)

# Optional Jacobian API contract regression
add_executable(test_optional_jacobian test_optional_jacobian.cpp)
target_link_libraries(test_optional_jacobian PRIVATE integrators)
add_test(NAME optional_jacobian_test COMMAND test_optional_jacobian)

# VODE Jacobian accounting regression
add_executable(test_vode_jacobian_counter test_vode_jacobian_counter.cpp)
target_link_libraries(test_vode_jacobian_counter PRIVATE integrators)
add_test(NAME vode_jacobian_counter_test COMMAND test_vode_jacobian_counter)

else()
    add_executable(test_gpu_all test_gpu_all.cu)

    # The headers require C++20, so CUDA tests must compile as CUDA C++20 too.
    if(NOT (DEFINED CMAKE_CUDA20_STANDARD_COMPILE_OPTION
            OR "cuda_std_20" IN_LIST CMAKE_CUDA_COMPILE_FEATURES
            OR CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0))
        message(FATAL_ERROR
            "test_gpu_all requires CUDA C++20; use CUDA 12+ or a CMake CUDA "
            "toolchain that advertises cuda_std_20")
    endif()
    set_target_properties(test_gpu_all PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF)
    message(STATUS "test_gpu_all: using C++20 for CUDA (compiler ${CMAKE_CUDA_COMPILER_VERSION})")

    # Common CUDA target properties
    set_target_properties(test_gpu_all PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 90)

    if(INTEGRATORS_USE_CUSOLVERDX)
        set_target_properties(test_gpu_all PROPERTIES
            INTERPROCEDURAL_OPTIMIZATION ON)
    endif()

    # Enable extended device lambda support for NVCC
    target_compile_options(test_gpu_all PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)

    # Silence Clang host warnings about GNU-style line markers produced by NVCC
    # Only apply when compiling CUDA with a Clang host to avoid breaking GCC/MSVC.
    target_compile_options(test_gpu_all PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-gnu-line-marker>)
    target_link_libraries(test_gpu_all PRIVATE integrators)
    add_test(NAME gpu_all_tests COMMAND test_gpu_all)
    # Mark as GPU test so CI can identify/skip running if no GPU is present
    set_tests_properties(gpu_all_tests PROPERTIES
        LABELS "GPU"
        SKIP_RETURN_CODE 77)
endif()
