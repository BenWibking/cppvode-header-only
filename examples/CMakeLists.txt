# ABOUTME: CMake configuration for example programs
# ABOUTME: Builds simple ODE and Robertson problem examples
add_executable(simple_ode simple_ode.cpp)
target_link_libraries(simple_ode PRIVATE integrators)

add_executable(robertson robertson.cpp)
target_link_libraries(robertson PRIVATE integrators)

add_executable(primordial_chem primordial_chem.cpp)
target_link_libraries(primordial_chem PRIVATE integrators)
if(INTEGRATORS_CUDA_ENABLED)
    set_source_files_properties(primordial_chem.cpp PROPERTIES LANGUAGE CUDA)
    set_target_properties(primordial_chem PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 90)
    target_compile_options(primordial_chem PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr --fmad=false>)
    target_compile_options(primordial_chem PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-gnu-line-marker>)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(primordial_chem PRIVATE -ffp-contract=off)
endif()

# Add examples as tests
add_test(NAME simple_ode_test COMMAND simple_ode)
add_test(NAME robertson_test COMMAND robertson)
add_test(NAME primordial_chem_test COMMAND primordial_chem)
if(INTEGRATORS_CUDA_ENABLED)
    set_tests_properties(primordial_chem_test PROPERTIES
        LABELS "GPU"
        SKIP_RETURN_CODE 77)
endif()

# Fortran DVODE driver to log per-step sizes for Robertson problem
add_executable(robertson_dvode robertson_dvode.f90 ${CMAKE_SOURCE_DIR}/extern/dvode.f)
set_target_properties(robertson_dvode PROPERTIES LINKER_LANGUAGE Fortran)
set_source_files_properties(${CMAKE_SOURCE_DIR}/extern/dvode.f PROPERTIES Fortran_FORMAT FIXED Fortran_PREPROCESS OFF)
# Use legacy Fortran standard to silence warnings from vintage DVODE code
if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU|Intel|IntelLLVM")
    target_compile_options(robertson_dvode PRIVATE -std=legacy)
endif()
add_test(NAME robertson_dvode_runs COMMAND robertson_dvode)
