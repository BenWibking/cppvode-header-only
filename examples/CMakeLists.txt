# ABOUTME: CMake configuration for example programs
# ABOUTME: Builds simple ODE and Robertson problem examples
add_executable(simple_ode simple_ode.cpp)
target_link_libraries(simple_ode PRIVATE integrators)

add_executable(robertson robertson.cpp)
target_link_libraries(robertson PRIVATE integrators)

# Add examples as tests
add_test(NAME simple_ode_test COMMAND simple_ode)
add_test(NAME robertson_test COMMAND robertson)

# Fortran DVODE driver to log per-step sizes for Robertson problem
add_executable(robertson_dvode robertson_dvode.f90 ${CMAKE_SOURCE_DIR}/extern/dvode.f)
set_target_properties(robertson_dvode PROPERTIES LINKER_LANGUAGE Fortran)
set_source_files_properties(${CMAKE_SOURCE_DIR}/extern/dvode.f PROPERTIES Fortran_FORMAT FIXED Fortran_PREPROCESS OFF)
# Use legacy Fortran standard to silence warnings from vintage DVODE code
if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU|Intel|IntelLLVM")
    target_compile_options(robertson_dvode PRIVATE -std=legacy)
endif()
add_test(NAME robertson_dvode_runs COMMAND robertson_dvode)
