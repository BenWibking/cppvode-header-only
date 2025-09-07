# ABOUTME: CMake configuration for example programs
# ABOUTME: Builds simple ODE and Robertson problem examples
add_executable(simple_ode simple_ode.cpp)
target_link_libraries(simple_ode PRIVATE integrators)

add_executable(robertson robertson.cpp)
target_link_libraries(robertson PRIVATE integrators)

# Add examples as tests
add_test(NAME simple_ode_test COMMAND simple_ode)
add_test(NAME robertson_test COMMAND robertson)