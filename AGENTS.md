# Repository Guidelines

## Project Structure & Module Organization
- `include/integrators/`: Header-only C++20 library (VODE, Backward Euler, linear algebra, types). Include via `#include <integrators/integrators.hpp>`.
- `examples/`: Small programs demonstrating usage (`simple_ode.cpp`, `robertson.cpp`).
- `tests/`: Executable tests using `assert` (no external framework).
- `CMakeLists.txt`: Top-level build; adds `examples/` and `tests/` subdirectories; installs headers.

## Build, Test, and Development Commands
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
- Build: `cmake --build build -j`
- Run all tests: `ctest --test-dir build --output-on-failure`
- Run examples: `build/examples/simple_ode` and `build/examples/robertson`
- Useful options: `-DBUILD_TESTS=ON` (default), `-DBUILD_EXAMPLES=ON` (default), `-DINTEGRATORS_VODE_DEBUG=ON` for verbose VODE logs. Debug builds enable AddressSanitizer.

## Coding Style & Naming Conventions
- C++20, header-only. Prefer standard containers and algorithms.
- Files: `snake_case.hpp` in `include/integrators/`.
- Types/enums: `CamelCase` (e.g., `IntegratorState`, `ProblemTraits`). Functions/variables: `snake_case`.
- Indentation: 4 spaces; braces on same line for functions/types; keep headers self-contained.
- Use `clang-format` (LLVM style recommended). Keep includes minimal and ordered (project after standard headers).

## Testing Guidelines
- Location: `tests/` with files named `test_*.cpp` producing a zero-exit executable.
- Add in `tests/CMakeLists.txt` via `add_executable(...)`, `target_link_libraries(... integrators)`, and `add_test(NAME ... COMMAND ...)`.
- Run locally with `ctest` (above). Aim to cover numerical edge cases (stiffness, step adaptation, Jacobian paths).

## Commit & Pull Request Guidelines
- Commits: imperative and scoped (e.g., `feat(vode): improve corrector convergence`, `fix(linear_algebra): pivoting bug`). Keep diffs focused; reference issues when applicable.
- PRs: include a clear summary, motivation, and before/after notes. Link issues, show `ctest` output, and include example outputs for new integrators or options.

## Security & Configuration Tips
- Avoid UB in numerical code; prefer checked bounds when feasible.
- Reproducibility: document `CMAKE_CXX_COMPILER` and build type used.
- Diagnostics: use `-DINTEGRATORS_VODE_DEBUG=ON` and Debug builds to trace failures.
