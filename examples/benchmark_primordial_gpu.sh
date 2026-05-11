#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# ABOUTME: Run GPU primordial chemistry timing comparisons for VODE and ROS2S
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
exe="${repo_dir}/build/examples/primordial_chem"

grid=16
extra_args=()
integrators=(vode ros2s)

while (($# > 0)); do
    case "$1" in
        --exe)
            if (($# < 2)); then
                echo "missing value for --exe" >&2
                exit 2
            fi
            exe="$2"
            shift 2
            ;;
        --grid)
            if (($# < 2)); then
                echo "missing value for --grid" >&2
                exit 2
            fi
            grid="$2"
            shift 2
            ;;
        --order)
            if (($# < 2)); then
                echo "missing value for --order" >&2
                exit 2
            fi
            case "$2" in
                vode,ros2s)
                    integrators=(vode ros2s)
                    ;;
                ros2s,vode)
                    integrators=(ros2s vode)
                    ;;
                *)
                    echo "unsupported --order value: $2" >&2
                    echo "expected vode,ros2s or ros2s,vode" >&2
                    exit 2
                    ;;
            esac
            shift 2
            ;;
        --)
            shift
            extra_args+=("$@")
            break
            ;;
        *)
            extra_args+=("$1")
            shift
            ;;
    esac
done

if [[ ! -x "${exe}" ]]; then
    echo "primordial_chem executable not found or not executable: ${exe}" >&2
    echo "build it first, for example: cmake --build build -j --target primordial_chem" >&2
    exit 2
fi

run_case() {
    local integrator="$1"
    local log
    local status
    log="$(mktemp)"
    echo "=== ${integrator} ==="
    set +e
    "${exe}" --grid "${grid}" --integrator "${integrator}" "${extra_args[@]}" >"${log}" 2>&1
    status=$?
    set -e
    if [[ "${status}" -eq 0 ]]; then
        grep -E "^(integration backend|grid:|integrator:|cuda threads per block:|jacobian:|completed global kernel/step launches:|collapse loop walltime:|cuda step kernel\+synchronize walltime:|cuda step kernel event time:|cuda per-step result copy walltime:|cuda history copy walltime:|host status reduction walltime:|final state copy walltime:|completed collapse steps per cell:|integrator work totals:|integrator work per cell:|integrator work per completed collapse step:|state validity:|reference comparison:)" "${log}"
    else
        cat "${log}"
        rm -f "${log}"
        return "${status}"
    fi
    rm -f "${log}"
}

echo "Primordial chemistry GPU timing comparison"
echo "executable: ${exe}"
echo "grid: ${grid}^3"
printf 'order:'
printf ' %s' "${integrators[@]}"
printf '\n'
if ((${#extra_args[@]} > 0)); then
    printf 'extra args:'
    printf ' %q' "${extra_args[@]}"
    printf '\n'
fi
echo

for integrator in "${integrators[@]}"; do
    run_case "${integrator}"
    echo
done
