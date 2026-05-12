#!/usr/bin/env bash
#SBATCH --job-name=primordial_ros2s_h200
#SBATCH --partition=general-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --constraint=h200
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

grid_dim="${GRID_DIM:-64}"
binary="${BINARY:-./build/primordial_ros2s}"
reference="${REFERENCE:-final_states_grid${grid_dim}_cpu.bin}"

if [[ ! -x "$binary" ]]; then
  echo "missing executable: $binary" >&2
  echo "build the CUDA target before submitting this job" >&2
  exit 1
fi

if [[ ! -f "$reference" ]]; then
  echo "missing CPU reference solution: $reference" >&2
  echo "run run_primordial_ros2s_ref_amd24.slurm first, or set REFERENCE=/path/to/reference.bin" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is not available; cannot verify GPU model" >&2
  exit 1
fi

nvidia-smi
gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if ! grep -qi 'H200' <<<"$gpu_names"; then
  echo "expected an H200 GPU, but Slurm allocated:" >&2
  echo "$gpu_names" >&2
  exit 1
fi

srun --ntasks=1 --gres=gpu:h200:1 \
  "$binary" --grid "$grid_dim" --perturb --compare-final-state "$reference"
