#!/usr/bin/env bash
#SBATCH --job-name=primordial_ros2s_ref
#SBATCH --partition=amd24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=189521M
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

nthreads="${NTHREADS:-${SLURM_CPUS_PER_TASK:-48}}"
export OMP_NUM_THREADS="$nthreads"

srun --ntasks=1 --cpus-per-task="$nthreads" \
  ./build/primordial_ros2s_ref --grid 64 --perturb -N "$nthreads"
