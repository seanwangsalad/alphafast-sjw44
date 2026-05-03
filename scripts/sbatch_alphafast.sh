#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
#SBATCH --job-name=alphafast
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-common
#SBATCH --output=logs/alphafast_%j.out
#SBATCH --error=logs/alphafast_%j.err
#
# SLURM wrapper for run_alphafast.sh
#
# Usage:
#   sbatch scripts/sbatch_alphafast.sh \
#       --input_dir  /path/to/inputs \
#       --output_dir /path/to/outputs \
#       --db_dir     /path/to/databases \
#       --weights_dir /path/to/weights \
#       [--container /path/to/alphafast.sif] \
#       [--temp_dir  /scratch/$USER/alphafast_tmp] \
#       [--mmseqs_split_memory_limit 64G] \
#       [--lowram] \
#       [--gpu_devices 0,1,2,3] \
#       [--jax_compilation_cache_dir /scratch/$USER/alphafast_jax_cache]
#
# All arguments are forwarded to run_alphafast.sh.
#
# For multi-GPU, also bump #SBATCH --gres=gpu:N to match --gpu_devices.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs

exec "$SCRIPT_DIR/run_alphafast.sh" "$@"
