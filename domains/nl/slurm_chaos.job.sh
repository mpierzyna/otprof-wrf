#!/bin/bash

# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Job file for Chaos. Missing ressources are specfied in the sbatch call.
#SBATCH --partition=compute
#SBATCH --ntasks-per-core=1

# Set variables
CONDA_ENV=/home/max/mambaforge/envs/wrf_py312
STAGE=$1
SIM_DIR=$2

if [ -z "$STAGE" ] || [ -z "$SIM_DIR" ]; then
  echo "Usage: sbatch slurm_chaos.job.sh <stage> <sim_dir>"
  exit 1
fi

# Load conda environment
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Run (unbuffered)
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK \
NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK \
PYTHONUNBUFFERED=1 wrf-massive run --stages=$STAGE $SIM_DIR
