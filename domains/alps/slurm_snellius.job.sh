#!/bin/bash

# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Job file for Snellius. Missing ressources are specfied in the sbatch call.
#SBATCH --partition=rome
#SBATCH --ntasks-per-core=1
#SBATCH --mail-user=m.pierzyna@tudelft.nl
#SBATCH --mail-type=END,FAIL

# Set variables
STAGE=$1
SIM_DIR=$2

if [ -z "$STAGE" ] || [ -z "$SIM_DIR" ]; then
  echo "Usage: sbatch slurm_snellius.job.sh <stage> <sim_dir>"
  exit 1
fi

# Load modules
module load 2024
module load foss/2024a

# Run (unbuffered). `submit` passes the env file it was started with via $WRF_MASSIVE_ENV_FILE.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run wrf-massive run --stages="$STAGE" "$SIM_DIR"
