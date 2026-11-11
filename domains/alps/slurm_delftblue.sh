#!/bin/bash

# SPDX-FileCopyrightText: 2026 Maximilian Pierzyna
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Job file for DelftBlue. Missing ressources are specfied in the sbatch call.
#SBATCH --partition=compute-p1
#SBATCH --account=research-ceg-grs
#SBATCH --ntasks-per-core=1
#SBATCH --mail-type=END,FAIL

# Set variables
STAGE=$1
SIM_DIR=$2

if [ -z "$STAGE" ] || [ -z "$SIM_DIR" ]; then
  echo "Usage: sbatch slurm_delftblue.sh <stage> <sim_dir>"
  exit 1
fi

# Load modules
source /home/mpierzyna/wrf/wrf_wps_env.sh
module load slurm

# Make tmp WPS dir on node-local SSD
mkdir -p /tmp/wrf-massive

# Run (unbuffered). Fails if env.yaml changed since `submit` (checked via $WRF_MASSIVE_ENV_SHA256).
export RUNNER=srun
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
uv run wrf-massive run --stages="$STAGE" "$SIM_DIR"