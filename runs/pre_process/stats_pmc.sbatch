#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -J stats_pmc
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o stats_pmc.out


export PYTHONUNBUFFERED=1

python ../../scripts/stats_instances.py \