#!/bin/bash
#SBATCH -p debug
#SBATCH -J test_pre
#SBATCH -N 1
#SBATCH -o test_4_gpus.out


hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}

hostname=`sed -n '1p' ${hostfile}`

echo ${hostname}

# start main process
ssh ${hostname} "bash `pwd`/single_process.sh"

echo END
# sleep 600
