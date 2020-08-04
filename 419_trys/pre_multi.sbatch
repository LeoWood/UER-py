#!/bin/bash
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -J pre_multi



hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}

hostname=`sed -n '1p' ${hostfile}`
childname=`sed -n '2p' ${hostfile}`
echo ${hostname} ${childname}

world_size=8

gpu1=4
gpu2=5
gpu3=6
gpu4=7
# start child processes
ssh ${childname} "bash `pwd`/single_node.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=0
gpu2=1
gpu3=2
gpu4=3
# start main process
ssh ${hostname} "bash `pwd`/single_node.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4"

echo END

