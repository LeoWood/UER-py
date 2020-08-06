#!/bin/bash
#SBATCH -p normal
#SBATCH -N 4
#SBATCH -J pre_fp16
#SBATCH -o wanfang_r_128_from_base_16gpus_fp16.out
#SBATCH --gres=dcu:4


hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}

hostname=`sed -n '1p' ${hostfile}`
child1=`sed -n '2p' ${hostfile}`
child2=`sed -n '3p' ${hostfile}`
child3=`sed -n '4p' ${hostfile}`

echo ${hostname} ${child1} ${child2} ${child3} 

world_size=16

# start child processes
gpu1=4
gpu2=5
gpu3=6
gpu4=7
ssh ${child1} "bash `pwd`/wanfang_r_128_from_base_16gpus_fp16.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=8
gpu2=9
gpu3=10
gpu4=11
ssh ${child2} "bash `pwd`/wanfang_r_128_from_base_16gpus_fp16.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=12
gpu2=13
gpu3=14
gpu4=15
ssh ${child3} "bash `pwd`/wanfang_r_128_from_base_16gpus_fp16.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

# start main process
gpu1=0
gpu2=1
gpu3=2
gpu4=3
ssh ${hostname} "bash `pwd`/wanfang_r_128_from_base_16gpus_fp16.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4"

echo END

