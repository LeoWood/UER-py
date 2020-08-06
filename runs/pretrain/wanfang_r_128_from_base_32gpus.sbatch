#!/bin/bash
#SBATCH -p normal
#SBATCH -N 8
#SBATCH -J pre_32gpus
#SBATCH -o wanfang_r_128_from_base_32gpus.out
#SBATCH --gres=dcu:4


hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}

hostname=`sed -n '1p' ${hostfile}`
child1=`sed -n '2p' ${hostfile}`
child2=`sed -n '3p' ${hostfile}`
child3=`sed -n '4p' ${hostfile}`
child4=`sed -n '5p' ${hostfile}`
child5=`sed -n '6p' ${hostfile}`
child6=`sed -n '7p' ${hostfile}`
child7=`sed -n '8p' ${hostfile}`


echo ${hostname} ${child1} ${child2} ${child3} ${child4} ${child5} ${child6} ${child7} 

world_size=32

# start child processes
gpu1=4
gpu2=5
gpu3=6
gpu4=7
ssh ${child1} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=8
gpu2=9
gpu3=10
gpu4=11
ssh ${child2} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=12
gpu2=13
gpu3=14
gpu4=15
ssh ${child3} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=16
gpu2=17
gpu3=18
gpu4=19
ssh ${child4} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=20
gpu2=21
gpu3=22
gpu4=23
ssh ${child5} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=24
gpu2=25
gpu3=26
gpu4=27
ssh ${child6} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=28
gpu2=29
gpu3=30
gpu4=31
ssh ${child7} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

# start main process
gpu1=0
gpu2=1
gpu3=2
gpu4=3
ssh ${hostname} "bash `pwd`/wanfang_r_128_from_base_32gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4"

echo END

