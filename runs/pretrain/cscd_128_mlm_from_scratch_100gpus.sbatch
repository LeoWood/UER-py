#!/bin/bash
#SBATCH -p normal
#SBATCH -N 25
#SBATCH -J cscd1280
#SBATCH -o cscd_128_mlm_from_scratch_100gpus_68w_.out
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
child8=`sed -n '9p' ${hostfile}`
child9=`sed -n '10p' ${hostfile}`
child10=`sed -n '11p' ${hostfile}`
child11=`sed -n '12p' ${hostfile}`
child12=`sed -n '13p' ${hostfile}`
child13=`sed -n '14p' ${hostfile}`
child14=`sed -n '15p' ${hostfile}`
child15=`sed -n '16p' ${hostfile}`
child16=`sed -n '17p' ${hostfile}`
child17=`sed -n '18p' ${hostfile}`
child18=`sed -n '19p' ${hostfile}`
child19=`sed -n '20p' ${hostfile}`
child20=`sed -n '21p' ${hostfile}`
child21=`sed -n '22p' ${hostfile}`
child22=`sed -n '23p' ${hostfile}`
child23=`sed -n '24p' ${hostfile}`
child24=`sed -n '25p' ${hostfile}`



# echo ${hostname} ${child1} ${child2} ${child3} ${child4} ${child5} ${child6} ${child7} 

world_size=100

# start child processes
gpu1=4
gpu2=5
gpu3=6
gpu4=7
ssh ${child1} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=8
gpu2=9
gpu3=10
gpu4=11
ssh ${child2} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=12
gpu2=13
gpu3=14
gpu4=15
ssh ${child3} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=16
gpu2=17
gpu3=18
gpu4=19
ssh ${child4} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=20
gpu2=21
gpu3=22
gpu4=23
ssh ${child5} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=24
gpu2=25
gpu3=26
gpu4=27
ssh ${child6} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=28
gpu2=29
gpu3=30
gpu4=31
ssh ${child7} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=32
gpu2=33
gpu3=34
gpu4=35
ssh ${child8} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=36
gpu2=37
gpu3=38
gpu4=39
ssh ${child9} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=40
gpu2=41
gpu3=42
gpu4=43
ssh ${child10} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=44
gpu2=45
gpu3=46
gpu4=47
ssh ${child11} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=48
gpu2=49
gpu3=50
gpu4=51
ssh ${child12} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=52
gpu2=53
gpu3=54
gpu4=55
ssh ${child13} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=56
gpu2=57
gpu3=58
gpu4=59
ssh ${child14} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=60
gpu2=61
gpu3=62
gpu4=63
ssh ${child15} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=64
gpu2=65
gpu3=66
gpu4=67
ssh ${child16} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=68
gpu2=69
gpu3=70
gpu4=71
ssh ${child17} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=72
gpu2=73
gpu3=74
gpu4=75
ssh ${child18} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=76
gpu2=77
gpu3=78
gpu4=79
ssh ${child19} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=80
gpu2=81
gpu3=82
gpu4=83
ssh ${child20} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=84
gpu2=85
gpu3=86
gpu4=87
ssh ${child21} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=88
gpu2=89
gpu3=90
gpu4=91
ssh ${child22} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=92
gpu2=93
gpu3=94
gpu4=95
ssh ${child23} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=96
gpu2=97
gpu3=98
gpu4=99
ssh ${child24} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &


# start main process
gpu1=0
gpu2=1
gpu3=2
gpu4=3
ssh ${hostname} "bash `pwd`/cscd_128_mlm_from_scratch_100gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4"

echo END

