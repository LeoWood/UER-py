#!/bin/bash
#SBATCH -p normal
#SBATCH -N 50
#SBATCH -J r_roberta
#SBATCH -o r_512_mlm_from_roberta_wwm_200gpus_hand.out
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
child25=`sed -n '26p' ${hostfile}`
child26=`sed -n '27p' ${hostfile}`
child27=`sed -n '28p' ${hostfile}`
child28=`sed -n '29p' ${hostfile}`
child29=`sed -n '30p' ${hostfile}`
child30=`sed -n '31p' ${hostfile}`
child31=`sed -n '32p' ${hostfile}`
child32=`sed -n '33p' ${hostfile}`
child33=`sed -n '34p' ${hostfile}`
child34=`sed -n '35p' ${hostfile}`
child35=`sed -n '36p' ${hostfile}`
child36=`sed -n '37p' ${hostfile}`
child37=`sed -n '38p' ${hostfile}`
child38=`sed -n '39p' ${hostfile}`
child39=`sed -n '40p' ${hostfile}`
child40=`sed -n '41p' ${hostfile}`
child41=`sed -n '42p' ${hostfile}`
child42=`sed -n '43p' ${hostfile}`
child43=`sed -n '44p' ${hostfile}`
child44=`sed -n '45p' ${hostfile}`
child45=`sed -n '46p' ${hostfile}`
child46=`sed -n '47p' ${hostfile}`
child47=`sed -n '48p' ${hostfile}`
child48=`sed -n '49p' ${hostfile}`
child49=`sed -n '50p' ${hostfile}`



# echo ${hostname} ${child1} ${child2} ${child3} ${child4} ${child5} ${child6} ${child7} 

world_size=200

# start child processes
gpu1=4
gpu2=5
gpu3=6
gpu4=7
ssh ${child1} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=8
gpu2=9
gpu3=10
gpu4=11
ssh ${child2} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=12
gpu2=13
gpu3=14
gpu4=15
ssh ${child3} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=16
gpu2=17
gpu3=18
gpu4=19
ssh ${child4} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=20
gpu2=21
gpu3=22
gpu4=23
ssh ${child5} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=24
gpu2=25
gpu3=26
gpu4=27
ssh ${child6} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=28
gpu2=29
gpu3=30
gpu4=31
ssh ${child7} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=32
gpu2=33
gpu3=34
gpu4=35
ssh ${child8} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=36
gpu2=37
gpu3=38
gpu4=39
ssh ${child9} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=40
gpu2=41
gpu3=42
gpu4=43
ssh ${child10} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=44
gpu2=45
gpu3=46
gpu4=47
ssh ${child11} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=48
gpu2=49
gpu3=50
gpu4=51
ssh ${child12} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=52
gpu2=53
gpu3=54
gpu4=55
ssh ${child13} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=56
gpu2=57
gpu3=58
gpu4=59
ssh ${child14} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=60
gpu2=61
gpu3=62
gpu4=63
ssh ${child15} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=64
gpu2=65
gpu3=66
gpu4=67
ssh ${child16} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=68
gpu2=69
gpu3=70
gpu4=71
ssh ${child17} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=72
gpu2=73
gpu3=74
gpu4=75
ssh ${child18} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=76
gpu2=77
gpu3=78
gpu4=79
ssh ${child19} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=80
gpu2=81
gpu3=82
gpu4=83
ssh ${child20} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=84
gpu2=85
gpu3=86
gpu4=87
ssh ${child21} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=88
gpu2=89
gpu3=90
gpu4=91
ssh ${child22} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=92
gpu2=93
gpu3=94
gpu4=95
ssh ${child23} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=96
gpu2=97
gpu3=98
gpu4=99
ssh ${child24} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=100
gpu2=101
gpu3=102
gpu4=103
ssh ${child25} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=104
gpu2=105
gpu3=106
gpu4=107
ssh ${child26} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=108
gpu2=109
gpu3=110
gpu4=111
ssh ${child27} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=112
gpu2=113
gpu3=114
gpu4=115
ssh ${child28} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=116
gpu2=117
gpu3=118
gpu4=119
ssh ${child29} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=120
gpu2=121
gpu3=122
gpu4=123
ssh ${child30} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=124
gpu2=125
gpu3=126
gpu4=127
ssh ${child31} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=128
gpu2=129
gpu3=130
gpu4=131
ssh ${child32} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=132
gpu2=133
gpu3=134
gpu4=135
ssh ${child33} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=136
gpu2=137
gpu3=138
gpu4=139
ssh ${child34} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=140
gpu2=141
gpu3=142
gpu4=143
ssh ${child35} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=144
gpu2=145
gpu3=146
gpu4=147
ssh ${child36} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=148
gpu2=149
gpu3=150
gpu4=151
ssh ${child37} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=152
gpu2=153
gpu3=154
gpu4=155
ssh ${child38} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=156
gpu2=157
gpu3=158
gpu4=159
ssh ${child39} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=160
gpu2=161
gpu3=162
gpu4=163
ssh ${child40} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=164
gpu2=165
gpu3=166
gpu4=167
ssh ${child41} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=168
gpu2=169
gpu3=170
gpu4=171
ssh ${child42} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=172
gpu2=173
gpu3=174
gpu4=175
ssh ${child43} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=176
gpu2=177
gpu3=178
gpu4=179
ssh ${child44} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=180
gpu2=181
gpu3=182
gpu4=183
ssh ${child45} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=184
gpu2=185
gpu3=186
gpu4=187
ssh ${child46} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=188
gpu2=189
gpu3=190
gpu4=191
ssh ${child47} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=192
gpu2=193
gpu3=194
gpu4=195
ssh ${child48} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=196
gpu2=197
gpu3=198
gpu4=199
ssh ${child49} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

# start main process
gpu1=0
gpu2=1
gpu3=2
gpu4=3
ssh ${hostname} "bash `pwd`/r_512_mlm_from_roberta_wwm_200gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4"

echo END

