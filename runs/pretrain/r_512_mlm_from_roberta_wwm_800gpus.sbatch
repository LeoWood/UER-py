#!/bin/bash
#SBATCH -p normal
#SBATCH -N 200
#SBATCH -J r_roberta
#SBATCH -o r_512_mlm_from_roberta_wwm_800gpus.out
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
child50=`sed -n '51p' ${hostfile}`
child51=`sed -n '52p' ${hostfile}`
child52=`sed -n '53p' ${hostfile}`
child53=`sed -n '54p' ${hostfile}`
child54=`sed -n '55p' ${hostfile}`
child55=`sed -n '56p' ${hostfile}`
child56=`sed -n '57p' ${hostfile}`
child57=`sed -n '58p' ${hostfile}`
child58=`sed -n '59p' ${hostfile}`
child59=`sed -n '60p' ${hostfile}`
child60=`sed -n '61p' ${hostfile}`
child61=`sed -n '62p' ${hostfile}`
child62=`sed -n '63p' ${hostfile}`
child63=`sed -n '64p' ${hostfile}`
child64=`sed -n '65p' ${hostfile}`
child65=`sed -n '66p' ${hostfile}`
child66=`sed -n '67p' ${hostfile}`
child67=`sed -n '68p' ${hostfile}`
child68=`sed -n '69p' ${hostfile}`
child69=`sed -n '70p' ${hostfile}`
child70=`sed -n '71p' ${hostfile}`
child71=`sed -n '72p' ${hostfile}`
child72=`sed -n '73p' ${hostfile}`
child73=`sed -n '74p' ${hostfile}`
child74=`sed -n '75p' ${hostfile}`
child75=`sed -n '76p' ${hostfile}`
child76=`sed -n '77p' ${hostfile}`
child77=`sed -n '78p' ${hostfile}`
child78=`sed -n '79p' ${hostfile}`
child79=`sed -n '80p' ${hostfile}`
child80=`sed -n '81p' ${hostfile}`
child81=`sed -n '82p' ${hostfile}`
child82=`sed -n '83p' ${hostfile}`
child83=`sed -n '84p' ${hostfile}`
child84=`sed -n '85p' ${hostfile}`
child85=`sed -n '86p' ${hostfile}`
child86=`sed -n '87p' ${hostfile}`
child87=`sed -n '88p' ${hostfile}`
child88=`sed -n '89p' ${hostfile}`
child89=`sed -n '90p' ${hostfile}`
child90=`sed -n '91p' ${hostfile}`
child91=`sed -n '92p' ${hostfile}`
child92=`sed -n '93p' ${hostfile}`
child93=`sed -n '94p' ${hostfile}`
child94=`sed -n '95p' ${hostfile}`
child95=`sed -n '96p' ${hostfile}`
child96=`sed -n '97p' ${hostfile}`
child97=`sed -n '98p' ${hostfile}`
child98=`sed -n '99p' ${hostfile}`
child99=`sed -n '100p' ${hostfile}`
child100=`sed -n '101p' ${hostfile}`
child101=`sed -n '102p' ${hostfile}`
child102=`sed -n '103p' ${hostfile}`
child103=`sed -n '104p' ${hostfile}`
child104=`sed -n '105p' ${hostfile}`
child105=`sed -n '106p' ${hostfile}`
child106=`sed -n '107p' ${hostfile}`
child107=`sed -n '108p' ${hostfile}`
child108=`sed -n '109p' ${hostfile}`
child109=`sed -n '110p' ${hostfile}`
child110=`sed -n '111p' ${hostfile}`
child111=`sed -n '112p' ${hostfile}`
child112=`sed -n '113p' ${hostfile}`
child113=`sed -n '114p' ${hostfile}`
child114=`sed -n '115p' ${hostfile}`
child115=`sed -n '116p' ${hostfile}`
child116=`sed -n '117p' ${hostfile}`
child117=`sed -n '118p' ${hostfile}`
child118=`sed -n '119p' ${hostfile}`
child119=`sed -n '120p' ${hostfile}`
child120=`sed -n '121p' ${hostfile}`
child121=`sed -n '122p' ${hostfile}`
child122=`sed -n '123p' ${hostfile}`
child123=`sed -n '124p' ${hostfile}`
child124=`sed -n '125p' ${hostfile}`
child125=`sed -n '126p' ${hostfile}`
child126=`sed -n '127p' ${hostfile}`
child127=`sed -n '128p' ${hostfile}`
child128=`sed -n '129p' ${hostfile}`
child129=`sed -n '130p' ${hostfile}`
child130=`sed -n '131p' ${hostfile}`
child131=`sed -n '132p' ${hostfile}`
child132=`sed -n '133p' ${hostfile}`
child133=`sed -n '134p' ${hostfile}`
child134=`sed -n '135p' ${hostfile}`
child135=`sed -n '136p' ${hostfile}`
child136=`sed -n '137p' ${hostfile}`
child137=`sed -n '138p' ${hostfile}`
child138=`sed -n '139p' ${hostfile}`
child139=`sed -n '140p' ${hostfile}`
child140=`sed -n '141p' ${hostfile}`
child141=`sed -n '142p' ${hostfile}`
child142=`sed -n '143p' ${hostfile}`
child143=`sed -n '144p' ${hostfile}`
child144=`sed -n '145p' ${hostfile}`
child145=`sed -n '146p' ${hostfile}`
child146=`sed -n '147p' ${hostfile}`
child147=`sed -n '148p' ${hostfile}`
child148=`sed -n '149p' ${hostfile}`
child149=`sed -n '150p' ${hostfile}`
child150=`sed -n '151p' ${hostfile}`
child151=`sed -n '152p' ${hostfile}`
child152=`sed -n '153p' ${hostfile}`
child153=`sed -n '154p' ${hostfile}`
child154=`sed -n '155p' ${hostfile}`
child155=`sed -n '156p' ${hostfile}`
child156=`sed -n '157p' ${hostfile}`
child157=`sed -n '158p' ${hostfile}`
child158=`sed -n '159p' ${hostfile}`
child159=`sed -n '160p' ${hostfile}`
child160=`sed -n '161p' ${hostfile}`
child161=`sed -n '162p' ${hostfile}`
child162=`sed -n '163p' ${hostfile}`
child163=`sed -n '164p' ${hostfile}`
child164=`sed -n '165p' ${hostfile}`
child165=`sed -n '166p' ${hostfile}`
child166=`sed -n '167p' ${hostfile}`
child167=`sed -n '168p' ${hostfile}`
child168=`sed -n '169p' ${hostfile}`
child169=`sed -n '170p' ${hostfile}`
child170=`sed -n '171p' ${hostfile}`
child171=`sed -n '172p' ${hostfile}`
child172=`sed -n '173p' ${hostfile}`
child173=`sed -n '174p' ${hostfile}`
child174=`sed -n '175p' ${hostfile}`
child175=`sed -n '176p' ${hostfile}`
child176=`sed -n '177p' ${hostfile}`
child177=`sed -n '178p' ${hostfile}`
child178=`sed -n '179p' ${hostfile}`
child179=`sed -n '180p' ${hostfile}`
child180=`sed -n '181p' ${hostfile}`
child181=`sed -n '182p' ${hostfile}`
child182=`sed -n '183p' ${hostfile}`
child183=`sed -n '184p' ${hostfile}`
child184=`sed -n '185p' ${hostfile}`
child185=`sed -n '186p' ${hostfile}`
child186=`sed -n '187p' ${hostfile}`
child187=`sed -n '188p' ${hostfile}`
child188=`sed -n '189p' ${hostfile}`
child189=`sed -n '190p' ${hostfile}`
child190=`sed -n '191p' ${hostfile}`
child191=`sed -n '192p' ${hostfile}`
child192=`sed -n '193p' ${hostfile}`
child193=`sed -n '194p' ${hostfile}`
child194=`sed -n '195p' ${hostfile}`
child195=`sed -n '196p' ${hostfile}`
child196=`sed -n '197p' ${hostfile}`
child197=`sed -n '198p' ${hostfile}`
child198=`sed -n '199p' ${hostfile}`
child199=`sed -n '200p' ${hostfile}`


world_size=800

# start child processes
gpu1=4
gpu2=5
gpu3=6
gpu4=7
ssh ${child1} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=8
gpu2=9
gpu3=10
gpu4=11
ssh ${child2} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=12
gpu2=13
gpu3=14
gpu4=15
ssh ${child3} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=16
gpu2=17
gpu3=18
gpu4=19
ssh ${child4} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=20
gpu2=21
gpu3=22
gpu4=23
ssh ${child5} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=24
gpu2=25
gpu3=26
gpu4=27
ssh ${child6} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=28
gpu2=29
gpu3=30
gpu4=31
ssh ${child7} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=32
gpu2=33
gpu3=34
gpu4=35
ssh ${child8} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=36
gpu2=37
gpu3=38
gpu4=39
ssh ${child9} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=40
gpu2=41
gpu3=42
gpu4=43
ssh ${child10} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=44
gpu2=45
gpu3=46
gpu4=47
ssh ${child11} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=48
gpu2=49
gpu3=50
gpu4=51
ssh ${child12} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=52
gpu2=53
gpu3=54
gpu4=55
ssh ${child13} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=56
gpu2=57
gpu3=58
gpu4=59
ssh ${child14} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=60
gpu2=61
gpu3=62
gpu4=63
ssh ${child15} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=64
gpu2=65
gpu3=66
gpu4=67
ssh ${child16} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=68
gpu2=69
gpu3=70
gpu4=71
ssh ${child17} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=72
gpu2=73
gpu3=74
gpu4=75
ssh ${child18} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=76
gpu2=77
gpu3=78
gpu4=79
ssh ${child19} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=80
gpu2=81
gpu3=82
gpu4=83
ssh ${child20} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=84
gpu2=85
gpu3=86
gpu4=87
ssh ${child21} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=88
gpu2=89
gpu3=90
gpu4=91
ssh ${child22} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=92
gpu2=93
gpu3=94
gpu4=95
ssh ${child23} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=96
gpu2=97
gpu3=98
gpu4=99
ssh ${child24} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=100
gpu2=101
gpu3=102
gpu4=103
ssh ${child25} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=104
gpu2=105
gpu3=106
gpu4=107
ssh ${child26} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=108
gpu2=109
gpu3=110
gpu4=111
ssh ${child27} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=112
gpu2=113
gpu3=114
gpu4=115
ssh ${child28} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=116
gpu2=117
gpu3=118
gpu4=119
ssh ${child29} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=120
gpu2=121
gpu3=122
gpu4=123
ssh ${child30} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=124
gpu2=125
gpu3=126
gpu4=127
ssh ${child31} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=128
gpu2=129
gpu3=130
gpu4=131
ssh ${child32} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=132
gpu2=133
gpu3=134
gpu4=135
ssh ${child33} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=136
gpu2=137
gpu3=138
gpu4=139
ssh ${child34} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=140
gpu2=141
gpu3=142
gpu4=143
ssh ${child35} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=144
gpu2=145
gpu3=146
gpu4=147
ssh ${child36} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=148
gpu2=149
gpu3=150
gpu4=151
ssh ${child37} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=152
gpu2=153
gpu3=154
gpu4=155
ssh ${child38} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=156
gpu2=157
gpu3=158
gpu4=159
ssh ${child39} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=160
gpu2=161
gpu3=162
gpu4=163
ssh ${child40} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=164
gpu2=165
gpu3=166
gpu4=167
ssh ${child41} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=168
gpu2=169
gpu3=170
gpu4=171
ssh ${child42} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=172
gpu2=173
gpu3=174
gpu4=175
ssh ${child43} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=176
gpu2=177
gpu3=178
gpu4=179
ssh ${child44} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=180
gpu2=181
gpu3=182
gpu4=183
ssh ${child45} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=184
gpu2=185
gpu3=186
gpu4=187
ssh ${child46} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=188
gpu2=189
gpu3=190
gpu4=191
ssh ${child47} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=192
gpu2=193
gpu3=194
gpu4=195
ssh ${child48} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=196
gpu2=197
gpu3=198
gpu4=199
ssh ${child49} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=200
gpu2=201
gpu3=202
gpu4=203
ssh ${child50} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=204
gpu2=205
gpu3=206
gpu4=207
ssh ${child51} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=208
gpu2=209
gpu3=210
gpu4=211
ssh ${child52} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=212
gpu2=213
gpu3=214
gpu4=215
ssh ${child53} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=216
gpu2=217
gpu3=218
gpu4=219
ssh ${child54} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=220
gpu2=221
gpu3=222
gpu4=223
ssh ${child55} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=224
gpu2=225
gpu3=226
gpu4=227
ssh ${child56} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=228
gpu2=229
gpu3=230
gpu4=231
ssh ${child57} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=232
gpu2=233
gpu3=234
gpu4=235
ssh ${child58} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=236
gpu2=237
gpu3=238
gpu4=239
ssh ${child59} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=240
gpu2=241
gpu3=242
gpu4=243
ssh ${child60} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=244
gpu2=245
gpu3=246
gpu4=247
ssh ${child61} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=248
gpu2=249
gpu3=250
gpu4=251
ssh ${child62} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=252
gpu2=253
gpu3=254
gpu4=255
ssh ${child63} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=256
gpu2=257
gpu3=258
gpu4=259
ssh ${child64} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=260
gpu2=261
gpu3=262
gpu4=263
ssh ${child65} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=264
gpu2=265
gpu3=266
gpu4=267
ssh ${child66} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=268
gpu2=269
gpu3=270
gpu4=271
ssh ${child67} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=272
gpu2=273
gpu3=274
gpu4=275
ssh ${child68} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=276
gpu2=277
gpu3=278
gpu4=279
ssh ${child69} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=280
gpu2=281
gpu3=282
gpu4=283
ssh ${child70} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=284
gpu2=285
gpu3=286
gpu4=287
ssh ${child71} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=288
gpu2=289
gpu3=290
gpu4=291
ssh ${child72} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=292
gpu2=293
gpu3=294
gpu4=295
ssh ${child73} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=296
gpu2=297
gpu3=298
gpu4=299
ssh ${child74} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=300
gpu2=301
gpu3=302
gpu4=303
ssh ${child75} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=304
gpu2=305
gpu3=306
gpu4=307
ssh ${child76} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=308
gpu2=309
gpu3=310
gpu4=311
ssh ${child77} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=312
gpu2=313
gpu3=314
gpu4=315
ssh ${child78} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=316
gpu2=317
gpu3=318
gpu4=319
ssh ${child79} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=320
gpu2=321
gpu3=322
gpu4=323
ssh ${child80} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=324
gpu2=325
gpu3=326
gpu4=327
ssh ${child81} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=328
gpu2=329
gpu3=330
gpu4=331
ssh ${child82} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=332
gpu2=333
gpu3=334
gpu4=335
ssh ${child83} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=336
gpu2=337
gpu3=338
gpu4=339
ssh ${child84} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=340
gpu2=341
gpu3=342
gpu4=343
ssh ${child85} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=344
gpu2=345
gpu3=346
gpu4=347
ssh ${child86} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=348
gpu2=349
gpu3=350
gpu4=351
ssh ${child87} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=352
gpu2=353
gpu3=354
gpu4=355
ssh ${child88} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=356
gpu2=357
gpu3=358
gpu4=359
ssh ${child89} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=360
gpu2=361
gpu3=362
gpu4=363
ssh ${child90} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=364
gpu2=365
gpu3=366
gpu4=367
ssh ${child91} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=368
gpu2=369
gpu3=370
gpu4=371
ssh ${child92} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=372
gpu2=373
gpu3=374
gpu4=375
ssh ${child93} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=376
gpu2=377
gpu3=378
gpu4=379
ssh ${child94} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=380
gpu2=381
gpu3=382
gpu4=383
ssh ${child95} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=384
gpu2=385
gpu3=386
gpu4=387
ssh ${child96} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=388
gpu2=389
gpu3=390
gpu4=391
ssh ${child97} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=392
gpu2=393
gpu3=394
gpu4=395
ssh ${child98} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=396
gpu2=397
gpu3=398
gpu4=399
ssh ${child99} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=400
gpu2=401
gpu3=402
gpu4=403
ssh ${child100} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=404
gpu2=405
gpu3=406
gpu4=407
ssh ${child101} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=408
gpu2=409
gpu3=410
gpu4=411
ssh ${child102} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=412
gpu2=413
gpu3=414
gpu4=415
ssh ${child103} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=416
gpu2=417
gpu3=418
gpu4=419
ssh ${child104} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=420
gpu2=421
gpu3=422
gpu4=423
ssh ${child105} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=424
gpu2=425
gpu3=426
gpu4=427
ssh ${child106} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=428
gpu2=429
gpu3=430
gpu4=431
ssh ${child107} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=432
gpu2=433
gpu3=434
gpu4=435
ssh ${child108} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=436
gpu2=437
gpu3=438
gpu4=439
ssh ${child109} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=440
gpu2=441
gpu3=442
gpu4=443
ssh ${child110} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=444
gpu2=445
gpu3=446
gpu4=447
ssh ${child111} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=448
gpu2=449
gpu3=450
gpu4=451
ssh ${child112} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=452
gpu2=453
gpu3=454
gpu4=455
ssh ${child113} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=456
gpu2=457
gpu3=458
gpu4=459
ssh ${child114} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=460
gpu2=461
gpu3=462
gpu4=463
ssh ${child115} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=464
gpu2=465
gpu3=466
gpu4=467
ssh ${child116} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=468
gpu2=469
gpu3=470
gpu4=471
ssh ${child117} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=472
gpu2=473
gpu3=474
gpu4=475
ssh ${child118} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=476
gpu2=477
gpu3=478
gpu4=479
ssh ${child119} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=480
gpu2=481
gpu3=482
gpu4=483
ssh ${child120} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=484
gpu2=485
gpu3=486
gpu4=487
ssh ${child121} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=488
gpu2=489
gpu3=490
gpu4=491
ssh ${child122} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=492
gpu2=493
gpu3=494
gpu4=495
ssh ${child123} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=496
gpu2=497
gpu3=498
gpu4=499
ssh ${child124} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=500
gpu2=501
gpu3=502
gpu4=503
ssh ${child125} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=504
gpu2=505
gpu3=506
gpu4=507
ssh ${child126} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=508
gpu2=509
gpu3=510
gpu4=511
ssh ${child127} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=512
gpu2=513
gpu3=514
gpu4=515
ssh ${child128} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=516
gpu2=517
gpu3=518
gpu4=519
ssh ${child129} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=520
gpu2=521
gpu3=522
gpu4=523
ssh ${child130} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=524
gpu2=525
gpu3=526
gpu4=527
ssh ${child131} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=528
gpu2=529
gpu3=530
gpu4=531
ssh ${child132} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=532
gpu2=533
gpu3=534
gpu4=535
ssh ${child133} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=536
gpu2=537
gpu3=538
gpu4=539
ssh ${child134} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=540
gpu2=541
gpu3=542
gpu4=543
ssh ${child135} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=544
gpu2=545
gpu3=546
gpu4=547
ssh ${child136} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=548
gpu2=549
gpu3=550
gpu4=551
ssh ${child137} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=552
gpu2=553
gpu3=554
gpu4=555
ssh ${child138} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=556
gpu2=557
gpu3=558
gpu4=559
ssh ${child139} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=560
gpu2=561
gpu3=562
gpu4=563
ssh ${child140} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=564
gpu2=565
gpu3=566
gpu4=567
ssh ${child141} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=568
gpu2=569
gpu3=570
gpu4=571
ssh ${child142} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=572
gpu2=573
gpu3=574
gpu4=575
ssh ${child143} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=576
gpu2=577
gpu3=578
gpu4=579
ssh ${child144} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=580
gpu2=581
gpu3=582
gpu4=583
ssh ${child145} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=584
gpu2=585
gpu3=586
gpu4=587
ssh ${child146} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=588
gpu2=589
gpu3=590
gpu4=591
ssh ${child147} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=592
gpu2=593
gpu3=594
gpu4=595
ssh ${child148} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=596
gpu2=597
gpu3=598
gpu4=599
ssh ${child149} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=600
gpu2=601
gpu3=602
gpu4=603
ssh ${child150} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=604
gpu2=605
gpu3=606
gpu4=607
ssh ${child151} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=608
gpu2=609
gpu3=610
gpu4=611
ssh ${child152} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=612
gpu2=613
gpu3=614
gpu4=615
ssh ${child153} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=616
gpu2=617
gpu3=618
gpu4=619
ssh ${child154} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=620
gpu2=621
gpu3=622
gpu4=623
ssh ${child155} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=624
gpu2=625
gpu3=626
gpu4=627
ssh ${child156} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=628
gpu2=629
gpu3=630
gpu4=631
ssh ${child157} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=632
gpu2=633
gpu3=634
gpu4=635
ssh ${child158} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=636
gpu2=637
gpu3=638
gpu4=639
ssh ${child159} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=640
gpu2=641
gpu3=642
gpu4=643
ssh ${child160} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=644
gpu2=645
gpu3=646
gpu4=647
ssh ${child161} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=648
gpu2=649
gpu3=650
gpu4=651
ssh ${child162} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=652
gpu2=653
gpu3=654
gpu4=655
ssh ${child163} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=656
gpu2=657
gpu3=658
gpu4=659
ssh ${child164} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=660
gpu2=661
gpu3=662
gpu4=663
ssh ${child165} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=664
gpu2=665
gpu3=666
gpu4=667
ssh ${child166} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=668
gpu2=669
gpu3=670
gpu4=671
ssh ${child167} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=672
gpu2=673
gpu3=674
gpu4=675
ssh ${child168} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=676
gpu2=677
gpu3=678
gpu4=679
ssh ${child169} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=680
gpu2=681
gpu3=682
gpu4=683
ssh ${child170} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=684
gpu2=685
gpu3=686
gpu4=687
ssh ${child171} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=688
gpu2=689
gpu3=690
gpu4=691
ssh ${child172} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=692
gpu2=693
gpu3=694
gpu4=695
ssh ${child173} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=696
gpu2=697
gpu3=698
gpu4=699
ssh ${child174} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=700
gpu2=701
gpu3=702
gpu4=703
ssh ${child175} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=704
gpu2=705
gpu3=706
gpu4=707
ssh ${child176} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=708
gpu2=709
gpu3=710
gpu4=711
ssh ${child177} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=712
gpu2=713
gpu3=714
gpu4=715
ssh ${child178} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=716
gpu2=717
gpu3=718
gpu4=719
ssh ${child179} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=720
gpu2=721
gpu3=722
gpu4=723
ssh ${child180} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=724
gpu2=725
gpu3=726
gpu4=727
ssh ${child181} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=728
gpu2=729
gpu3=730
gpu4=731
ssh ${child182} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=732
gpu2=733
gpu3=734
gpu4=735
ssh ${child183} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=736
gpu2=737
gpu3=738
gpu4=739
ssh ${child184} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=740
gpu2=741
gpu3=742
gpu4=743
ssh ${child185} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=744
gpu2=745
gpu3=746
gpu4=747
ssh ${child186} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=748
gpu2=749
gpu3=750
gpu4=751
ssh ${child187} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=752
gpu2=753
gpu3=754
gpu4=755
ssh ${child188} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=756
gpu2=757
gpu3=758
gpu4=759
ssh ${child189} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=760
gpu2=761
gpu3=762
gpu4=763
ssh ${child190} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=764
gpu2=765
gpu3=766
gpu4=767
ssh ${child191} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=768
gpu2=769
gpu3=770
gpu4=771
ssh ${child192} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=772
gpu2=773
gpu3=774
gpu4=775
ssh ${child193} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=776
gpu2=777
gpu3=778
gpu4=779
ssh ${child194} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=780
gpu2=781
gpu3=782
gpu4=783
ssh ${child195} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=784
gpu2=785
gpu3=786
gpu4=787
ssh ${child196} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=788
gpu2=789
gpu3=790
gpu4=791
ssh ${child197} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=792
gpu2=793
gpu3=794
gpu4=795
ssh ${child198} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

gpu1=796
gpu2=797
gpu3=798
gpu4=799
ssh ${child199} "bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &

# start main process
gpu1=0
gpu2=1
gpu3=2
gpu4=3
ssh ${hostname} bash `pwd`/r_512_mlm_from_roberta_wwm_800gpus.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4"

echo END