# -*- encoding: utf-8 -*-

#File    :   create_sbatch.py
#Time    :   2020/09/24 16:40:41
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com

import sys
import os
os.chdir(sys.path[0])


nodes = 200
job = "r_roberta"
name = "r_512_mlm_from_roberta_wwm_" + str(4*nodes) + "gpus"

with open(name + ".sbatch",'w',encoding='utf-8') as f:
    f.write("#!/bin/bash")
    f.write("\n")
    f.write("#SBATCH -p normal")
    f.write("\n")
    f.write("#SBATCH -N " + str(nodes)) 
    f.write("\n")
    f.write("#SBATCH -J " + job)
    f.write("\n")
    f.write("#SBATCH -o " + name + ".out")
    f.write("\n")
    f.write("#SBATCH --gres=dcu:4")
    f.write("\n")
    f.write("\n")
    f.write("hostfile=./$SLURM_JOB_ID")
    f.write("\n")
    f.write("scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}")
    f.write("\n")
    f.write("\n")
    f.write(r"hostname=`sed -n '1p' ${hostfile}`")
    f.write("\n")
    for i in range(1,nodes):
        f.write("child" + str(i) + "=`sed -n '" + str(i+1) + r"p' ${hostfile}`")
        f.write("\n")
    f.write("\n")
    f.write("\n")
    f.write("world_size=" + str(nodes*4))
    f.write("\n")
    f.write("\n")
    f.write("# start child processes")
    f.write("\n")
    for i in range(1,nodes):
        f.write("gpu1=" + str(i*4))
        f.write("\n")
        f.write("gpu2=" + str(i*4+1))
        f.write("\n")
        f.write("gpu3=" + str(i*4+2))
        f.write("\n")
        f.write("gpu4=" + str(i*4+3))
        f.write("\n")
        f.write("ssh ${child" + str(i) + '} "bash `pwd`/' + name + r'.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4" &')
        f.write("\n")
        f.write("\n")
    f.write("# start main process")
    f.write("\n")
    f.write("gpu1=0")
    f.write("\n")
    f.write("gpu2=1")
    f.write("\n")
    f.write("gpu3=2")
    f.write("\n")
    f.write("gpu4=3")
    f.write("\n")
    f.write(r'ssh ${hostname} bash `pwd`/' + name + r'.sh ${hostname} $world_size $gpu1 $gpu2 $gpu3 $gpu4"')
    f.write("\n")
    f.write("\n")
    f.write("echo END")





if __name__ == '__main__':
    pass