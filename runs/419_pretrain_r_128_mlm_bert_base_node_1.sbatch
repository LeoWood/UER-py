#!/bin/bash
#SBATCH -p normal
#SBATCH -J pre_r_128
#SBATCH -N 1
#SBATCH -o pretrain_r_128_mlm_bert_base_node_1.out
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:4

DIR=`pwd`
hostfile=${DIR}/tmp
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}

master_ip=`cat ${hostfile}` > ${DIR}/hostfile-tmp

echo $master_ip

export PYTHONUNBUFFERED=1
export PATH=/public/software/deeplearning/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export GLOO_SOCKET_IFNAME=ib0
export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0

python ../pretrain.py \
--dataset_path ../corpora/cscd_r.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../models/google_zh_model.bin \
--output_model_path ../output_pre/pretrain_r_128_mlm_bert_base_2_nodes.bin  \
--output_log_path pretrain_r_128_mlm_bert_base_2_nodes.csv  \
--world_size 8 \
--gpu_ranks 0 1 2 3 \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 80 \
--embedding bert \
--target mlm \
--backend gloo \
--master_ip tcp://$master_ip:34567 \
--add_pos 0 \
--add_term 0 \

