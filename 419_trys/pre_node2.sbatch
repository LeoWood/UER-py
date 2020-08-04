#!/bin/bash
#SBATCH -p debug
#SBATCH -J pret_node_2
#SBATCH -N 1

export PYTHONUNBUFFERED=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0

export GLOO_SOCKET_IFNAME=ib0

master_ip=`cat tmp`

echo $master_ip


/public/software/deeplearning/anaconda3/bin/python3 ../pretrain_419.py \
--dataset_path ../corpora/cscd_r.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../models/google_zh_model.bin \
--output_model_path ../output_pre/pre_2_nodes.bin  \
--output_log_path pre_node_2.csv  \
--world_size 8 \
--gpu_ranks 4 5 6 7 \
--master_ip "tcp://${master_ip}:34567" \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 80 \
--embedding bert \
--target mlm \
--backend gloo \
--add_pos 0 \
--add_term 0 \

