#!/bin/bash
#SBATCH -p debug
#SBATCH -J pretrain_single
#SBATCH -N 1
#SBATCH -o pretrain_single.out

export PYTHONUNBUFFERED=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0

hostname=`scontrol show hostnames $SLURM_JOB_NODELIST`
echo $hostname


/public/software/deeplearning/anaconda3/bin/python3 ../pretrain.py \
--dataset_path ../corpora/cscd_r.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../models/google_zh_model.bin \
--output_model_path ../output_pre/pretrain_r_128_mlm_bert_base_single.bin  \
--output_log_path pretrain_r_128_mlm_bert_base.csv  \
--world_size 4 \
--gpu_ranks 0 1 2 3 \
--master_ip "tcp://${hostname}:23456" \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 80 \
--embedding bert \
--target mlm \
--backend gloo \
--add_pos 0 \
--add_term 0 \

