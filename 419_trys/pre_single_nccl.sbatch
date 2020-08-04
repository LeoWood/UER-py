#!/bin/bash
#SBATCH -p debug
#SBATCH -J pre_nccl
#SBATCH -N 1

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

hostname=`scontrol show hostnames $SLURM_JOB_NODELIST`
echo $hostname


python ../pretrain.py \
--dataset_path ../corpora/cscd_r.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../models/google_zh_model.bin \
--output_model_path ../output_pre/pretrain_r_128_mlm_bert_base_single.bin  \
--output_log_path pretrain_r_128_mlm_bert_base.csv  \
--world_size 4 \
--gpu_ranks 0 1 2 3 \
--master_ip tcp://${hostname}:34567 \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 72 \
--embedding bert \
--target mlm \
--backend nccl \
--add_pos 0 \
--add_term 0 \

