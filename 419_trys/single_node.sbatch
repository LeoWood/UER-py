#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -J torch-1node-example
#SBATCH -n 32

export PYTHONUNBUFFERED=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
/public/software/deeplearning/anaconda3/bin/python3 /work1/zzx6320/lh/Projects/UER-py/pretrain.py \
--dataset_path /work1/zzx6320/lh/Projects/UER-py/corpora/cscd_r.pt \
--vocab_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_vocab.txt \
--pretrained_model_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_model.bin \
--output_model_path /work1/zzx6320/lh/Projects/UER-py/output_pre/pretrain_r_128_mlm_bert_base.bin  \
--output_log_path pretrain_r_128_mlm_bert_base.csv  \
--world_size 4 \
--gpu_ranks 0 1 2 3 \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 80 \
--embedding bert \
--target mlm \
--backend gloo \
--master_ip tcp://localhost:34567 \
--add_pos 0 \
--add_term 0 


echo END
