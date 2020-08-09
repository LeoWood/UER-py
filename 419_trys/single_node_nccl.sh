#!/bin/bash


module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1


DIST_URL=$1
WORLD_SIZE=$2
gpu1=$3
gpu2=$4
gpu3=$5
gpu4=$6


python /work1/zzx6320/lh/Projects/UER-py/pretrain.py \
--dataset_path /work1/zzx6320/lh/Projects/UER-py/corpora/cscd_r.pt \
--vocab_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_vocab.txt \
--pretrained_model_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_model.bin \
--output_model_path /work1/zzx6320/lh/Projects/UER-py/output_pre/pre_multi.bin  \
--output_log_path pre_multi.csv  \
--world_size $WORLD_SIZE \
--gpu_ranks $gpu1 $gpu2 $gpu3 $gpu4 \
--master_ip tcp://${DIST_URL}:34567 \
--report_steps 10 \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 32 \
--embedding bert \
--target mlm \
--backend nccl \
--add_pos 0 \
--add_term 0 \
