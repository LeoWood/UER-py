#!/bin/bash
export PYTHONUNBUFFERED=1

export PATH=/public/software/deeplearning/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export GLOO_SOCKET_IFNAME=ib0
export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0


DIST_URL=$1
WORLD_SIZE=$2
gpu1=$3
gpu2=$4
gpu3=$5
gpu4=$6


/public/software/deeplearning/anaconda3/bin/python3 /work1/zzx6320/lh/Projects/UER-py/pretrain_419.py \
--dataset_path /work1/zzx6320/lh/Projects/UER-py/corpora/cscd_r.pt \
--vocab_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_vocab.txt \
--pretrained_model_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_model.bin \
--output_model_path /work1/zzx6320/lh/Projects/UER-py/output_pre/pre_multi.bin  \
--output_log_path pre_multi.csv  \
--world_size $WORLD_SIZE \
--gpu_ranks $gpu1 $gpu2 $gpu3 $gpu4 \
--master_ip "tcp://${DIST_URL}i:34567" \
--report_steps 10 \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 80 \
--embedding bert \
--target mlm \
--backend gloo \
--add_pos 0 \
--add_term 0 \

