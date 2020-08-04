#!/bin/bash
export PATH=/public/software/deeplearning/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
#export GLOO_SOCKET_IFNAME=ib0,ib1,ib2,ib3
export GLOO_SOCKET_IFNAME=ib0
export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0


/public/software/deeplearning/anaconda3/bin/python3 /work1/zzx6320/lh/Projects/UER-py/pretrain.py \
--dataset_path /work1/zzx6320/lh/Projects/UER-py/corpora/cscd_r.pt \
--vocab_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_vocab.txt \
--pretrained_model_path /work1/zzx6320/lh/Projects/UER-py/models/google_zh_model.bin \
--output_model_path /work1/zzx6320/lh/Projects/UER-py/output_pre/pretrain_r_128_mlm_bert_base.bin  \
--output_log_path pretrain_r_128_mlm_bert_base.csv  \
--world_size 2 \
--gpu_ranks 0 1 \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 80 \
--embedding bert \
--target mlm \
--backend gloo \
--add_pos 0 \
--add_term 0 
