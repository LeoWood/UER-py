#!/bin/bash
#SBATCH -p debug
#SBATCH -J pre_1_gpu
#SBATCH -N 1
#SBATCH -o pre_1_gpu.out

export PYTHONUNBUFFERED=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8
export HSA_USERPTR_FOR_PAGED_MEM=0

hostname=`scontrol show hostnames $SLURM_JOB_NODELIST`
echo $hostname



startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`
 
/public/software/deeplearning/anaconda3/bin/python3 ../pretrain_419.py \
--dataset_path ../corpora/cscd_r.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../models/google_zh_model.bin \
--output_model_path ../output_pre/pre_1_gpu.bin  \
--output_log_path pre_1_gpu.csv  \
--world_size 1 \
--gpu_ranks 0 \
--master_ip tcp://${hostname}:34567 \
--report_steps 10 \
--total_steps 4000 \
--save_checkpoint_steps 1000 \
--encoder bert \
--batch_size 80 \
--embedding bert \
--target mlm \
--backend gloo \
--add_pos 0 \
--add_term 0 \


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
endTime_s=`date +%s`
 
sumTime=$[ $endTime_s - $startTime_s ]
 
echo "$startTime ---> $endTime" "Total:$sumTime seconds" 