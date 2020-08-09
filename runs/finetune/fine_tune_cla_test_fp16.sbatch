#!/bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J tune_fp16
#SBATCH -o fine_tune_cla_test_fp16.out
#SBATCH --gres=dcu:4

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

export PRETRAINED_MODEL=google_zh_model.bin
export TUNE_MODEL=fine_tune_0.bin
export VOCAB=google_zh_vocab.txt
export EMBEDDING=bert
export LOG_FILE=fine_tune_base_1.log
export GPU=0

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

echo "cla_16:" >> $LOG_FILE
python ../../run_classifier_csci_emb.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/wanfang_16000/train.tsv \
--dev_path ../../datasets/wanfang_16000/dev.tsv \
--test_path ../../datasets/wanfang_16000/test.tsv \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 3 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank $GPU \
--fp16 \
--add_pos 0 \
--add_term 0 \
--init_pos 0 \
--init_term 0

endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
endTime_s=`date +%s`
 
sumTime=$[ $endTime_s - $startTime_s ]
 
echo "$startTime ---> $endTime" "Total:$sumTime seconds" 