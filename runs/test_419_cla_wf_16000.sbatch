#!/bin/bash
#SBATCH -p debug
#SBATCH -J test_cla
#SBATCH -N 1
#SBATCH -o test_cla.out

export PATH=/public/software/deeplearning/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH

export PRETRAINED_MODEL=google_zh_model.bin
export TUNE_MODEL=fine_tune_0.bin
export VOCAB=google_zh_vocab.txt
export EMBEDDING=bert
export LOG_FILE=test_419_cla_wf_16000.log
export GPU=0

echo "cla_16:" >> $LOG_FILE
/public/software/deeplearning/anaconda3/bin/python3 ../run_classifier_csci_emb.py \
--pretrained_model_path ../models/$PRETRAINED_MODEL \
--vocab_path ../models/$VOCAB \
--output_model_path ../output_tune/$TUNE_MODEL \
--config_path ../models/bert_base_config.json \
--train_path ../datasets/wanfang_16000/train.tsv \
--dev_path ../datasets/wanfang_16000/dev.tsv \
--test_path ../datasets/wanfang_16000/test.tsv \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank $GPU \
--add_pos 0 \
--add_term 0 \
--init_pos 0 \
--init_term 0