#!/bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J cla_base
#SBATCH -o cla_base.out
#SBATCH --gres=dcu:4

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

export PRETRAINED_MODEL=google_zh_model.bin
export TUNE_MODEL=cla_base.bin
export VOCAB=google_zh_vocab.txt
export EMBEDDING=cscibert
export LOG_FILE=cla_base.log
export GPU=0

python ../../run_classifier_csci_emb_old.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/wanfang_16000/train.tsv \
--dev_path ../../datasets/wanfang_16000/dev.tsv \
--test_path ../../datasets/wanfang_16000/test.tsv \
--train_pt_path ../../datasets/wanfang_16000/train.pt \
--dev_pt_path ../../datasets/wanfang_16000/dev.pt \
--test_pt_path ../../datasets/wanfang_16000/test.pt \
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
--init_term 0 \
--preprocess 1

echo "pre_processed cla16"

python ../../run_classifier_csci_emb_old.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/wanfang_32000/train.tsv \
--dev_path ../../datasets/wanfang_32000/dev.tsv \
--test_path ../../datasets/wanfang_32000/test.tsv \
--train_pt_path ../../datasets/wanfang_32000/train.pt \
--dev_pt_path ../../datasets/wanfang_32000/dev.pt \
--test_pt_path ../../datasets/wanfang_32000/test.pt \
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
--init_term 0 \
--preprocess 1

echo "pre_processed cla32"


echo "cla_16_base_pos:" >> $LOG_FILE
python ../../run_classifier_csci_emb_old.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/wanfang_16000/train.tsv \
--dev_path ../../datasets/wanfang_16000/dev.tsv \
--test_path ../../datasets/wanfang_16000/test.tsv \
--train_pt_path ../../datasets/wanfang_16000/train.pt \
--dev_pt_path ../../datasets/wanfang_16000/dev.pt \
--test_pt_path ../../datasets/wanfang_16000/test.pt \
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
--add_pos 1 \
--add_term 0 \
--init_pos 1 \
--init_term 0 \
--preprocess 0



echo "cla_32_base_pos:" >> $LOG_FILE
python ../../run_classifier_csci_emb_old.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/wanfang_32000/train.tsv \
--dev_path ../../datasets/wanfang_32000/dev.tsv \
--test_path ../../datasets/wanfang_32000/test.tsv \
--train_pt_path ../../datasets/wanfang_32000/train.pt \
--dev_pt_path ../../datasets/wanfang_32000/dev.pt \
--test_pt_path ../../datasets/wanfang_32000/test.pt \
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
--add_pos 1 \
--add_term 0 \
--init_pos 1 \
--init_term 0 \
--preprocess 0
