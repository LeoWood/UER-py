#!/bin/bash

#SBATCH -p normal
#SBATCH -N 1
#SBATCH -J keywords
#SBATCH -o keywords_base_pos_term_debug.out
#SBATCH --gres=dcu:4

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

export PRETRAINED_MODEL=google_zh_model.bin
export TUNE_MODEL=keywords_base_pos_term_debug.bin
export VOCAB=google_zh_vocab.txt
export EMBEDDING=cscibert
export LOG_FILE=keywords_base_pos_term_debug.log
export GPU=0


echo "base+pos:" >> $LOG_FILE
python ../../run_ner_csci_emb_old.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/keywords_zh/train.tsv \
--dev_path ../../datasets/keywords_zh/dev.tsv \
--test_path ../../datasets/keywords_zh/test.tsv \
--train_pt_path ../../datasets/keywords_zh/train.pt \
--dev_pt_path ../../datasets/keywords_zh/dev.pt \
--test_pt_path ../../datasets/keywords_zh/test.pt \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 512 \
--batch_size 8 \
--report_steps 100 \
--gpu_rank $GPU \
--add_pos 1 \
--add_term 0 \
--init_pos 1 \
--init_term 0 \
--preprocess 0 \
--terms_path ../../uer/utils/medical_terms/Med_Keywords.txt

echo "base+term:" >> $LOG_FILE
python ../../run_ner_csci_emb_old.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/keywords_zh/train.tsv \
--dev_path ../../datasets/keywords_zh/dev.tsv \
--test_path ../../datasets/keywords_zh/test.tsv \
--train_pt_path ../../datasets/keywords_zh/train_keywords.pt \
--dev_pt_path ../../datasets/keywords_zh/dev_keywords.pt \
--test_pt_path ../../datasets/keywords_zh/test_keywords.pt \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 512 \
--batch_size 8 \
--report_steps 100 \
--gpu_rank $GPU \
--add_pos 0 \
--add_term 1 \
--init_pos 0 \
--init_term 1 \
--preprocess 0 \
--terms_path ../../uer/utils/medical_terms/Med_Keywords.txt

