#!/usr/bin/env bash
#SBATCH -p normal
#SBATCH -J large
#SBATCH -N 1
#SBATCH -o fine_tune_large_sup.out

export PYTHONUNBUFFERED=1
export PATH=/public/software/deeplearning/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

export PRETRAINED_MODEL=mixed_large_24_model.bin
export VOCAB=google_zh_vocab.txt
export EMBEDDING=bert
export LOG_FILE=fine_tune_large_sup.log
export GPU=0


echo "amttl:" >> $LOG_FILE
python ../run_ner_csci_emb.py \
--pretrained_model_path ../models/$PRETRAINED_MODEL \
--vocab_path ../models/$VOCAB \
--output_model_path ../models/fine_tune_2.bin \
--config_path ../models/bert_large_config.json \
--train_path ../datasets/amttl/train.tsv \
--dev_path ../datasets/amttl/dev.tsv \
--test_path ../datasets/amttl/test.tsv \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 10 \
--seq_length 128 \
--batch_size 10 \
--report_steps 50 \
--gpu_rank $GPU \
--add_pos 0 \
--add_term 0 \
--init_pos 0 \
--init_term 0

echo "ccks:" >> $LOG_FILE
python ../run_ner_csci_emb.py \
--pretrained_model_path ../models/$PRETRAINED_MODEL \
--vocab_path ../models/$VOCAB \
--output_model_path ../models/fine_tune_2.bin \
--config_path ../models/bert_large_config.json \
--train_path ../datasets/ccks/train.tsv \
--dev_path ../datasets/ccks/dev.tsv \
--test_path ../datasets/ccks/test.tsv \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 10 \
--seq_length 128 \
--batch_size 10 \
--report_steps 50 \
--gpu_rank $GPU \
--add_pos 0 \
--add_term 0 \
--init_pos 0 \
--init_term 0