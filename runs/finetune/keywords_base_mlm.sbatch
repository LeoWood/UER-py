#!/bin/bash

#SBATCH -p normal
#SBATCH -N 1
#SBATCH -J keywords
#SBATCH -o keywords_base_mlm.out
#SBATCH --gres=dcu:4

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

export PRETRAINED_MODEL=google_zh_model.bin
export TUNE_MODEL=keywords_base_mlm.bin
export VOCAB=google_zh_vocab.txt
export EMBEDDING=bert
export LOG_FILE=keywords_base_mlm.log
export GPU=0

echo "base:" >> $LOG_FILE
python ../../run_ner_csci_emb.py \
--pretrained_model_path ../../models/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/keywords_zh/train.tsv \
--dev_path ../../datasets/keywords_zh/dev.tsv \
--test_path ../../datasets/keywords_zh/test.tsv \
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
--add_term 0 \
--init_pos 0 \
--init_term 0 


export PRETRAINED_MODEL=cscd_512_mlm_from_base_100gpus_30w_best.bin
echo "cscd_512:" >> $LOG_FILE
python ../../run_ner_csci_emb.py \
--pretrained_model_path ../../output_pre/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/keywords_zh/train.tsv \
--dev_path ../../datasets/keywords_zh/dev.tsv \
--test_path ../../datasets/keywords_zh/test.tsv \
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
--add_term 0 \
--init_pos 0 \
--init_term 0 

export PRETRAINED_MODEL=r_512_mlm_from_base_100gpus_48w_best.bin
echo "r_512:" >> $LOG_FILE
python ../../run_ner_csci_emb.py \
--pretrained_model_path ../../output_pre/$PRETRAINED_MODEL \
--vocab_path ../../models/$VOCAB \
--output_model_path ../../output_tune/$TUNE_MODEL \
--config_path ../../models/bert_base_config.json \
--train_path ../../datasets/keywords_zh/train.tsv \
--dev_path ../../datasets/keywords_zh/dev.tsv \
--test_path ../../datasets/keywords_zh/test.tsv \
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
--add_term 0 \
--init_pos 0 \
--init_term 0 
