#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export PRETRAINED_MODEL=for_test/cscd_R_based_on_google_zh_512_200w+best.bin
export TUNE_MODEL=fine_tune_0.bin
export VOCAB=google_zh_vocab.txt
export EMBEDDING=bert
export LOG_FILE=fine_tune_move_zh.log
export GPU=0

echo "512:" >> $LOG_FILE
python ../run_classifier_csci_emb.py \
--pretrained_model_path ../output_pre/$PRETRAINED_MODEL \
--vocab_path ../models/$VOCAB \
--output_model_path ../output_tune/$TUNE_MODEL \
--config_path ../models/bert_base_config.json \
--train_path ../datasets/move_zh/train.tsv \
--dev_path ../datasets/move_zh/dev.tsv \
--test_path ../datasets/move_zh/test.tsv \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 128 \
--batch_size 32 \
--report_steps 100 \
--gpu_rank $GPU \
--add_pos 0 \
--add_term 0 \
--init_pos 0 \
--init_term 0

export PYTHONUNBUFFERED=1
export PRETRAINED_MODEL=for_test/cscd_R_csci_mlm_based_on_google_zh_220w+best.bin
export EMBEDDING=cscibert

python ../run_classifier_csci_emb_old.py \
--pretrained_model_path ../output_pre/$PRETRAINED_MODEL \
--vocab_path ../models/$VOCAB \
--output_model_path ../models/fine_tune.bin \
--config_path ../models/bert_base_config.json \
--train_path ../datasets/move_zh/train.tsv \
--dev_path ../datasets/move_zh/dev.tsv \
--test_path ../datasets/move_zh/test.tsv \
--train_pt_path ../datasets/move_zh/train.pt \
--dev_pt_path ../datasets/move_zh/dev.pt \
--test_pt_path ../datasets/move_zh/test.pt \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 128 \
--batch_size 32 \
--report_steps 50 \
--gpu_rank $GPU \
--add_pos 1 \
--add_term 1 \
--init_pos 1 \
--init_term 1 \
--preprocess 1


echo "pos+term:" >> $LOG_FILE
python ../run_classifier_csci_emb_old.py \
--pretrained_model_path ../output_pre/$PRETRAINED_MODEL \
--vocab_path ../models/$VOCAB \
--output_model_path ../models/fine_tune.bin \
--config_path ../models/bert_base_config.json \
--train_path ../datasets/move_zh/train.tsv \
--dev_path ../datasets/move_zh/dev.tsv \
--test_path ../datasets/move_zh/test.tsv \
--train_pt_path ../datasets/move_zh/train.pt \
--dev_pt_path ../datasets/move_zh/dev.pt \
--test_pt_path ../datasets/move_zh/test.pt \
--log_path $LOG_FILE \
--embedding $EMBEDDING \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 128 \
--batch_size 32 \
--report_steps 50 \
--gpu_rank $GPU \
--add_pos 1 \
--add_term 1 \
--init_pos 1 \
--init_term 1 \
--preprocess 0