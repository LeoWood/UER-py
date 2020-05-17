#!/usr/bin/env bash
python run_ner_csci_emb.py \
--pretrained_model_path models/cscd_R_csci_mlm_no_pos_based_on_google_zh-best.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/ner_ccks_bert_base.bin \
--train_path datasets/ccks/train.tsv \
--dev_path datasets/ccks/dev.tsv \
--test_path datasets/ccks/test.tsv \
--log_path ./models/cla_cmedqa2_test.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 128 \
--batch_size 16 \
--report_steps 50 \
--gpu_rank 0 \
--add_pos 0 \
--add_term 1
