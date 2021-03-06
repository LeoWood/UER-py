#!/usr/bin/env bash
python run_classifier_csci_emb.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_wf_16_csci_emb_bert_base.bin \
--train_path datasets/wanfang_16000/train.tsv \
--dev_path datasets/wanfang_16000/dev.tsv \
--test_path datasets/wanfang_16000/test.tsv \
--log_path ./models/cla_wf_16_csci_emb_pos_bert_base.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank 0 \
--add_pos 1 \
--add_term 1
