#!/usr/bin/env bash
python run_ner.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/ner_ccks_bert_base.bin \
--train_path datasets/ccsk/train.tsv \
--dev_path datasets/ccks/dev.tsv \
--test_path datasets/ccks/test.tsv \
--embedding bert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 8 \
--seq_length 128 \
--batch_size 32 \
--report_steps 50
