#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python3 run_ner.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/ner_amttl_bert_base.bin \
--train_path datasets/amttl/train.tsv \
--dev_path datasets/amttl/dev.tsv \
--test_path datasets/amttl/test.tsv \
--embedding bert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 3 \
--batch_size 32 \
--report_steps 50
