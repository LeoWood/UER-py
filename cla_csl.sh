#!/usr/bin/env bash
python run_classifier.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_csl_bert_base.bin \
--train_path datasets/cls/train.tsv \
--dev_path datasets/cls/dev.tsv \
--test_path datasets/cls/test.tsv \
--embedding bert \
--encoder bert \
--learning_rate 1e-5 \
--warmup 0.1 \
--epochs_num 8 \
--seq_length 256 \
--batch_size 10 \
--report_steps 100 \
--gpu_rank 0