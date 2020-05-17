#!/usr/bin/env bash
export PYTHONUNBUFFERED=1

python run_classifier_csci_emb.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_cmedqa_csl_pos_term_base.bin \
--train_path datasets/cmedqa/train.tsv \
--dev_path datasets/cmedqa/dev.tsv \
--test_path datasets/cmedqa/test.tsv \
--log_path ./models/cla_cmedqa_csl_pos_term_base.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank 1 \
--add_pos 1 \
--add_term 0

python run_classifier_csci_emb.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_cmedqa_csl_pos_term_base.bin \
--train_path datasets/cmedqa/train.tsv \
--dev_path datasets/cmedqa/dev.tsv \
--test_path datasets/cmedqa/test.tsv \
--log_path ./models/cla_cmedqa_csl_pos_term_base.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank 1 \
--add_pos 0 \
--add_term 1

python run_classifier_csci_emb.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_cmedqa_csl_pos_term_base.bin \
--train_path datasets/cmedqa/train.tsv \
--dev_path datasets/cmedqa/dev.tsv \
--test_path datasets/cmedqa/test.tsv \
--log_path ./models/cla_cmedqa_csl_pos_term_base.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 3 \
--report_steps 100 \
--gpu_rank 1 \
--add_pos 1 \
--add_term 1

python run_classifier_csci_emb.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_cmedqa_csl_pos_term_base.bin \
--train_path datasets/csl/train.tsv \
--dev_path datasets/csl/dev.tsv \
--test_path datasets/csl/test.tsv \
--log_path ./models/cla_cmedqa_csl_pos_term_base.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank 1 \
--add_pos 1 \
--add_term 0

python run_classifier_csci_emb.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_cmedqa_csl_pos_term_base.bin \
--train_path datasets/csl/train.tsv \
--dev_path datasets/csl/dev.tsv \
--test_path datasets/csl/test.tsv \
--log_path ./models/cla_cmedqa_csl_pos_term_base.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank 1 \
--add_pos 0 \
--add_term 1

python run_classifier_csci_emb.py \
--pretrained_model_path models/google_zh_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_cmedqa_csl_pos_term_base.bin \
--train_path datasets/csl/train.tsv \
--dev_path datasets/csl/dev.tsv \
--test_path datasets/csl/test.tsv \
--log_path ./models/cla_cmedqa_csl_pos_term_base.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 400 \
--batch_size 5 \
--report_steps 100 \
--gpu_rank 1 \
--add_pos 1 \
--add_term 1