#!/usr/bin/env bash
python run_ner_csci_emb.py \
--pretrained_model_path models/bert-base-chinese-pytorch_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/ner_ccks_bert_wwm.bin \
--train_path datasets/ccks/train.tsv \
--dev_path datasets/ccks/dev.tsv \
--test_path datasets/ccks/test.tsv \
--log_path ./models/ner_ccsk_pos_bert_wwm.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 128 \
--batch_size 16 \
--report_steps 50 \
--gpu_rank 0 \
--add_pos 1 \
--add_term 0

python run_ner_csci_emb.py \
--pretrained_model_path models/bert-base-chinese-pytorch_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/ner_ccks_bert_wwm.bin \
--train_path datasets/ccks/train.tsv \
--dev_path datasets/ccks/dev.tsv \
--test_path datasets/ccks/test.tsv \
--log_path ./models/ner_ccsk_term_bert_wwm.log \
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

python run_ner_csci_emb.py \
--pretrained_model_path models/bert-base-chinese-pytorch_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/ner_ccks_bert_wwm.bin \
--train_path datasets/ccks/train.tsv \
--dev_path datasets/ccks/dev.tsv \
--test_path datasets/ccks/test.tsv \
--log_path ./models/ner_ccsk_pos_term_bert_wwm.log \
--embedding cscibert \
--encoder bert \
--learning_rate 2e-5 \
--warmup 0.1 \
--epochs_num 5 \
--seq_length 128 \
--batch_size 16 \
--report_steps 50 \
--gpu_rank 0 \
--add_pos 1 \
--add_term 1
