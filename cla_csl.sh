#!/usr/bin/env bash
python3 run_classifier.py \
--pretrained_model_path models/book_review_mlm_model.bin \
--vocab_path models/google_zh_vocab.txt \
--output_model_path ./models/cla_csl_bert_base.bin \
--train_path datasets/cls/train.tsv \
--dev_path datasets/cls/dev.tsv \
--test_path datasets/cls/test.tsv \
--epochs_num 3 \
--batch_size 32 \
--encoder bert