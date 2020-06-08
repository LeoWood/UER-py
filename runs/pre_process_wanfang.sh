#!/usr/bin/env bash
python ../preprocess.py \
--corpus_path ../corpora/pre_training_R_all.txt \
--vocab_path ../models/google_zh_vocab.txt \
--dataset_path ../wanfang_r_128.pt \
--seq_length 128 \
--processes_num 4 \
--target mlm