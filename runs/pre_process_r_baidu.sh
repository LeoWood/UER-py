#!/usr/bin/env bash
python ../preprocess_baidu.py \
--corpus_path ../../data/data38224/R.txt \
--vocab_path ../models/google_zh_vocab.txt \
--dataset_path ../cscd_r.pt \
--seq_length 128 \
--processes_num 2 \
--target mlm