#!/usr/bin/env bash
python ../preprocess.py \
--corpus_path ../corpora/R.txt \
--vocab_path ../models/google_zh_vocab.txt \
--dataset_path cscd_r_512.pt \
--seq_length 512 \
--processes_num 4 \
--target mlm