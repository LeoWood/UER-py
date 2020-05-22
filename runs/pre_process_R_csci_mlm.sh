#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
python ../preprocess.py \
--corpus_path corpora/R.txt \
--vocab_path models/google_zh_vocab.txt \
--dataset_path cscd_r_pos_term_512.pt \
--seq_length 512 \
--processes_num 4 \
--target csci_mlm 