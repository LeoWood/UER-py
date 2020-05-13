#!/usr/bin/env bash
python preprocess.py \
--corpus_path corpora/R_test.txt \
--vocab_path models/google_zh_vocab.txt \
--dataset_path cscd_r_csci_mlm.pt \
--processes_num 1 \
--target csci_mlm \
--add_pos 1