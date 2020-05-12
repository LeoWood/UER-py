#!/usr/bin/env bash
python preprocess.py \
--corpus_path corpora/R.txt \
--vocab_path models/google_zh_vocab.txt \
--dataset_path cscd_r_csci_mlm.pt \
--processes_num 4 \
--target csci_mlm \
--add_pos 1