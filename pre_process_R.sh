#!/usr/bin/env bash
python preprocess.py \
--corpus_path corpora/R.txt \
--vocab_path models/google_zh_vocab.txt \
--dataset_path cscd_r.pt \
--processes_num 8 \
--target mlm