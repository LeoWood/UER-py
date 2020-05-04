#!/usr/bin/env bash
python preprocess.py \
--corpus_path corpora/book_review_bert.txt \
--vocab_path models/google_zh_vocab.txt \
--dataset_path dataset/book_review_bert.pt \
--processes_num 3 \
--target csci_mlm