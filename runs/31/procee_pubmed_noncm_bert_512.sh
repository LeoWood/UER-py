python ../../preprocess.py \
--corpus_path ../../corpora/pubmed_oa_noncm.txt \
--vocab_path ../../models/google_uncased_en_vocab.txt \
--dataset_path ../../corpora/pubmed_oa_noncm_uncased.pt \
--seq_length 512 \
--processes_num 16 \
--target bert \
--stats_tokens