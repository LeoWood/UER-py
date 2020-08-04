python ../preprocess.py ^
--corpus_path ../corpora/CSCD.txt ^
--vocab_path ../models/google_zh_vocab.txt ^
--dataset_path cscd.pt ^
--seq_length 128 ^
--processes_num 4 ^
--target mlm