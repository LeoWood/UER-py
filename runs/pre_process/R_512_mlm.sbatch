#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -J process
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o r_512_mlm.out


python ../../preprocess.py \
--corpus_path ../../corpora/R_512_mlm.txt \
--vocab_path ../../models/google_zh_vocab.txt \
--dataset_path ../../corpora/r_512_mlm.pt \
--seq_length 512 \
--processes_num 32 \
--target mlm