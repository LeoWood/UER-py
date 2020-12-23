#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -J process
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o r_512_bert.out

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python ../../preprocess.py \
--corpus_path ../../corpora/R_512_bert.txt \
--vocab_path ../../models/google_zh_vocab.txt \
--dataset_path ../../corpora/r_512_bert.pt \
--seq_length 512 \
--processes_num 32 \
--target bert