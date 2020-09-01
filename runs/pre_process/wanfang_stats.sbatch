#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J stats
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o wanfang_r_stats.out

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python ../../preprocess.py \
--corpus_path ../../corpora/wanfang.txt \
--vocab_path ../../models/google_zh_vocab.txt \
--dataset_path ../../corpora/wanfang_r_128.pt \
--stats_tokens \
--stats_lines \
