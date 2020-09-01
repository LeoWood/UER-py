#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J stats
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o r_stats_lines.out

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python ../../preprocess.py \
--corpus_path ../../corpora/R.txt \
--vocab_path ../../models/google_zh_vocab.txt \
--dataset_path ../../corpora/cscd_512_mlm.pt \
--stats_lines \
