#!/bin/bash
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J cscd_bert
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o create_cscd_512_bert_from_abs.out

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

python ../../work/CSCD_R_512/create_bert_data_from_abs.py \