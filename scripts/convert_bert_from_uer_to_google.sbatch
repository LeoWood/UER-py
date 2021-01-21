#!/bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J convert
#SBATCH -o convert_uer_google.out
#SBATCH --gres=dcu:4

# module rm compiler/rocm/2.9
# module load compiler/rocm/3.3
# export MIOPEN_DEBUG_DISABLE_FIND_DB=1
# export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
# export PYTHONUNBUFFERED=1

export PYTHONUNBUFFERED=1
export PATH=/public/software/deeplearning/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export MIOPEN_DEBUG_DISABLE_FIND_DB=1


/public/software/deeplearning/anaconda3/bin/python3 convert_bert_from_uer_to_google.py \
--input_model_path ../output_pre/pubmed_bert_from_base_200gpus_best_0121.bin \
--output_model_path ../output_pre/pubmed_ckpts/pubmed_bert_from_base_200gpus_best_0121.ckpt