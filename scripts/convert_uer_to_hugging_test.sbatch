#!/bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J convert
#SBATCH -o convert_bert_base.out
#SBATCH --gres=dcu:4

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1


python convert_bert_from_uer_to_huggingface.py \
--input_model_path ../models/google_zh_model.bin \
--output_model_path ../models/hugging_google_zh_model.bin
