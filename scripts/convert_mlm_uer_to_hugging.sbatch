#!/bin/bash

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -J convert
#SBATCH -o convert_r_128_512.out
#SBATCH --gres=dcu:4

module rm compiler/rocm/2.9
module load compiler/rocm/3.3
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export LD_LIBRARY_PATH=/public/software/deeplearning/anaconda3/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1


python convert_mlm_from_uer_to_huggingface.py \
--input_model_path ../output_pre/r_128_512_mlm_from_base_100gpus_20w_best.bin \
--bert_model_path ../models/google_zh_model.bin \
--output_model_path ../output_pre/hugging_r_128_512_mlm_from_base_100gpus_20w_best.bin
 ， 。 。