#!/usr/bin/env bash
python convert_bert_from_huggingface_to_uer.py \
--input_model_path models/chinese_wwm_ext_L-12_H-768_A-12/bert_model.ckpt \
--output_model_path models/bert_wwm_ext_model.bin
