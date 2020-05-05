#!/usr/bin/env bash
python pretrain.py \
--dataset_path cscd_r.pt \
--vocab_path models/google_zh_vocab.txt \
--pretrained_model_path models/cscd_R_based_on_google_zh-400000.bin \
--output_model_path models/cscd_R_based_on_google_zh_400000+.bin  \
--output_log_path models/cscd_R_based_on_google_zh_400000-500000.csv  \
--world_size 2 \
--gpu_ranks 0 1 \
--total_steps 100000 \
--save_checkpoint_steps 10000 \
--encoder bert \
--target mlm