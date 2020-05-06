#!/usr/bin/env bash
python pretrain.py \
--dataset_path cscd_r.pt \
--vocab_path models/google_zh_vocab.txt \
--pretrained_model_path models/google_zh_model.bin \
--output_model_path models/cscd_R_csci_mlm_no_pos_based_on_google_zh.bin  \
--output_log_path models/cscd_R_csci_mlm_no_pos.csv  \
--world_size 2 \
--gpu_ranks 0 1 \
--total_steps 500000 \
--save_checkpoint_steps 10000 \
--encoder bert \
--target csci_mlm \
--add_pos 0
