#!/bin/bash

DIST_URL=159.226.102.34
WORLD_SIZE=3

python ../../pretrain.py \
--dataset_path ../../corpora/pubmed_oa_noncm.pt \
--vocab_path ../../models/google_uncased_en_vocab.txt \
--pretrained_model_path ../../models/google_uncased_en.bin \
--output_model_path ../../output_pre/pretrain_r_512_bert_from_base_3gpus_uncased_34.bin  \
--output_log_path ../../output_pre/pretrain_r_512_bert_from_base_3gpus_uncased_34.csv  \
--world_size $WORLD_SIZE \
--gpu_ranks 0 1 2 \
--master_ip tcp://${DIST_URL}:34567 \
--report_steps 100 \
--total_steps 5000000 \
--save_checkpoint_steps 10000 \
--encoder bert \
--batch_size 20 \
--embedding bert \
--target bert \
--backend nccl \
--add_pos 0 \
--add_term 0 \

