export PYTHONUNBUFFERED=1
python ../pretrain.py \
--dataset_path ../cscd_r.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../models/cscd_R_based_on_google_zh_600000_best.bin \
--output_model_path ../models/cscd_R_based_on_google_zh_60w+.bin  \
--output_log_path ../models/cscd_R_based_on_google_zh_60w+.csv  \
--world_size 2 \
--gpu_ranks 0 1 \
--total_steps 1000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 32 \
--embedding bert \
--target mlm \
--add_pos 0 \
--add_term 0
