export PYTHONUNBUFFERED=1
python ../pretrain.py \
--dataset_path ../wanfang_r_128.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../models/google_zh_model.bin \
--output_model_path ../output_pre/wanfang_r_128_mlm_bert_base.bin  \
--output_log_path ../output_pre/wanfang_r_128_mlm_bert_base.csv  \
--world_size 2 \
--gpu_ranks 0 1 \
--total_steps 1000000 \
--save_checkpoint_steps 10000 \
--encoder bert \
--batch_size 48 \
--embedding bert \
--target mlm \
--add_pos 0 \
--add_term 0
