export PYTHONUNBUFFERED=1
python ../pretrain_baidu.py \
--dataset_path ../cscd_r.pt \
--vocab_path ../models/google_zh_vocab.txt \
--pretrained_model_path ../../data/data38224/bert_wwm.bin \
--output_model_path ../output_pre/pre_cscd_r_bert_wwm.bin  \
--output_log_path ../output_pre/pre_cscd_r_bert_wwm+.csv  \
--world_size 1 \
--gpu_ranks 0 \
--total_steps 2000000 \
--save_checkpoint_steps 100000 \
--encoder bert \
--batch_size 192 \
--embedding bert \
--target mlm \
--add_pos 0 \
--add_term 0
