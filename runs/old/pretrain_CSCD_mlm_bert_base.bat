python ../pretrain.py ^
--dataset_path ../cscd.pt ^
--vocab_path ../models/google_zh_vocab.txt ^
--pretrained_model_path ../output_pre/pre_cscd_from_bert_base-20w.bin ^
--output_model_path ../output_pre/pre_cscd_from_bert_base_20w+.bin  ^
--output_log_path ../output_pre/pre_cscd_from_bert_base_20w+.csv  ^
--world_size 1 ^
--gpu_ranks 1 ^
--total_steps 5000000 ^
--save_checkpoint_steps 50000 ^
--encoder bert ^
--batch_size 45 ^
--embedding bert ^
--target mlm ^
--add_pos 0 ^
--add_term 0
