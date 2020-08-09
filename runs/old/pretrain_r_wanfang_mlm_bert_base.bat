python ../pretrain.py ^
--dataset_path ../wanfang_r_128.pt ^
--vocab_path ../models/google_zh_vocab.txt ^
--pretrained_model_path ../output_pre/pre_r_wanfang_from_bert_base-20w.bin ^
--output_model_path ../output_pre/pre_r_wanfang_from_bert_base_20w+.bin  ^
--output_log_path ../output_pre/pre_r_wanfang_from_bert_base_20w+.csv  ^
--world_size 1 ^
--gpu_ranks 0 ^
--total_steps 5000000 ^
--save_checkpoint_steps 10000 ^
--encoder bert ^
--batch_size 40 ^
--embedding bert ^
--target mlm ^
--add_pos 0 ^
--add_term 0