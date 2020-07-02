git python ../pretrain.py ^
--dataset_path ../cscd_r_csci_mlm.pt ^
--vocab_path ../models/google_zh_vocab.txt ^
--pretrained_model_path ../output_pre/cscd_R_csci_mlm_based_on_google_zh_410w.bin ^
--output_model_path ../output_pre/cscd_R_csci_mlm_based_on_google_zh_410w+.bin  ^
--output_log_path ../output_pre/cscd_R_csci_mlm_410w+.csv  ^
--world_size 1 ^
--gpu_ranks 1 ^
--total_steps 5000000 ^
--save_checkpoint_steps 10000 ^
--encoder bert ^
--batch_size 40 ^
--embedding cscibert ^
--target csci_mlm ^
--add_pos 1 ^
--add_term 1
