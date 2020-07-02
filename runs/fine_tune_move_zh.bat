set PRETRAINED_MODEL=google_zh_model.bin
set TUNE_MODEL=fine_tune_move_zh_bert_base.bin
set VOCAB=google_zh_vocab.txt
set EMBEDDING=bert
set LOG_FILE=fine_tune_move_zh_bert_base.log
set GPU=1

python ../run_classifier_csci_emb.py ^
--pretrained_model_path ../models/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path E:\LiuHuan\Projects\UER-py\work\fine_tune_data\move_zh_r\uer_data\train.tsv ^
--dev_path E:\LiuHuan\Projects\UER-py\work\fine_tune_data\move_zh_r\uer_data\dev.tsv ^
--test_path E:\LiuHuan\Projects\UER-py\work\fine_tune_data\move_zh_r\uer_data\test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 128 ^
--batch_size 32 ^
--report_steps 100 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0
