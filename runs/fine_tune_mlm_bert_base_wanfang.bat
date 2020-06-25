set PRETRAINED_MODEL=pre_r_wanfang_from_bert_base_20w-250w-best.bin
set TUNE_MODEL=fine_tune_0.bin
set VOCAB=google_zh_vocab.txt
set EMBEDDING=bert
set LOG_FILE=pre_r_wanfang_from_bert_base_20w-250w-best.log
set GPU=0


echo "amttl:" >> %LOG_FILE%
python ../run_ner_csci_emb.py ^
--pretrained_model_path ../output_pre/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/amttl/train.tsv ^
--dev_path ../datasets/amttl/dev.tsv ^
--test_path ../datasets/amttl/test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 10 ^
--seq_length 128 ^
--batch_size 16 ^
--report_steps 50 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0

echo "ccks:" >> %LOG_FILE%
python ../run_ner_csci_emb.py ^
--pretrained_model_path ../output_pre/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/ccks/train.tsv ^
--dev_path ../datasets/ccks/dev.tsv ^
--test_path ../datasets/ccks/test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 10 ^
--seq_length 128 ^
--batch_size 16 ^
--report_steps 50 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0

echo "cnmer:" >> %LOG_FILE%
python ../run_ner_csci_emb.py ^
--pretrained_model_path ../output_pre/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/cnmer/train.tsv ^
--dev_path ../datasets/cnmer/dev.tsv ^
--test_path ../datasets/cnmer/test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 10 ^
--seq_length 256 ^
--batch_size 5 ^
--report_steps 50 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0

echo "cmedqa:" >> %LOG_FILE%
python ../run_classifier_csci_emb.py ^
--pretrained_model_path ../output_pre/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/cmedqa/train.tsv ^
--dev_path ../datasets/cmedqa/dev.tsv ^
--test_path ../datasets/cmedqa/test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 256 ^
--batch_size 10 ^
--report_steps 100 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0

echo "csl:" >> %LOG_FILE%
python ../run_classifier_csci_emb.py ^
--pretrained_model_path ../output_pre/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/csl/train.tsv ^
--dev_path ../datasets/csl/dev.tsv ^
--test_path ../datasets/csl/test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 1e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 256 ^
--batch_size 10 ^
--report_steps 100 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0

echo "cla_16:" >> %LOG_FILE%
python ../run_classifier_csci_emb.py ^
--pretrained_model_path ../output_pre/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/wanfang_16000/train.tsv ^
--dev_path ../datasets/wanfang_16000/dev.tsv ^
--test_path ../datasets/wanfang_16000/test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 400 ^
--batch_size 5 ^
--report_steps 100 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0

echo "cla_32:" >> %LOG_FILE%
python ../run_classifier_csci_emb.py ^
--pretrained_model_path ../output_pre/%PRETRAINED_MODEL% ^
--vocab_path ../models/%VOCAB% ^
--output_model_path ../output_tune/%TUNE_MODEL% ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/wanfang_32000/train.tsv ^
--dev_path ../datasets/wanfang_32000/dev.tsv ^
--test_path ../datasets/wanfang_32000/test.tsv ^
--log_path %LOG_FILE% ^
--embedding %EMBEDDING% ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 400 ^
--batch_size 5 ^
--report_steps 100 ^
--gpu_rank %GPU% ^
--add_pos 0 ^
--add_term 0 ^
--init_pos 0 ^
--init_term 0