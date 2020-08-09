set PYTHONUNBUFFERED=1
set PRETRAINED_MODEL=google_zh_model.bin
set VOCAB=google_zh_vocab.txt
set EMBEDDING=cscibert
set LOG_FILE=pos_term_bert_base.log
set GPU=0


echo "amttl:" >> $LOG_FILE
python ../run_ner_csci_emb_old.py ^
--pretrained_model_path ../models/$PRETRAINED_MODEL ^
--vocab_path ../models/$VOCAB ^
--output_model_path ../models/fine_tune.bin ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/amttl/train.tsv ^
--dev_path ../datasets/amttl/dev.tsv ^
--test_path ../datasets/amttl/test.tsv ^
--train_pt_path ../datasets/amttl/train.pt ^
--dev_pt_path ../datasets/amttl/dev.pt ^
--test_pt_path ../datasets/amttl/test.pt ^
--log_path $LOG_FILE ^
--embedding $EMBEDDING ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 10 ^
--seq_length 128 ^
--batch_size 16 ^
--report_steps 50 ^
--gpu_rank $GPU ^
--add_pos 1 ^
--add_term 1 ^
--init_pos 1 ^
--init_term 1 ^
--preprocess 1

echo "ccks:" >> $LOG_FILE
python ../run_ner_csci_emb_old.py ^
--pretrained_model_path ../models/$PRETRAINED_MODEL ^
--vocab_path ../models/$VOCAB ^
--output_model_path ../models/fine_tune.bin ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/ccks/train.tsv ^
--dev_path ../datasets/ccks/dev.tsv ^
--test_path ../datasets/ccks/test.tsv ^
--train_pt_path ../datasets/ccks/train.pt ^
--dev_pt_path ../datasets/ccks/dev.pt ^
--test_pt_path ../datasets/ccks/test.pt ^
--log_path $LOG_FILE ^
--embedding $EMBEDDING ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 10 ^
--seq_length 128 ^
--batch_size 16 ^
--report_steps 50 ^
--gpu_rank $GPU ^
--add_pos 1 ^
--add_term 1 ^
--init_pos 1 ^
--init_term 1 ^
--preprocess 1

echo "cnmer:" >> $LOG_FILE
python ../run_ner_csci_emb_old.py ^
--pretrained_model_path ../models/$PRETRAINED_MODEL ^
--vocab_path ../models/$VOCAB ^
--output_model_path ../models/fine_tune.bin ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/cnmer/train.tsv ^
--dev_path ../datasets/cnmer/dev.tsv ^
--test_path ../datasets/cnmer/test.tsv ^
--train_pt_path ../datasets/cnmer/train.pt ^
--dev_pt_path ../datasets/cnmer/dev.pt ^
--test_pt_path ../datasets/cnmer/test.pt ^
--log_path $LOG_FILE ^
--embedding $EMBEDDING ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 10 ^
--seq_length 256 ^
--batch_size 5 ^
--report_steps 50 ^
--gpu_rank $GPU ^
--add_pos 1 ^
--add_term 1 ^
--init_pos 1 ^
--init_term 1 ^
--preprocess 1

echo "cmedqa:" >> $LOG_FILE
python ../run_classifier_csci_emb_old.py ^
--pretrained_model_path ../models/$PRETRAINED_MODEL ^
--vocab_path ../models/$VOCAB ^
--output_model_path ../models/fine_tune.bin ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/cmedqa/train.tsv ^
--dev_path ../datasets/cmedqa/dev.tsv ^
--test_path ../datasets/cmedqa/test.tsv ^
--train_pt_path ../datasets/cmedqa/train.pt ^
--dev_pt_path ../datasets/cmedqa/dev.pt ^
--test_pt_path ../datasets/cmedqa/test.pt ^
--log_path $LOG_FILE ^
--embedding $EMBEDDING ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 256 ^
--batch_size 10 ^
--report_steps 100 ^
--gpu_rank $GPU ^
--add_pos 1 ^
--add_term 1 ^
--init_pos 1 ^
--init_term 1 ^
--preprocess 1

echo "csl:" >> $LOG_FILE
python ../run_classifier_csci_emb_old.py ^
--pretrained_model_path ../models/$PRETRAINED_MODEL ^
--vocab_path ../models/$VOCAB ^
--output_model_path ../models/fine_tune.bin ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/csl/train.tsv ^
--dev_path ../datasets/csl/dev.tsv ^
--test_path ../datasets/csl/test.tsv ^
--train_pt_path ../datasets/csl/train.pt ^
--dev_pt_path ../datasets/csl/dev.pt ^
--test_pt_path ../datasets/csl/test.pt ^
--log_path $LOG_FILE ^
--embedding $EMBEDDING ^
--encoder bert ^
--learning_rate 1e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 256 ^
--batch_size 10 ^
--report_steps 100 ^
--gpu_rank $GPU ^
--add_pos 1 ^
--add_term 1 ^
--init_pos 1 ^
--init_term 1 ^
--preprocess 1

echo "cla_16:" >> $LOG_FILE
python ../run_classifier_csci_emb_old.py ^
--pretrained_model_path ../models/$PRETRAINED_MODEL ^
--vocab_path ../models/$VOCAB ^
--output_model_path ../models/fine_tune.bin ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/wanfang_16000/train.tsv ^
--dev_path ../datasets/wanfang_16000/dev.tsv ^
--test_path ../datasets/wanfang_16000/test.tsv ^
--train_pt_path ../datasets/wanfang_16000/train.pt ^
--dev_pt_path ../datasets/wanfang_16000/dev.pt ^
--test_pt_path ../datasets/wanfang_16000/test.pt ^
--log_path $LOG_FILE ^
--embedding $EMBEDDING ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 400 ^
--batch_size 5 ^
--report_steps 100 ^
--gpu_rank $GPU ^
--add_pos 1 ^
--add_term 1 ^
--init_pos 1 ^
--init_term 1 ^
--preprocess 1

echo "cla_32:" >> $LOG_FILE
python ../run_classifier_csci_emb_old.py ^
--pretrained_model_path ../models/$PRETRAINED_MODEL ^
--vocab_path ../models/$VOCAB ^
--output_model_path ../models/fine_tune.bin ^
--config_path ../models/bert_base_config.json ^
--train_path ../datasets/wanfang_32000/train.tsv ^
--dev_path ../datasets/wanfang_32000/dev.tsv ^
--test_path ../datasets/wanfang_32000/test.tsv ^
--train_pt_path ../datasets/wanfang_32000/train.pt ^
--dev_pt_path ../datasets/wanfang_32000/dev.pt ^
--test_pt_path ../datasets/wanfang_32000/test.pt ^
--log_path $LOG_FILE ^
--embedding $EMBEDDING ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 5 ^
--seq_length 400 ^
--batch_size 5 ^
--report_steps 100 ^
--gpu_rank $GPU ^
--add_pos 1 ^
--add_term 1 ^
--init_pos 1 ^
--init_term 1 ^
--preprocess 1