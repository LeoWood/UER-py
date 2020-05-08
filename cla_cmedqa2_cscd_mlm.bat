python run_classifier.py ^
--pretrained_model_path models/cscd_R_based_on_google_zh_600000_best.bin ^
--vocab_path models/google_zh_vocab.txt ^
--output_model_path ./models/cla_cmedqa2_bcscd_mlm.bin ^
--train_path datasets/cmedqa2/train.tsv ^
--dev_path datasets/cmedqa2/dev.tsv ^
--test_path datasets/cmedqa2/test.tsv ^
--embedding bert ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 8 ^
--seq_length 256 ^
--batch_size 10 ^
--report_steps 100 ^
--gpu_rank 1
