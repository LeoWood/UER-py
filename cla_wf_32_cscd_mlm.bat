python run_classifier.py ^
--pretrained_model_path models/cscd_R_based_on_google_zh_600000_best.bin ^
--vocab_path models/google_zh_vocab.txt ^
--output_model_path ./models/cla_wf_32_bcscd_mlm.bin ^
--train_path datasets/wanfang_32000/train.tsv ^
--dev_path datasets/wanfang_32000/dev.tsv ^
--test_path datasets/wanfang_32000/test.tsv ^
--embedding bert ^
--encoder bert ^
--learning_rate 2e-5 ^
--warmup 0.1 ^
--epochs_num 3 ^
--seq_length 400 ^
--batch_size 5 ^
--report_steps 100 ^
--gpu_rank 1
