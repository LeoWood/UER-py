# -*- encoding:utf-8 -*-
"""
  This script provides an exmaple to wrap UER-py for classification.
"""
import torch
import os
import json
import random
import argparse
import collections
import torch.nn as nn

from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils.tokenizer import * 
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model

import pkuseg
import time

# import os
# import sys
# os.chdir(sys.path[0])


pku_seg = pkuseg.pkuseg(model_name="medicine",user_dict="../uer/utils/pku_seg_dict.txt")
pku_seg_pos = pkuseg.pkuseg(model_name="medicine",user_dict="../uer/utils/pku_seg_dict.txt",postag=True)


pos_dict = {}
pos_dict_reverse = {}
with open('../uer/utils/pos_tags.txt','r',encoding='utf-8') as f:
    i = 0
    for line in f.readlines():
        if line:
            pos_dict[line.strip().split()[0]] = i
            pos_dict_reverse[i] = line.strip().split()[0]
            i += 1


# 获取本地术语表
a = []
with open('../uer/utils/medical_terms/medical_terms(final).txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        a.append(line)

term_set = set(a)


class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()

    def forward(self, src, label, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask)
        # Encoder.
        output = self.encoder(emb, mask)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.") 
    parser.add_argument("--test_path", type=str,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/bert_base_config.json", type=str,
                        help="Path of the config file.")
    parser.add_argument("--log_path", default="./models/test.log", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--add_pos", type=int, default=0,
                        help="if you want to add pos infomation in csci_mlm target, use 1/0 = yes/no.")
    parser.add_argument("--init_pos", type=int, default=0,
                        help="if you have to init pos_embedding as there might not be term_embedding in the pretrained model, use 1/0 = yes/no.")
    parser.add_argument("--add_term", type=int, default=0,
                        help="if you want to add term infomation in csci_mlm target, use 1/0 = yes/no.")
    parser.add_argument("--init_term", type=int, default=0,
                        help="if you have to init term_embedding as there might not be term_embedding in the pretrained model, use 1/0 = yes/no.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=128,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word", "cscibert"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer." 
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    
    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=3,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")
    # GPU
    parser.add_argument("--gpu_rank", type=str, default='0',
                        help="Gpu Rank.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_rank

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set) 

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location='cuda:' + args.gpu_rank), strict=False)
        ## 对加入的pos_embedding和term_embedding的初始化
        for n, p in list(model.named_parameters()):
            if args.init_pos and n == "embedding.pos_embedding.weight":
                print("pos_embedding 随机初始化")
                p.data.normal_(0, 0.02)
            if args.init_term and n == "embedding.term_embedding.weight":
                print("term_embedding 随机初始化")
                p.data.normal_(0, 0.02)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build classification model.
    model = BertClassifier(args, model)


    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, term_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            term_ids_batch = term_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, term_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            term_ids_batch = term_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, term_ids_batch

    # Build tokenizer.
    tokenizer = globals()[args.tokenizer.capitalize() + "Tokenizer"](args)

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0:
                    continue
                line = line.strip().split('\t')
                if len(line) == 2:
                    label = int(line[columns["label"]])
                    text = line[columns["text_a"]]

                    tokens = []
                    for word in pku_seg.cut(text):
                        for w in tokenizer.tokenize(word):
                            tokens.append(vocab.get(w))

                    tokens = [CLS_ID] + tokens
                    mask = [1] * len(tokens)

                    src_pos = []
                    src_term = []
                    ## 加入pos 和terms
                    for (word, tag) in pku_seg_pos.cut(text):
                        piece_num = len(tokenizer.tokenize(word))
                        if word in term_set:
                            # print(word)
                            [src_term.append(1) for i in range(piece_num)]
                        else:
                            [src_term.append(2) for i in range(piece_num)]

                        [src_pos.append(pos_dict[tag]) for i in range(piece_num)]


                    src_pos = [pos_dict['[CLS]']] + src_pos
                    src_term = [2] + src_term


                    if len(tokens) > args.seq_length:
                        tokens = tokens[:args.seq_length]
                        mask = mask[:args.seq_length]
                        src_pos = src_pos[:args.seq_length]
                        src_term = src_term[:args.seq_length]


                    while len(tokens) < args.seq_length:
                        tokens.append(0)
                        mask.append(0)
                        src_pos.append(pos_dict['[PAD]'])
                        src_term.append(0)

                    # if line_id < 3:
                    #     print("数据读取示例：")
                    #     print('Tokens:')
                    #     print([(i,a) for (i,a) in enumerate(tokenizer.convert_ids_to_tokens(tokens))])
                    #
                    #     print("pos:")
                    #     print([(i,pos_dict_reverse[a]) for (i,a) in enumerate(src_pos)])
                    #     print("term:")
                    #     print([(i,a) for (i,a) in enumerate(src_term)])
                    #
                    #     print("label:")
                    #     print(label)
                    #
                    #     print("mask:")
                    #     print([(i,a) for (i,a) in enumerate(mask)])


                    dataset.append((tokens, label, mask, src_pos, src_term))

                elif len(line) == 3: # For sentence pair input.
                    label = int(line[columns["label"]])
                    text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]

                    tokens_a = []
                    for word in pku_seg.cut(text_a):
                        for w in tokenizer.tokenize(word):
                            tokens_a.append(vocab.get(w))

                    tokens_a = [CLS_ID] + tokens_a + [SEP_ID]

                    tokens_b = []
                    for word in pku_seg.cut(text_b):
                        for w in tokenizer.tokenize(word):
                            tokens_b.append(vocab.get(w))

                    tokens_b = tokens_b + [SEP_ID]

                    tokens = tokens_a + tokens_b
                    mask = [1] * len(tokens_a) + [2] * len(tokens_b)

                    src_pos_a = []
                    src_pos_b = []
                    src_term_a = []
                    src_term_b = []
                    ## 加入pos 和terms

                    for (word, tag) in pku_seg_pos.cut(text_a):
                        piece_num = len(tokenizer.tokenize(word))
                        if word in term_set:
                            # print(word)
                            [src_term_a.append(1) for i in range(piece_num)]
                        else:
                            [src_term_a.append(2) for i in range(piece_num)]

                        [src_pos_a.append(pos_dict[tag]) for i in range(piece_num)]

                    src_pos_a = [pos_dict['[CLS]']] + src_pos_a + [pos_dict['[SEP]']]
                    src_term_a = [2] + src_term_a + [2]

                    for (word, tag) in pku_seg_pos.cut(text_b):
                        piece_num = len(tokenizer.tokenize(word))
                        if word in term_set:
                            # print(word)
                            [src_term_b.append(1) for i in range(piece_num)]
                        else:
                            [src_term_b.append(2) for i in range(piece_num)]

                        [src_pos_b.append(pos_dict[tag]) for i in range(piece_num)]

                    src_pos_b = src_pos_b + [pos_dict['[SEP]']]
                    src_term_b = src_term_b + [2]

                    src_pos = src_pos_a + src_pos_b
                    src_term = src_term_a + src_term_b

                    if len(src_pos) != len(tokens):
                        print('Tokens:')
                        print([(i, a) for (i, a) in enumerate(tokenizer.convert_ids_to_tokens(tokens))])

                        print("pos:")
                        print([(i, pos_dict_reverse[a]) for (i, a) in enumerate(src_pos)])
                        print("term:")
                        print([(i, a) for (i, a) in enumerate(src_term)])
                        exit()


                    if len(tokens) > args.seq_length:
                        tokens = tokens[:args.seq_length]
                        mask = mask[:args.seq_length]
                        src_pos = src_pos[:args.seq_length]
                        src_term = src_term[:args.seq_length]

                    while len(tokens) < args.seq_length:
                        tokens.append(0)
                        mask.append(0)
                        src_pos.append(pos_dict['[PAD]'])
                        src_term.append(0)
                    dataset.append((tokens, label, mask, src_pos, src_term))

                elif len(line) == 4: # For dbqa input.
                    qid=int(line[columns["qid"]])
                    label = int(line[columns["label"]])
                    text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]

                    tokens_a = [vocab.get(t) for t in tokenizer.tokenize(text_a)]
                    tokens_a = [CLS_ID] + tokens_a + [SEP_ID]
                    tokens_b = [vocab.get(t) for t in tokenizer.tokenize(text_b)]
                    tokens_b = tokens_b + [SEP_ID]

                    tokens = tokens_a + tokens_b
                    mask = [1] * len(tokens_a) + [2] * len(tokens_b)

                    if len(tokens) > args.seq_length:
                        tokens = tokens[:args.seq_length]
                        mask = mask[:args.seq_length]
                    while len(tokens) < args.seq_length:
                        tokens.append(0)
                        mask.append(0)
                    dataset.append((tokens, label, mask, qid))
                else:
                    pass

        return dataset

    # Evaluation function.
    def evaluate(args, is_test):
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([sample[3] for sample in dataset])
        term_ids = torch.LongTensor([sample[4] for sample in dataset])

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()
        
        if not args.mean_reciprocal_rank:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, term_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, term_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                term_ids_batch = term_ids_batch.to(device)
                with torch.no_grad():
                    if args.add_pos and args.add_term:
                        loss, logits = model((input_ids_batch,pos_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
                    elif args.add_pos:
                        loss, logits = model((input_ids_batch,pos_ids_batch), label_ids_batch, mask_ids_batch)
                    elif args.add_term:
                        loss, logits = model((input_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
                    else:
                        loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)

                logits = nn.Softmax(dim=1)(logits)
                pred = torch.argmax(logits, dim=1)
                gold = label_ids_batch
                for j in range(pred.size()[0]):
                    confusion[pred[j], gold[j]] += 1
                correct += torch.sum(pred == gold).item()

            # if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")
            f1_all = []
            for i in range(confusion.size()[0]):
                p = confusion[i,i].item()/confusion[i,:].sum().item()
                r = confusion[i,i].item()/confusion[:,i].sum().item()
                f1 = 2*p*r / (p+r)
                f1_all.append((f1,confusion[i,:].sum().item()))
                print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i,p,r,f1))
            f1_weighted = 0
            num_all = 0
            for f1,num in f1_all:
                f1_weighted += f1*num
                num_all += num
            f1_weighted = f1_weighted/num_all

            print("weited_f1: {:.4f}".format(f1_weighted))
            print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct/len(dataset), correct, len(dataset)))
            return f1_weighted
        else:
            for i, (input_ids_batch, label_ids_batch,  mask_ids_batch, pos_ids_batch, term_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, term_ids)):
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                term_ids_batch = term_ids_batch.to(device)
                with torch.no_grad():
                    if args.add_pos and args.add_term:
                        loss, logits = model((input_ids_batch,pos_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
                    elif args.add_pos:
                        loss, logits = model((input_ids_batch,pos_ids_batch), label_ids_batch, mask_ids_batch)
                    elif args.add_term:
                        loss, logits = model((input_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
                    else:
                        loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch)

                logits = nn.Softmax(dim=1)(logits)
                if i == 0:
                    logits_all=logits
                if i >= 1:
                    logits_all=torch.cat((logits_all,logits),0)
        
            order = -1
            gold = []
            for i in range(len(dataset)):
                qid = dataset[i][3]
                label = dataset[i][1]
                if qid == order:
                    j += 1
                    if label == 1:
                        gold.append((qid,j))
                else:
                    order = qid
                    j = 0
                    if label == 1:
                        gold.append((qid,j))


            label_order = []
            order = -1
            for i in range(len(gold)):
                if gold[i][0] == order:
                    templist.append(gold[i][1])
                elif gold[i][0] != order:
                    order=gold[i][0]
                    if i > 0:
                        label_order.append(templist)
                    templist = []
                    templist.append(gold[i][1])
            label_order.append(templist)

            order = -1
            score_list = []
            for i in range(len(logits_all)):
                score = float(logits_all[i][1])
                qid=int(dataset[i][3])
                if qid == order:
                    templist.append(score)
                else:
                    order = qid
                    if i > 0:
                        score_list.append(templist)
                    templist = []
                    templist.append(score)
            score_list.append(templist)

            rank = []
            pred = []
            for i in range(len(score_list)):
                if len(label_order[i])==1:
                    if label_order[i][0] < len(score_list[i]):
                        true_score = score_list[i][label_order[i][0]]
                        score_list[i].sort(reverse=True)
                        for j in range(len(score_list[i])):
                            if score_list[i][j] == true_score:
                                rank.append(1 / (j + 1))
                    else:
                        rank.append(0)

                else:
                    true_rank = len(score_list[i])
                    for k in range(len(label_order[i])):
                        if label_order[i][k] < len(score_list[i]):
                            true_score = score_list[i][label_order[i][k]]
                            temp = sorted(score_list[i],reverse=True)
                            for j in range(len(temp)):
                                if temp[j] == true_score:
                                    if j < true_rank:
                                        true_rank = j
                    if true_rank < len(score_list[i]):
                        rank.append(1 / (true_rank + 1))
                    else:
                        rank.append(0)
            MRR = sum(rank) / len(rank)
            print("Mean Reciprocal Rank: {:.4f}".format(MRR))
            return MRR

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    input_ids = torch.LongTensor([example[0] for example in trainset])
    label_ids = torch.LongTensor([example[1] for example in trainset])
    mask_ids = torch.LongTensor([example[2] for example in trainset])
    pos_ids = torch.LongTensor([example[3] for example in trainset])
    term_ids = torch.LongTensor([example[4] for example in trainset])

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps*args.warmup, t_total=train_steps)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    total_loss = 0.
    result = 0.0
    best_result = 0.0
    
    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, term_ids_batch) in enumerate(
                batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, term_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            term_ids_batch = term_ids_batch.to(device)
            if args.add_pos and args.add_term:
                loss, _ = model((input_ids_batch,pos_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
            elif args.add_pos:
                loss, _ = model((input_ids_batch,pos_ids_batch), label_ids_batch, mask_ids_batch)
            elif args.add_term:
                loss, _ = model((input_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
            else:
                loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch)
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i+1, total_loss / args.report_steps))
                total_loss = 0.

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()
            scheduler.step()
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
            print('~~~ Best Result Until Now ~~~')
            with open(args.log_path,'a',encoding='utf-8') as f:
                f.write('BEST F1 on dev Until Now :'+ str(result) + '\n')
        else:
            continue

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        model = load_model(model, args.output_model_path)
        result = evaluate(args, True)
        with open(args.log_path, 'a', encoding='utf-8') as f:
            f.write('F1 on test:' + str(result) + '\n')


if __name__ == "__main__":
    main()
