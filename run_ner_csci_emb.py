# -*- encoding:utf -*-

import random
import argparse
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import *
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.model_loader import load_model

import os
import sys
os.chdir(sys.path[0])


import pkuseg
import time


pku_seg = pkuseg.pkuseg(model_name="medicine",user_dict="uer/utils/pku_seg_dict.txt")
pku_seg_pos = pkuseg.pkuseg(model_name="medicine",user_dict="uer/utils/pku_seg_dict.txt",postag=True)


pos_dict = {}
pos_dict_reverse = {}
with open('uer/utils/pos_tags.txt','r',encoding='utf-8') as f:
    i = 0
    for line in f.readlines():
        if line:
            pos_dict[line.strip().split()[0]] = i
            pos_dict_reverse[i] = line.strip().split()[0]
            i += 1

print(pos_dict)
print(pos_dict_reverse)

# 获取本地术语表
a = []
with open('uer/utils/medical_terms/medical_terms(final).txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        a.append(line)

term_set = set(a)


class BertTagger(nn.Module):
    def __init__(self, args, model):
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, label, mask):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]

        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        """
        # Embedding.
        emb = self.embedding(src, mask)
        # Encoder.
        output = self.encoder(emb, mask)
        # Target.
        output = self.output_layer(output)

        output = output.contiguous().view(-1, self.labels_num)
        output = self.softmax(output)

        label = label.contiguous().view(-1,1)
        label_mask = (label > 0).float().to(torch.device(label.device))
        one_hot = torch.zeros(label_mask.size(0),  self.labels_num). \
                  to(torch.device(label.device)). \
                  scatter_(1, label, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator
        predict = output.argmax(dim=-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )
        
        return loss, correct, predict, label


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/ner_model.bin", type=str,
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
    parser.add_argument("--add_term", type=int, default=0,
                        help="if you want to add term infomation in csci_mlm target, use 1/0 = yes/no.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=128, type=int,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word", "cscibert"], default="bert",
                        help="Emebdding type.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    
    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

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
    parser.add_argument("--dropout", type=float, default=0.1,
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

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_rank

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    labels_map = {"[PAD]": 0}
    begin_ids = []

    # Find tagging labels
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            labels = line.strip().split("\t")[1].split()
            for l in labels:
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))
                    labels_map[l] = len(labels_map)
    

    print("Labels: ", labels_map)
    args.labels_num = len(labels_map)

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
            if n == "embedding.pos_embedding.weight" or n == "embedding.term_embedding.weight":
                p.data.normal_(0, 0.02)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build sequence labeling model.
    model = BertTagger(args, model)


    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, term_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            term_ids_batch = term_ids[i * batch_size: (i + 1) * batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, term_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:, :]
            term_ids_batch = term_ids[instances_num // batch_size * batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, term_ids_batch

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            f.readline()
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                tokens, labels = line.strip().split("\t")
                text = ''.join([t for t in tokens.split(" ")])

                tokens = [vocab.get(t) for t in tokens.split(" ")]
                labels = [labels_map[l] for l in labels.split(" ")]
                mask = [1] * len(tokens)

                src_pos = []
                src_term = []
                ## 加入pos 和terms


                for (word, tag) in pku_seg_pos.cut(text):
                    for w in word:
                        if word in term_set:
                            src_term.append(1)
                        else:
                            src_term.append(2)
                        src_pos.append(pos_dict[tag])

                assert len(src_pos) == len(tokens)

                if len(tokens) > args.seq_length:
                    tokens = tokens[:args.seq_length]
                    labels = labels[:args.seq_length]
                    mask = mask[:args.seq_length]
                    src_pos = src_pos[:args.seq_length]
                    src_term = src_term[:args.seq_length]


                while len(tokens) < args.seq_length:
                    tokens.append(0)
                    labels.append(0)
                    mask.append(0)
                    src_pos.append(pos_dict['[PAD]'])
                    src_term.append(0)
                dataset.append([tokens, labels, mask, src_pos, src_term])
        
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

        instances_num = input_ids.size(0)
        batch_size = args.batch_size

        if is_test:
            print("Batch size: ", batch_size)
            print("The number of test instances:", instances_num)

    
        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0

        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)

        model.eval()

        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, term_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, term_ids)):

            # print('Tokens:')
            # print([(i, vocab.i2w[a]) for (i, a) in enumerate(input_ids_batch[0])])
            #
            # print("pos:")
            # print([(i, pos_dict_reverse[a.item()]) for (i, a) in enumerate(pos_ids_batch[0])])
            #
            # print("term:")
            # print([(i, a.item()) for (i, a) in enumerate(term_ids_batch[0])])
            #
            # print("label:")
            # print(label_ids_batch[0])
            #
            # print("mask:")
            # print([(i, a.item()) for (i, a) in enumerate(mask_ids_batch[0])])

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            term_ids_batch = term_ids_batch.to(device)


            if args.add_pos and args.add_term:
                loss, _, pred, gold = model((input_ids_batch,pos_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
            elif args.add_pos:
                loss, _, pred, gold = model((input_ids_batch,pos_ids_batch), label_ids_batch, mask_ids_batch)
            elif args.add_term:
                loss, _, pred, gold = model((input_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
            else:
                loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch)


            # print('Tokens:')
            # print([(i, vocab.i2w[a]) for (i, a) in enumerate(input_ids_batch[0])])
            #
            # print([(i, a.item()) for (i, a) in enumerate(gold)])
            # print([(i, a.item()) for (i, a) in enumerate(pred)])
            # print(gold)
            # print(pred)
            # print("begin_ids:",begin_ids)
            # exit()

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    gold_entities_num += 1
 
            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:
                    pred_entities_num += 1


            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    start = j
                    for k in range(j+1, gold.size()[0]):
                        if gold[k].item() == labels_map["[PAD]"] or gold[k].item() == labels_map["O"] or gold[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = gold.size()[0] - 1
                    gold_entities_pos.append((start, end))
            
            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:
                    start = j
                    for k in range(j+1, pred.size()[0]):
                        if pred[k].item() == labels_map["[PAD]"] or pred[k].item() == labels_map["O"] or pred[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = pred.size()[0] - 1
                    pred_entities_pos.append((start, end))

            for entity in pred_entities_pos:
                if entity not in gold_entities_pos:
                    continue
                for j in range(entity[0], entity[1]+1):
                    if gold[j].item() != pred[j].item():
                        break
                else: 
                    correct += 1

        print("Report precision, recall, and f1:")
        # print(correct)
        # print(pred_entities_num)
        # print(gold_entities_num)
        #
        # if not pred_entities_num:
        #     return 0
        # if not correct:
        #     return 0

        p = correct/pred_entities_num
        r = correct/gold_entities_num
        f1 = 2*p*r/(p+r)
        print("{:.3f}, {:.3f}, {:.3f}".format(p,r,f1))

        return f1

    # Training phase.
    print("Start training.")
    instances = read_dataset(args.train_path)

    input_ids = torch.LongTensor([ins[0] for ins in instances])
    label_ids = torch.LongTensor([ins[1] for ins in instances])
    mask_ids = torch.LongTensor([ins[2] for ins in instances])
    pos_ids = torch.LongTensor([ins[3] for ins in instances])
    term_ids = torch.LongTensor([ins[4] for ins in instances])

    instances_num = input_ids.size(0)
    batch_size = args.batch_size
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

    # if torch.cuda.device_count() > 1:
    #     print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
    #     model = torch.nn.DataParallel(model)

    total_loss = 0.
    f1 = 0.0
    best_f1 = 0.0

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, term_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, term_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            term_ids_batch = term_ids_batch.to(device)

            if args.add_pos and args.add_term:
                loss, _, _, _ = model((input_ids_batch,pos_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
            elif args.add_pos:
                loss, _, _, _ = model((input_ids_batch,pos_ids_batch), label_ids_batch, mask_ids_batch)
            elif args.add_term:
                loss, _, _, _ = model((input_ids_batch,term_ids_batch), label_ids_batch, mask_ids_batch)
            else:
                loss, _, _, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch)


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

        f1 = evaluate(args, False)
        if f1 > best_f1:
            best_f1 = f1
            save_model(model, args.output_model_path)
            print('~~~ Best Result Until Now ~~~')
            with open(args.log_path, 'a', encoding='utf-8') as f:
                f.write('BEST F1 on dev:' + str(f1) + '\n')
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

