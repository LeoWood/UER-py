# -*- encoding:utf -*-
"""
This script provides an example to wrap UER-py for NER.
"""
import random
import argparse

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

    # Model options.
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=128, type=int,
                        help="Sequence length.")
    parser.add_argument("--embedding", choices=["bert", "word"], default="bert",
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

    args = parser.parse_args()

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
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
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
    def batch_loader(batch_size, input_ids, label_ids, mask_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            f.readline()
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                tokens, labels = line.strip().split("\t")
                tokens = [vocab.get(t) for t in tokens.split(" ")]
                labels = [labels_map[l] for l in labels.split(" ")]
                mask = [1] * len(tokens)
                if len(tokens) > args.seq_length:
                    tokens = tokens[:args.seq_length]
                    labels = labels[:args.seq_length]
                    mask = mask[:args.seq_length]
                while len(tokens) < args.seq_length:
                    tokens.append(0)
                    labels.append(0)
                    mask.append(0)
                dataset.append([tokens, labels, mask])
        
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

        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch)
            
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

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    total_loss = 0.
    f1 = 0.0
    best_f1 = 0.0

    for epoch in range(1, args.epochs_num+1):
        model.train()
        for i, (input_ids_batch, label_ids_batch, mask_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids)):
            model.zero_grad()

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)

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
        else:
            continue

    # Evaluation phase.
    if args.test_path is not None:
        print("Test set evaluation.")
        model = load_model(model, args.output_model_path)
        evaluate(args, True)


if __name__ == "__main__":
    main()

