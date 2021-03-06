# -*- encoding:utf-8 -*-
import os
import sys
import time
from datetime import timedelta
import math
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from uer.model_loader import load_model
from uer.model_saver import save_model
from uer.model_builder import build_model
from uer.utils.optimizers import *
from uer.utils.data import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed

# pos_dict = {}
# pos_dict_reverse = {}
# with open('../uer/utils/pos_tags_old.txt','r',encoding='utf-8') as f:
#     i = 0
#     for line in f.readlines():
#         if line:
#             pos_dict[line.strip().split()[0]] = i
#             pos_dict_reverse[i] = line.strip().split()[0]
#             i += 1


def train_and_validate(args):
    set_seed(args.seed)

    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build model.
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model = load_model(model, args.pretrained_model_path) 
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)


    if args.dist_train:
        # Multiprocessing distributed mode.
        # mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model), daemon=False)
        mp.spawn(worker, nprocs=args.ranks_num, args=(args.gpu_ranks, args, model))
    elif args.single_gpu:
        # Single GPU mode.
        worker(args.gpu_id, None, args, model)
    else:
        # CPU mode.
        worker(None, None, args, model)


def worker(proc_id, gpu_ranks, args, model):
    """
    Args:
        proc_id: The id of GPU for single GPU mode;
                 The id of process (and GPU) for multiprocessing distributed mode.
        gpu_ranks: List of ranks of each process.
    """
    # set_seed(args.seed)

    if args.dist_train:
        rank = gpu_ranks[proc_id]
        gpu_id = proc_id

        # Initialize multiprocessing distributed training environment.
        dist.init_process_group(backend=args.backend,
                                init_method=args.master_ip,
                                world_size=args.world_size,
                                rank=rank,
                                timeout=timedelta(seconds=60))
                                
    elif args.single_gpu:
        rank = None
        gpu_id = proc_id
    else:
        rank = None
        gpu_id = None

    if args.dist_train:
        train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, rank, args.world_size, True)
    else:
        train_loader = globals()[args.target.capitalize() + "DataLoader"](args, args.dataset_path, args.batch_size, 0, 1, True)

    if gpu_id is not None: 
        torch.cuda.set_device(gpu_id)
        model.cuda(gpu_id)

    # Build optimizer.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.total_steps*args.warmup, t_total=args.total_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if args.dist_train:
        model = DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
        print("Worker %d is training ... " % rank)
    else:
        print("Worker is training ...")
    
    globals().get("train_"+args.target)(args, gpu_id, rank, train_loader, model, optimizer, scheduler)
    

def train_bert(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct_mlm, total_denominator = 0., 0. 
    # Calculate NSP accuracy.
    total_correct_nsp, total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    done_tokens = 0
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt_mlm, tgt_nsp, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt_mlm = tgt_mlm.cuda(gpu_id)
            tgt_nsp = tgt_nsp.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, (tgt_mlm, tgt_nsp), seg)
        loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator = loss_info
        
         # Backward.
        loss = loss_mlm + loss_nsp
        total_loss += loss.item()
        total_loss_mlm += loss_mlm.item()
        total_loss_nsp += loss_nsp.item()
        total_correct_mlm += correct_mlm.item()
        total_correct_nsp += correct_nsp.item()
        total_denominator += denominator.item()
        total_instances += src.size(0)
        done_tokens += src.size(0) * src.size(1)

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps
            loss_mlm = total_loss_mlm / args.report_steps
            loss_nsp = total_loss_nsp / args.report_steps

            elapsed = time.time() - start_time

            if args.dist_train:
                done_tokens *= args.world_size

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_mlm: {:3.3f}"
                  "| loss_nsp: {:3.3f}"
                  "| acc_mlm: {:3.3f}"
                  "| acc_nsp: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    loss_mlm,
                    loss_nsp,
                    total_correct_mlm / total_denominator,
                    total_correct_nsp  / total_instances))
            
            done_tokens = 0
            total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
            total_correct_mlm, total_denominator = 0., 0.
            total_correct_nsp, total_instances = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_lm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0. 
    # Calculate NSP accuracy.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    total_correct / total_denominator))
            
            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_bilm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_loss_forward, total_loss_backward = 0., 0., 0.
    # Calculate BiLM accuracy.
    total_correct_forward, total_correct_backward, total_denominator = 0., 0., 0. 
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt_forward, tgt_backward, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt_forward = tgt_forward.cuda(gpu_id)
            tgt_backward = tgt_backward.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, (tgt_forward, tgt_backward), seg)
        loss_forward, loss_backward, correct_forward, correct_backward, denominator = loss_info
        
        # Backward.
        loss = loss_forward + loss_backward
        total_loss += loss.item()
        total_loss_forward += loss_forward.item()
        total_loss_backward += loss_backward.item()
        total_correct_forward += correct_forward.item()
        total_correct_backward += correct_backward.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| loss_forward {:3.3f}"
                  "| loss_backward {:3.3f}"
                  "| acc_forward: {:3.3f}"
                  "| acc_backward: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss,
                    loss_forward,
                    loss_backward,
                    total_correct_forward / total_denominator,
                    total_correct_backward / total_denominator))
            
            total_loss, total_loss_forward, total_loss_backward = 0., 0., 0.
            total_correct_forward, total_correct_backward, total_denominator = 0., 0., 0. 

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_cls(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss = 0.
    total_correct, total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)
        
        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_instances += src.size(0)

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            print("| {:8d}/{:8d} steps"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps, 
                    total_steps, 
                    done_tokens / elapsed, 
                    loss, 
                    total_correct / total_instances))
            
            total_loss = 0.
            total_correct = 0.
            total_instances = 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps))

        steps += 1


def train_mlm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0. 
    # Calculate NSP accuracy.
    total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    report_dict = {}
    report_dict['steps'] = []
    report_dict['loss'] = []
    report_dict['acc'] = []
    with open(args.output_log_path,'a', encoding='utf-8') as f:
        f.write('steps,loss,acc\n')


    while True:
        if steps == total_steps + 1:
            break
        src, tgt, seg = next(loader_iter)

        if gpu_id is not None:
            src = src.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)

        # Forward.
        loss_info = model(src, tgt, seg)
        loss, correct, denominator = loss_info
        
        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        # print("gpu_id:",gpu_id)

        # loss = total_loss / args.report_steps

        # elapsed = time.time() - start_time

        # done_tokens = \
        #     args.batch_size * src.size(1) * args.report_steps * args.world_size \
        #     if args.dist_train \
        #     else args.batch_size * src.size(1) * args.report_steps

        # acc = total_correct / total_denominator
        # print("| {:8d}/{:8d} steps"
        #         "| {:7.2f} steps/s"
        #         "| {:8.2f} tokens/s"
        #         "| loss {:7.2f}"
        #         "| acc: {:3.3f}".format(
        #         steps,
        #         total_steps,
        #         args.report_steps / elapsed,
        #         done_tokens / elapsed, 
        #         loss, 
        #         acc))

        if steps % args.report_steps == 0  and \
            (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src.size(1) * args.report_steps * args.world_size \
                if args.dist_train \
                else args.batch_size * src.size(1) * args.report_steps

            acc = total_correct / total_denominator
            print("| {:8d}/{:8d} steps"
                  "| {:7.2f} steps/s"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                    steps,
                    total_steps,
                    args.report_steps / elapsed,
                    done_tokens / elapsed, 
                    loss, 
                    acc))

            report_dict['steps'].append(steps)
            report_dict['loss'].append(loss)
            report_dict['acc'].append(acc)

            with open(args.output_log_path,'a', encoding='utf-8') as f:
                f.write(str(steps) + ',' + str(loss) + ',' + str(acc) + '\n')

            best_score = max(report_dict['acc'])
            if acc >= best_score and acc >= 0.85 and \
                    (not args.dist_train or (args.dist_train and rank == 0)):
                save_model(model, args.output_model_path + "-best")
                print("~~~New Best Score acc: {:3.3f}~~~".format(acc))


            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps) + "-" + str(round(loss,2)))

        steps += 1

    # report = pd.DataFrame(report_dict)
    # report.to_csv(args.output_log_path)



def train_csci_mlm(args, gpu_id, rank, loader, model, optimizer, scheduler):
    model.train()
    start_time = time.time()
    total_loss, total_loss_mlm, total_loss_nsp = 0., 0., 0.
    # Calculate MLM accuracy.
    total_correct, total_denominator = 0., 0.
    # Calculate NSP accuracy.
    total_instances = 0., 0.
    steps = 1
    total_steps = args.total_steps
    loader_iter = iter(loader)

    report_dict = {}
    report_dict['steps'] = []
    report_dict['loss'] = []
    report_dict['acc'] = []
    with open(args.output_log_path,'a', encoding='utf-8') as f:
        f.write('steps,loss,acc\n')

    while True:
        if steps == total_steps + 1:
            break
        src_word, src_pos, src_term, tgt, seg = next(loader_iter)

        # vocab = Vocab()
        # vocab.load(args.vocab_path)
        # print('Tokens:')
        # print([(i,vocab.i2w[a]) for (i,a) in enumerate(src_word[0])])
    
        # print("pos:")
        # print([(i,pos_dict_reverse[a.item()]) for (i,a) in enumerate(src_pos[0])])

        # print("term:")
        # print([(i,a) for (i,a) in enumerate(src_term[0])])
    
    
        # print("mask:")
        # print([(i,a) for (i,a) in enumerate(seg[0])])

        # exit()

        if gpu_id is not None:
            src_word = src_word.cuda(gpu_id)
            src_pos = src_pos.cuda(gpu_id)
            src_term = src_term.cuda(gpu_id)
            tgt = tgt.cuda(gpu_id)
            seg = seg.cuda(gpu_id)

        # Forward.
        if args.add_pos and args.add_term:
            loss_info = model((src_word, src_pos, src_term), tgt, seg)
        elif args.add_term:
            loss_info = model((src_word, src_term), tgt, seg)
        elif args.add_pos:
            loss_info = model((src_word, src_pos), tgt, seg)
        else:
            loss_info = model(src_word, tgt, seg)

        loss, correct, denominator = loss_info

        # Backward.
        total_loss += loss.item()
        total_correct += correct.item()
        total_denominator += denominator.item()

        loss = loss / args.accumulation_steps

        if args.fp16:
            with args.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if steps % args.accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        if steps % args.report_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):

            loss = total_loss / args.report_steps

            elapsed = time.time() - start_time

            done_tokens = \
                args.batch_size * src_word.size(1) * args.report_steps * args.world_size \
                    if args.dist_train \
                    else args.batch_size * src_word.size(1) * args.report_steps

            acc = total_correct / total_denominator
            print("| {:8d}/{:8d} steps"
                  "| {:7.2f} steps/s"
                  "| {:8.2f} tokens/s"
                  "| loss {:7.2f}"
                  "| acc: {:3.3f}".format(
                steps,
                total_steps,
                args.report_steps / elapsed,
                done_tokens / elapsed,
                loss,
                acc))

            report_dict['steps'].append(steps)
            report_dict['loss'].append(loss)
            report_dict['acc'].append(acc)

            with open(args.output_log_path,'a', encoding='utf-8') as f:
                f.write(str(steps) + ',' + str(loss) + ',' + str(acc) + '\n')

            best_score = max(report_dict['acc'])
            if acc >= best_score and acc >= 0.85 and \
                    (not args.dist_train or (args.dist_train and rank == 0)):
                save_model(model, args.output_model_path + "-best")
                print("~~~New Best Score acc: {:3.3f} loss: {:3.2f}~~~".format(acc,loss))

            total_loss = 0.
            total_correct, total_denominator = 0., 0.

            start_time = time.time()

        if steps % args.save_checkpoint_steps == 0 and \
                (not args.dist_train or (args.dist_train and rank == 0)):
            save_model(model, args.output_model_path + "-" + str(steps) + "-" + str(round(loss, 2)))

        steps += 1

    # report = pd.DataFrame(report_dict)
    # report.to_csv(args.output_log_path)


# def train_nsp(args, gpu_id, rank, loader, model, optimizer):
#     model.train()
#     start_time = time.time()
#     total_loss = 0.
#     total_correct, total_instances = 0., 0.
#     steps = 1
#     total_steps = args.total_steps
#     loader_iter = iter(loader)

#     while True:
#         if steps == total_steps + 1:
#             break
#         src, tgt, seg = next(loader_iter)

#         if gpu_id is not None:
#             src = src.cuda(gpu_id)
#             tgt = tgt.cuda(gpu_id)
#             seg = seg.cuda(gpu_id)
        
#         # Forward.
#         loss_info = model(src, tgt, seg)
#         loss, correct = loss_info
        
#         # Backward.
#         total_loss += loss.item()
#         total_correct += correct.item()
#         total_instances += src.size(0)

#         loss = loss / args.accumulation_steps
#         loss.backward()

#         if steps % args.accumulation_steps == 0:
#             optimizer.step()
#             model.zero_grad()
        
#         if steps % args.report_steps == 0  and \
#             (not args.dist_train or (args.dist_train and rank == 0)):

#             loss = total_loss / args.report_steps

#             elapsed = time.time() - start_time

#             done_tokens = \
#                 args.batch_size * src.size(1) * args.report_steps * args.world_size \
#                 if args.dist_train \
#                 else args.batch_size * src.size(1) * args.report_steps

#             print("| {:8d}/{:8d} steps"
#                   "| {:8.2f} tokens/s"
#                   "| loss {:7.2f}"
#                   "| acc: {:3.3f}".format(
#                     steps, 
#                     total_steps, 
#                     done_tokens / elapsed, 
#                     loss, 
#                     total_correct / total_instances))
            
#             total_loss = 0.
#             total_correct = 0.
#             total_instances = 0.

#             start_time = time.time()

#         if steps % args.save_checkpoint_steps == 0 and \
#                 (not args.dist_train or (args.dist_train and rank == 0)):
#             save_model(model, args.output_model_path + "-" + str(steps))

#         steps += 1


# def train_s2s(args, gpu_id, rank, loader, model, optimizer):
#     model.train()
#     start_time = time.time()
#     total_loss= 0.
#     total_correct, total_denominator = 0., 0. 
#     steps = 1
#     total_steps = args.total_steps
#     loader_iter = iter(loader)

#     while True:
#         if steps == total_steps + 1:
#             break
#         src, tgt, seg = next(loader_iter)

#         if gpu_id is not None:
#             src = src.cuda(gpu_id)
#             tgt = tgt.cuda(gpu_id)
#             seg = seg.cuda(gpu_id)
        
#         # Forward.
#         loss_info = model(src, tgt, seg)
#         loss, correct, denominator = loss_info
        
#         # Backward.
#         total_loss += loss.item()
#         total_correct += correct.item()
#         total_denominator += denominator.item()

#         loss = loss / args.accumulation_steps
#         loss.backward()

#         if steps % args.accumulation_steps == 0:
#             optimizer.step()
#             model.zero_grad()
        
#         if steps % args.report_steps == 0  and \
#             (not args.dist_train or (args.dist_train and rank == 0)):

#             loss = total_loss / args.report_steps

#             elapsed = time.time() - start_time

#             done_tokens = \
#                 args.batch_size * src.size(1) * args.report_steps * args.world_size \
#                 if args.dist_train \
#                 else args.batch_size * src.size(1) * args.report_steps

#             print("| {:8d}/{:8d} steps"
#                   "| {:8.2f} tokens/s"
#                   "| loss {:7.2f}"
#                   "| acc: {:3.3f}".format(
#                     steps, 
#                     total_steps, 
#                     done_tokens / elapsed, 
#                     loss, 
#                     total_correct / total_denominator))
            
#             total_loss = 0.
#             total_correct, total_denominator = 0., 0.

#             start_time = time.time()

#         if steps % args.save_checkpoint_steps == 0 and \
#                 (not args.dist_train or (args.dist_train and rank == 0)):
#             save_model(model, args.output_model_path + "-" + str(steps))

#         steps += 1
