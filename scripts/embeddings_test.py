#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/5/11 15:34

import torch
import argparse
import sys
import os
import numpy as np

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)
print(sys.path)

from uer.layers.embeddings import BertEmbedding, WordEmbedding, CscibertEmbedding


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--emb_size", type=int, default=768, help="Embedding dimension.")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
args = parser.parse_args()

embedding = BertEmbedding(args,30000)

batch_size = 32

src = torch.LongTensor(np.random.randint(30000,size=(batch_size,128)))
# print("src:",src)

seg = torch.LongTensor(np.zeros((batch_size,128)))
# print("seg",seg)

print(embedding)

embs = embedding(src,seg)

print("embs:",embs)

print("embs_size:",embs.shape)


if __name__ == '__main__':
    pass