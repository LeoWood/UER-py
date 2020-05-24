# -*- encoding:utf-8 -*-
"""
Test tokenizer with given vocab
"""

import sys
import os

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.tokenizer import BertTokenizer
import argparse



if __name__ == '__main__':

    vocab_path = r"E:\LiuHuan\Projects\UER-py\models\google_zh_vocab.txt"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vocab_path",default=vocab_path)
    args = parser.parse_args()

    tokenizer_1 = BertTokenizer(args)

    args.vocab_path = r"E:\LiuHuan\Projects\UER-py\corpora\cscd_r_vocab.txt"
    tokenizer_2 = BertTokenizer(args)

    args.vocab_path = r"E:\LiuHuan\Projects\bert-vocab-builder\cscd_r_vocab.txt"
    tokenizer_3 = BertTokenizer(args)

    while True:
        text = input()
        text = text.lower()
        print("google_zh_vocab")
        print(tokenizer_1.tokenize(text))
        print("uer_create_vocab:")
        print(tokenizer_2.tokenize(text))
        print("google_create_vocab:")
        print(tokenizer_3.tokenize(text))


