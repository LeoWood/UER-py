# -*- encoding:utf-8 -*-
"""
Test tokenizer with given vocab
"""

import sys
import os

uer_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(uer_dir)

from uer.utils.vocab import Vocab
from uer.utils.tokenizer import WordpieceTokenizer
from uer.utils.tokenizer import BertTokenizer
import argparse



if __name__ == '__main__':
    vocab = Vocab()
    print("input vocab path:")
    vocab_path = input()
    vocab.load(vocab_path)
    tokenizer = WordpieceTokenizer(vocab)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--vocab_path",default=vocab_path)
    args = parser.parse_args()

    tokenizer_bert = BertTokenizer(args)
    while True:
        text = input()
        text = text.lower()
        print("wordpiece tokens:")
        print(tokenizer.tokenize(text))
        print("bert tokens:")
        print(tokenizer_bert.tokenize(text))


