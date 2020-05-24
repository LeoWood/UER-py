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


if __name__ == '__main__':
    vocab = Vocab()
    print("input vocab path:")
    vocab_path = input()
    vocab.load(vocab_path)
    tokenizer = WordpieceTokenizer(vocab)
    while True:
        text = input()
        text = text.lower()
        print(tokenizer.tokenize(text))


