#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_data.py
@Time    :   2020/07/02 15:04:55
@Author  :   Leo Wood 
@Contact :   leowood@foxmail.com
@Desc    :   None
'''

import pandas as pd
from tqdm import tqdm


def read_data_from_file(file_in, file_out):
    with open(file_out, 'w', encoding='utf-8') as fw:
        fw.write("text_a\tlabel\n")
        with open(file_in, 'r', encoding='utf-8') as f:
            chars = []
            labels = []
            for line in tqdm(f.readlines()):
                line = line.strip()
                if not line:
                    if len(chars) > 0:
                        fw.write(' '.join(chars) + '\t' + ' '.join(labels) + '\n')
                        chars = []
                        labels = []

                if '\t' in line:
                    word = line.split('\t')[0]
                    label = line.split('\t')[1]
                    if len(word) > 1:
                        for w in word:
                            chars.append(w)
                            labels.append(label)
                    else:
                        chars.append(word)
                        labels.append(label)




if __name__ == '__main__':
    # read_data_from_file('data_origin/test.tsv', 'origin_train_data.tsv')
    # exit()
    read_data_from_file('data_origin/train.tsv', 'origin_train_data.tsv')
    df = pd.read_csv("origin_train_data.tsv", sep="\t")
    df = df.sample(frac=1, random_state=666).reset_index(drop=True)
    print(df.head())

    df_train = df[:10000]
    df_train.to_csv('uer_data/train.tsv', sep='\t', header=True, index=False)

    df_dev = df[10000:12000]
    df_dev.to_csv('uer_data/dev.tsv', sep='\t', header=True, index=False)

    df_test = df[12000:14000]
    df_test.to_csv('uer_data/test.tsv', sep='\t', header=True, index=False)
