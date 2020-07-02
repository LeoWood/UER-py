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

def read_data_from_file(file_in,file_out):
    with open(file_out,'w',encoding='utf-8') as fw:
        fw.write("text_a\tlabel\n")
        with open(file_in,'r',encoding='utf-8') as f:
            chars = []
            labels = []
            for line in f.readlines():
                line = line.strip()
                if not line:
                    if len(chars) > 0:
                        fw.write(' '.join(chars) + '\t' + ' '.join(labels) + '\n')
                if '\t' in line:
                    word = line.splie('\t')[0]
                    label = line.splie('\t')[1]
                    if len(word) > 1:
                        for w in word:
                            chars.append(w)
                            labels.append(label)
                    else:
                        chars.append(word)
                        labels.append(label)


if __name__ == '__main__':
    read_data_from_file('train.tsv','origin_train_data.tsv')
    df = pd.read_csv("origin_train_data.tsv",sep="\t")
    df = df.sample(frac=1, random_state=666).reset_index(drop=True)
    print(df.head())

    df_train = df[:60000]
    df_train.to_csv('uer_data/train.tsv', sep='\t', header=True, index=False)

    df_dev = df[60000:80000]
    df_dev.to_csv('uer_data/dev.tsv', sep='\t', header=True, index=False)

    df_test = df[80000:100000]
    df_test.to_csv('uer_data/test.tsv', sep='\t', header=True, index=False)
   

