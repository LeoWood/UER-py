#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/5/6 20:07


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # df = pd.read_csv(r'D:\UCAS\Phd\Projects\201908CsciBERT\预训练实验\PubMed英文预训练模型\pubmed_bert_from_base_40gpus_conti.csv')
    df = pd.read_csv('/Users/leo/Work/项目工作/NSTL预训练/参数实验/学习率/pubmed_bert_from_base_200gpus_lr_1e-5.csv')
    # df.plot(x='steps',y='loss',kind='line')
    # plt.show()
    # print(df.iloc[1]['acc_mlm'])
    # df['acc_mlm'] = pd.to_numeric(df['acc_mlm'])
    print(df.head())
    df.plot(x='steps', y='acc_mlm', kind='line')
    plt.show()
