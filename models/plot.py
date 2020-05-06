#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/5/6 20:07


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('cscd_R_based_on_google_zh_400000-500000.csv')
    df.plot(x='steps',y='loss',kind='line')
    plt.show(
    df.plot(x='steps', y='acc', kind='line')
    plt.show()
