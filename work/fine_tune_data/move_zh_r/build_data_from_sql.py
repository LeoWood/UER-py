#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   build_data_from_sql.py
@Time    :   2020/07/02 10:28:02
@Author  :   Leo Wood 
@Contact :   leowood@foxmail.com
@Desc    :   None
'''
from pySql import pySql
from Seg_Sents_Cn import Seg_Sents_Cn
import json
import pandas as pd
from tqdm import tqdm


# 提取语步标签
def extract_move_from_abs(abst, f):
    abst = abst.split("方法:")
    assert len(abst) == 2

    objective = abst[0][3:]

    abst = abst[1]

    abst = abst.split("结果:")
    assert len(abst) == 2

    method = abst[0]

    abst = abst[1]

    abst = abst.split("结论:")
    assert len(abst) == 2

    result = abst[0]

    conclusion = abst[1]

    i = 0
    for move in [objective, method, result, conclusion]:
        sens = Seg_Sents_Cn(move)
        [f.write(str(i) + '\t' + sen + '\n') for sen in sens]
        i += 1

def extract_move_from_abs_1(abst, f):
    abst = abst.split("方法：")
    assert len(abst) == 2

    objective = abst[0][3:]

    abst = abst[1]

    abst = abst.split("结果：")
    assert len(abst) == 2

    method = abst[0]

    abst = abst[1]

    abst = abst.split("结论：")
    assert len(abst) == 2

    result = abst[0]

    conclusion = abst[1]

    i = 0
    for move in [objective, method, result, conclusion]:
        sens = Seg_Sents_Cn(move)
        [f.write(str(i) + '\t' + sen + '\n') for sen in sens]
        i += 1


if __name__ == '__main__':
    for f in ['train.tsv','dev.tsv','test.tsv']:
        df = pd.read_csv(f, sep='\t')
        df['label'] = df['label'].astype('int')
        df.to_csv(f, sep='\t', header=True, index=False)
    # print(df.head())
    exit()

    # ## 读取数据库信息
    # with open('db_info.json', 'r', encoding='utf-8') as f:
    #     db_info = json.load(f)
    # db_info = db_info['cscd']
    # db_server = pySql(ip=db_info['ip'], user=db_info['user'], pwd=db_info['pwd'], db=db_info['db'])
    #
    # sql = "SELECT paper_id,abstract FROM [CSCD].[dbo].[article_info] where abstract like '目的%' and classification like 'R%'"
    #
    # # 原始数据
    # df = db_server.read_sql(sql)
    # df.to_csv("origin_data_from_sql.csv", encoding="utf_8_sig")
    #
    # df = pd.read_csv("origin_data_from_sql.csv", encoding="utf_8_sig")
    # df = df.sample(frac=1, random_state=666).reset_index(drop=True)
    #
    # # 全部数据提取语步，放入文件
    # with open('all_data.tsv', 'w', encoding='utf-8') as f:
    #     for j in tqdm(range(len(df))):
    #         abst = df.iloc[j]['abstract']
    #         if "。" in abst:
    #             if "方法:" in abst:
    #                 try:
    #                     extract_move_from_abs(abst, f)
    #                 except:
    #                     print(abst)
    #
    #             if "方法：" in abst:
    #                 try:
    #                     extract_move_from_abs_1(abst, f)
    #                 except:
    #                     print(abst)

    # 重新读取，划分train,dev,test
    df = pd.read_csv('all_data.tsv', sep='\t', names=['label', 'text_a'])

    df = df.sample(frac=1,random_state=666).reset_index(drop=True)
    # df['label'] = df['label'].astype('int')

    df_train = df[:10000]
    df_train.to_csv('train.tsv', sep='\t', header=True, index=False)

    df_dev = df[10000:12000]
    df_dev.to_csv('dev.tsv', sep='\t', header=True, index=False)

    df_test = df[12000:14000]
    df_test.to_csv('test.tsv', sep='\t', header=True, index=False)
