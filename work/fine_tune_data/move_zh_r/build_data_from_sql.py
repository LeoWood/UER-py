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

# 提取语步标签
def extract_move_from_abs(abst,f):
    abst = abst.split("方法：")

    objective = abst[0][3:]
    assert len(abst) == 2

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
    for move in [object,method,result,conclusion]:
        sens = Seg_Sents_Cn(move)
        [f.write(str(i) + '\t' + sen + '\n') for sen in sens]
        i += 1




if __name__ == '__main__':
   ## 读取数据库信息
    with open('db_info.json','r',encoding='utf-8') as f:
        db_info = json.load(f)
    db_info = db_info['cscd']
    db_server = pySql(ip=db_info['ip'], user=db_info['user'], pwd = db_info['pwd'], db = db_info['db'])

    sql = "SELECT paper_id,abstract FROM [CSCD].[dbo].[article_info] where abstract like '目的:%' and classification like 'R%'"

    # 原始数据
    df = db_server.read_sql(sql)
    df.to_csv("origin_data_from_sql.csv",encoding="utf_8_sig")

    df = df.sample(frac=1,random_state=666).reset_index(drop=True)

    # 全部数据提取语步，放入文件
    with open('all_data.tsv','w',encoding='utf-8') as f:
        for data in df:
            extract_move_from_abs(data['abstract'],f)
    
    # 重新读取，划分train,dev,test
    df = pd.read_csv('all_data.tsv', sep='\t', names=['label', 'text_a'])

    df = df.sample(frac=1).reset_index(drop=True)

    df_train = df[:60000]
    df_train.to_csv('train.tsv', sep='\t', header=True, index=False)

    df_dev = df[60000:80000]
    df_dev.to_csv('dev.tsv', sep='\t', header=True, index=False)

    df_test = df[80000:100000]
    df_test.to_csv('test.tsv', sep='\t', header=True, index=False)






        

