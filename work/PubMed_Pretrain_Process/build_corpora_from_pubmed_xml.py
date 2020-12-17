# -*- encoding: utf-8 -*-

#File    :   pubmed_parser_test.py
#Time    :   2020/12/17 10:15:13
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com

import pubmed_parser as pp
import os
from .Seg_Sents_En import seg_sens
from tqdm import tqdm

path = ''
file_count = 0

with open('pubmed_oa_noncm.txt','w',encoding='utf-8') as f:
    # 遍历整个文件夹，查找nxml文件
    for path,dir_list,file_list in os.walk(path):  
        for file_name in file_list:
            file_count += 1
print(file_count)
exit()

path = ''
file_count = 0

with open('pubmed_oa_noncm.txt','w',encoding='utf-8') as f:
    # 遍历整个文件夹，查找nxml文件
    for path,dir_list,file_list in os.walk(path):  
        for file_name in tqdm(file_list):
            file_count += 1
            file = os.path.join(path, file_name)
            dicts_out = pp.parse_pubmed_paragraph(file, all_paragraph=False)
            for dict in dicts_out:
                para = dict['text']
                sens = seg_sens(para)
                for sen in sens:
                    f.write(sen+'\n')
                f.write('\n')
print(file_count)

if __name__ == '__main__':
    pass