# -*- encoding: utf-8 -*-

#File    :   pubmed_parser_test.py
#Time    :   2020/12/17 10:15:13
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com

import pubmed_parser as pp
import os
# from Seg_Sents_En import seg_sens
# import spacy
# nlp = spacy.load("en_core_sci_sm")
from tqdm import tqdm
import os
import sys 
os.chdir(sys.path[0])

path = r'/data/PubMed_articles'
file_count = 0
files = []


# 遍历整个文件夹，查找nxml文件
for path,dir_list,file_list in os.walk(path):
    for file_name in file_list:
        if '.nxml' in file_name:
            file_count += 1
            files.append(os.path.join(path, file_name))
            file = os.path.join(path, file_name)
            print(file)
            with open('pubmed_oa_paras.txt','w',encoding='utf-8') as f:
                dicts_out = pp.parse_pubmed_paragraph(file, all_paragraph=False)
                if dicts_out:
                    for dict in dicts_out:
                        print(dict)
                        para = dict['text'].strip()
                        if para:
                            f.write(para+'\n')
                    f.write('\n')
                    if file_count == 1000:
                        exit()

print(file_count)
# exit()
#
# path = ''
# file_count = 0


with open('pubmed_oa_paras.txt','w',encoding='utf-8') as f:
    # 遍历整个文件夹，查找nxml文件
    for file in tqdm(files):
        print(file)
        dicts_out = pp.parse_pubmed_paragraph(file, all_paragraph=False)
        for dict in dicts_out:
            para = dict['text']
            f.write(para+'\n')
        f.write('\n')
        exit()

if __name__ == '__main__':
    pass
