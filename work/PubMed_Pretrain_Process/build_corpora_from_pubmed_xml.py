# -*- encoding: utf-8 -*-

#File    :   pubmed_parser_test.py
#Time    :   2020/12/17 10:15:13
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com

import pubmed_parser as pp
import os
from Seg_Sents_En import seg_sens
# import spacy
# nlp = spacy.load("en_core_sci_sm")
from tqdm import tqdm

path = r'/data/PubMed_articles'
file_count = 0
files = []


# 遍历整个文件夹，查找nxml文件
for path,dir_list,file_list in os.walk(path):
    for file_name in file_list:
        if '.nxml' in file_name:
            file_count += 1
            files.append(os.path.join(path, file_name))
print(file_count)


with open('corpora/pubmed_oa_noncm.txt','w',encoding='utf-8') as f:
    # 遍历整个文件夹，查找nxml文件
    for file in tqdm(files[:1000]):
        dicts_out = pp.parse_pubmed_paragraph(file, all_paragraph=False)
        if dicts_out:
            for dict in dicts_out:
                para = dict['text'].strip()
                if para:
                    sens = seg_sens(para)
                    for sen in sens:
                        f.write(sen+'\n')
                    f.write(para+'\n')

if __name__ == '__main__':
    pass