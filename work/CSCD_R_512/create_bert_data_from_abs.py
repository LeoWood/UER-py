# -*- encoding: utf-8 -*-

#File    :   create_bert_data_from_abs.py
#Time    :   2020/12/22 20:57:29
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com

from Seg_Sents_Cn import Seg_Sents_Cn
from tqdm import tqdm

with open('../../corpora/R_512_bert.txt','w',encoding='utf-8') as fw:
    with open('../../corpora/R_512_mlm.txt','r',encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            sens = Seg_Sents_Cn(line)
            for sen in sens:
                fw.write(sen+'\n')
            fw.write('\n')



if __name__ == '__main__':
    pass