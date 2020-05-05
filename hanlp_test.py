#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/5/5 8:57

import time
import re
import hanlp
cut = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)

a = []
term_dict = {}
with open('uer/utils/MedicalTerms.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        a.append(line)
        term_dict[line.lower()] = 1

max_num = max([len(line) for line in a])
print('max_num:',max_num)



def seg_char(sent):
    """
    把句子按字分开，不破坏英文及数字结构
    """
    # 按中文汉字分割
    pattern = re.compile(r'([\u4e00-\u9fa5])')
    parts = pattern.split(sent)
    parts = [w for w in parts if len(w.strip())>0]

    # 按英文标点符号分割
    chars_list = []
    pattern = re.compile(r'([-,?:;\'"!`()<>，。@#￥&*~；/=’‘、！])')
    for part in parts:
        chars = pattern.split(part)
        chars = [w for w in chars if len(w.strip())>0]
        chars_list += chars

    return chars_list

def max_match(txt, ano_dict, max_num):
    word_list = seg_char(txt) # 中文单字切割，保留英文和数字
    new_word_list = []
    N = len(word_list)
    k = max_num
    i = 0
    while i < N:
        if i <= N - k:
            j = k
            while j > 0:
                token_tmp = word_list[i:i + j]
                # print(token_tmp)
                if token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp,'！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                new_word_list += word_list[i]
                i += 1
        else:
            j = N - i
            while j > 0:
                token_tmp = word_list[i:i + j]
                # print(token_tmp)
                if token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp, '！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                new_word_list += word_list[i]
                i += 1
    return new_word_list

if __name__ == '__main__':
    while True:
        text = input()
        t0 = time.time()
        # text = "怎么说呢，“这已经不是文学了，而是改造犯人的刑罚”，至少每个自己（并非每个人）都会有自己的地下室，幽暗深邃，时而歇斯底里地宣泄而出，又晦明晦暗；有理性，有规律，但我的独立与自由意志更为优先，由此人类也就是经常忘恩负义了，既是创造，又会毁灭，反而担心的是眼前没有路；观念与文明的面具"
        print(tagger(cut(text)))
        t1 = time.time()
        print('hanlp用时：', t1-t0)
        print(max_match(text, term_dict, max_num))
        t2 = time.time()
        print('max_match用时：', t2-t1)
        print('总用时：', t2-t0)
        time.sleep(1)


