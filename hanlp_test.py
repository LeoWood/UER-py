#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/5/5 8:57

import time
import re
from tqdm import tqdm
# import hanlp
# cut = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
# tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)

from hanlp.common.trie import Trie

import hanlp

print('what what what??/')
time.sleep(3)
print("nononnonon")
for i in range(1000):
    print(i)
    time.sleep(1)

print(3/0)
exit()

# tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')
# text = 'NLP统计模型没有加规则，聪明人知道自己加。英文、数字、自定义词典统统都是规则。'
# print(tokenizer(text))
#
# trie = Trie()
# trie.update({'自定义': 'custom', '词典': 'dict', '聪明人': 'smart'})
#
#
# def split_sents(text: str, trie: Trie):
#     words = trie.parse_longest(text)
#     sents = []
#     pre_start = 0
#     offsets = []
#     for word, value, start, end in words:
#         if pre_start != start:
#             sents.append(text[pre_start: start])
#             offsets.append(pre_start)
#         pre_start = end
#     if pre_start != len(text):
#         sents.append(text[pre_start:])
#         offsets.append(pre_start)
#     return sents, offsets, words
#
#
# print(split_sents(text, trie))
#
#
# def merge_parts(parts, offsets, words):
#     items = [(i, p) for (i, p) in zip(offsets, parts)]
#     items += [(start, [word]) for (word, value, start, end) in words]
#     # In case you need the tag, use the following line instead
#     # items += [(start, [(word, value)]) for (word, value, start, end) in words]
#     return [each for x in sorted(items) for each in x[1]]
#
#
# tokenizer = hanlp.pipeline() \
#     .append(split_sents, output_key=('parts', 'offsets', 'words'), trie=trie) \
#     .append(tokenizer, input_key='parts', output_key='tokens') \
#     .append(merge_parts, input_key=('tokens', 'offsets', 'words'), output_key='merged')
#
# print(tokenizer(text))
# exit()
#
#
# a = []
# term_dict = {}
# with open('uer/utils/medical_terms/medical_terms.txt', 'r', encoding='utf-8') as f:
#     for line in f.readlines():
#         line = line.strip()
#         a.append(line)
#         term_dict[line.lower()] = 1
#
#
# max_num = max([len(line) for line in a])
# print('max_num:',max_num)



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
    # print(word_list)
    new_word_list = []
    term_labels = []
    N = len(word_list)
    k = max_num
    i = 0
    while i < N:
        if i <= N - k:
            j = k
            while j > 0:
                token_tmp = ''.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp,'！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    term_labels.append(1)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                new_word_list.append(word_list[i])
                term_labels.append(0)
                i += 1
        else:
            j = N - i
            while j > 0:
                token_tmp = ''.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp, '！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    term_labels.append(1)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                new_word_list.append(word_list[i])
                term_labels.append(0)
                i += 1
    return new_word_list,term_labels

if __name__ == '__main__':

    pipeline = hanlp.pipeline() \
        .append(cut, output_key='tokens') \
        .append(tagger, output_key='part_of_speech_tags')

    with open('corpora/R_test.txt','r',encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    print(len(lines))
    t0 = time.time()
    cut_lines = cut(lines)
    print(len(cut_lines))
    t1 = time.time()
    print('cut用时：', t1 - t0)
    tags = tagger(cut_lines)
    t2 = time.time()
    print('tag用时：', t2 - t1)
    p = pipeline(lines)
    t3= time.time()
    print('pipeline用时：', t3 - t2)

    # for line in tqdm(lines):
    #     p = pipeline(line)


    batch = 1000
    nums = int(len(lines) / batch)
    for i in tqdm(range(nums)):
        p = pipeline(lines[i * batch: (i + 1) * batch])

        p = pipeline(lines[nums * batch:len(lines)])

    t4 = time.time()
    print('batch用时：', t4 - t3)

    exit()

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


