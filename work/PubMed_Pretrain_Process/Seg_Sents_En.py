# -*- encoding: utf-8 -*-

#File    :   Seg_Sents_En.py
#Time    :   2020/09/12 14:18:14
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com


import spacy
nlp = spacy.load("en_core_sci_sm")

def seg_sens(text):

    sentences = [str(sen) for sen in nlp(text).sents]

    indexs = ['1.','2.','3.','4.','5.','6.','7.','8.','9.']

    flag = 1
    while flag:
        flag = 0
        for i, sen in enumerate(sentences):
            if i > 0:
                sen_pre = sentences[i - 1]
            else:
                sen_pre = ''
            if i < (len(sentences) - 1):
                sen_next = sentences[i + 1]
            else:
                sen_next = ''

            # 带序号的
            if sen in indexs:
                sentences[i+1] = sen + ' ' + sen_next
                sentences.remove(sen)
                flag = 1
                break

            # 带符号的
            if sen[-3:] == "+/-":
                sentences[i + 1] = sen + ' ' + sen_next
                sentences.remove(sen)
                flag = 1
                break

            # 小写字母开头
            if 'a' <= sen[0] <= 'z':
                sentences[i-1] = sen_pre + ' ' + sen
                sentences.remove(sen)
                flag = 1
                break

            # 小于30字符
            if len(sen)<30:
                sentences[i - 1] = sen_pre + ' ' + sen
                sentences.remove(sen)
                flag = 1
                break

            # 括号不对称的
            if ")" in sen and (sen_pre.count("(") + sen_pre.count(")")) % 2 == 1:
                sentences[i - 1] = sen_pre + ' ' + sen
                sentences.remove(sen)
                flag = 1
                break
            
    return sentences

if __name__ == '__main__':
    pass