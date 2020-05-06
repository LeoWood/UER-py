#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/5/6 9:32


if __name__ == '__main__':

    # 初始terms，包含MeSH和MedicalKG
    with open('medical_terms/MedicalTerms(MeSH+Meddical_KG).txt','r',encoding='utf-8') as f:
        mesh_medkg_terms = [line.strip() for line in f.readlines() if line.strip()]
        print('mesh_medkg_terms:',mesh_medkg_terms[:100])


    # CSCD医学论文关键词
    with open('medical_terms/Med_Keywords.txt','r',encoding='utf-8') as f:
        med_keywords = [line.strip().lower() for line in f.readlines() if line.strip()]
        print('med_keywords:',med_keywords[:100])

    # wiki中文词表
    with open('medical_terms/wiki_word_vocab.txt','r',encoding='utf-8') as f:
        wiki_vocabs = [line.strip().split()[0] for line in f.readlines() if line.strip().split()[0]]
        print('wiki_vocabs:',wiki_vocabs[:100])

    # CNDbpedia
    with open('medical_terms/CnDbpedia.spo','r',encoding='utf-8') as f:
        cn_dbpedia = [line.strip().split()[0] for line in f.readlines() if line.strip().split()[0]]
        print('cn_dbpedia:',cn_dbpedia[:100])

    # HowNet
    with open('medical_terms/HowNet.spo','r',encoding='utf-8') as f:
        hownet = [line.strip().split()[0] for line in f.readlines() if line.strip().split()[0]]
        print('hownet:',hownet[:100])

    final_terms = list(set(mesh_medkg_terms))
    print('初始大小：',len(final_terms))

    # 加入关键词
    for word in med_keywords:
        if len(word)<=50 and word not in final_terms:
            final_terms.append(word)
    print('加入关键词:',len(final_terms))


    # 去掉wiki中文词表
    for word in wiki_vocabs:
        if word in final_terms:
            final_terms.remove(word)
    print('去掉去掉wiki中文词表:',len(final_terms))


    # 去掉CNDbpedia
    for word in cn_dbpedia:
        if word in final_terms:
            final_terms.remove(word)
    print('去掉CNDbpedia:',len(final_terms))


    # 去掉HowNet
    for word in hownet:
        if word in final_terms:
            final_terms.remove(word)
    print('去掉HowNet:',len(final_terms))

    with open('medical_terms.txt','w',encoding='utf-8') as f:
        [f.write(term + '\n') for term in final_terms]





