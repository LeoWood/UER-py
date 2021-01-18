# -*- encoding: utf-8 -*-

#File    :   stats_instances.py
#Time    :   2021/01/15 20:20:59
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com

import pickle

if __name__ == '__main__':
    ins_count = 0
    with open('/work1/zzx6320/lh/Projects/UER-py/corpora/pubmed_oa_noncm.pt','rb') as f:
        i = 0
        while True:
            try:
                instance = pickle.load(f)
                ins_count += len(instance)
            except EOFError:
                break
            i += 1
    print("instances: ", ins_count)