# -*- encoding: utf-8 -*-

#File    :   pubmed_parser_test.py
#Time    :   2020/12/17 10:15:13
#Author  :   Leo Wood 
#Contact :   leowood@foxmail.com

import pubmed_parser as pp
import os
from Seg_Sents_En import seg_sens
from tqdm import tqdm
from multiprocessing import Pool


def worker(proc_id, start, end, files):
    print("Worker %d is building corpus ... " % proc_id)
    with open("corpora/pubmed_oa_noncm-"+ str(proc_id) + ".txt",'w',encoding='utf-8') as f:
        for file in tqdm(files[start:end]):
            dicts_out = pp.parse_pubmed_paragraph(file, all_paragraph=False)
            if dicts_out:
                for dict in dicts_out:
                    para = dict['text'].strip()
                    if para:
                        sens = []
                        try:
                            sens = seg_sens(para)
                        except Exception as e:
                            pass
                        for sen in sens:
                            f.write(sen+'\n')
                        f.write('\n')

# def merge_dataset(dataset_path, workers_num):
#     # Merge datasets.
#     f_writer = open(dataset_path, "wb")
#     for i in range(workers_num):
#         tmp_dataset_reader = open("dataset-tmp-"+str(i)+".pt", "rb")
#         while True:
#             tmp_data = tmp_dataset_reader.read(2^20)
#             if tmp_data:
#                 f_writer.write(tmp_data)
#             else:
#                 break
#         tmp_dataset_reader.close()
#         os.remove("dataset-tmp-"+str(i)+".pt")
#     f_writer.close()


if __name__ == '__main__':

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

    # 设置进程数
    workers_num = 10
    
    pool = Pool(workers_num)
    for i in range(workers_num):
        start = i * file_count // workers_num
        end = (i+1) * file_count // workers_num
        pool.apply_async(func=worker, args=[i, start, end, files])
    pool.close()
    pool.join()