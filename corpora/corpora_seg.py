# import pkuseg
from tqdm import tqdm
import re

# pku_seg = pkuseg.pkuseg(model_name="medicine")

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
    pattern = re.compile(r'([,?:;\'"!()<>，。；’‘、！])')
    for part in parts:
        chars = pattern.split(part)
        chars = [w for w in chars if len(w.strip())>0]
        chars_list += chars

    return chars_list

with open('R_seg.txt','w',encoding='utf-8') as fw:
    with open('R.txt','r',encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line:
                fw.write(' '.join(seg_char(line)) + '\n')


