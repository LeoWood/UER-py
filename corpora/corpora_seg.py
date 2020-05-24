import pkuseg
from tqdm import tqdm

pku_seg = pkuseg.pkuseg(model_name="medicine")

with open('R_seg.txt','w',encoding='utf-8') as fw:
    with open('R.txt','r',encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line:
                fw.write(' '.join(pku_seg.cut(line)) + '\n')


