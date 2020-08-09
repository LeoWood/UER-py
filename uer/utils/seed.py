import random
import os
import numpy as np
import torch

def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ## 增加两个seed
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    ##
    torch.backends.cudnn.deterministic = True

