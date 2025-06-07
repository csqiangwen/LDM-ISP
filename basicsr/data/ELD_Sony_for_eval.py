import os
import glob
import torch
import random
import cv2
import numpy as np
from pathlib import Path
import torch.utils.data as data
from natsort import natsorted

## from StableSR
from basicsr.utils.registry import DATASET_REGISTRY


## The data folders (gt root) will like：
## gt root
##      |__ 
##          000000_250 (index_amplify ratio)
##                  |__ long.png (GT), short.npy
##          000001_100
##                  |__ long.png (GT), short.npy

    
@DATASET_REGISTRY.register(suffix='basicsr')
class ELDDataset_eval(data.Dataset):
    def __init__(self):
        super(ELDDataset_eval,self).__init__()

        self.root_path = Path('/disk1/wenqiang/Documents/data/LLIE/ELD/SonyA7S2_group')
        self.folder_list = natsorted(os.listdir(self.root_path))
        self.img_num = len(self.folder_list)

    def __getitem__(self, index):
        short_img = np.load(os.path.join(self.root_path, self.folder_list[index], 'short.npy'))
        long_img = cv2.imread(os.path.join(self.root_path, self.folder_list[index], 'long.png'))

        # short_img = cv2.cvtColor(short_img, cv2.COLOR_BGR2RGB)
        long_img = cv2.cvtColor(long_img, cv2.COLOR_BGR2RGB)

        short_img = np.clip(short_img, 0, 1)
        long_img = np.clip(long_img / 255., 0, 1)

        short_img = torch.from_numpy(short_img.transpose(2,0,1).astype(np.float32)) * 2 - 1
        long_img = torch.from_numpy(long_img.transpose(2,0,1).astype(np.float32)) * 2 - 1

        return {'short_img': short_img, 'long_img': long_img, 'folder_name': self.folder_list[index]}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'ELD Sony Eval Dataset'
