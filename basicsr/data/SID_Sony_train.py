import os
import glob
import torch
import random
import cv2
import numpy as np
from pathlib import Path
import torch.utils.data as data
import scipy.stats as st

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
class SonyDataset_train(data.Dataset):
    def __init__(self):
        super(SonyDataset_train,self).__init__()

        self.root_path = Path('/disk1/wenqiang/Documents/data/LLIE/SID/Sony/SID_train_crop512_stride512')
        self.folder_list = os.listdir(self.root_path)
        self.img_num = len(self.folder_list)

    def random_flip(self, img_list):
        if random.random() > 0.5:
            # horizontal
            for i, img in enumerate(img_list):
                img_list[i] = cv2.flip(img, 1)
            
        if random.random() > 0.5:
            # vertical
            for i, img in enumerate(img_list):
                img_list[i] = cv2.flip(img, 0)
        
        return img_list

    def __getitem__(self, index):
        short_img = np.load(os.path.join(self.root_path, self.folder_list[index], 'short.npy'))
        long_img = cv2.imread(os.path.join(self.root_path, self.folder_list[index], 'long.png'))

        # short_img = cv2.cvtColor(short_img, cv2.COLOR_BGR2RGB)
        long_img = cv2.cvtColor(long_img, cv2.COLOR_BGR2RGB)

        [short_img, long_img] = self.random_flip([short_img, long_img])
        short_img = np.clip(short_img, 0, 1)
        long_img = np.clip(long_img / 255., 0, 1)

        short_img = torch.from_numpy(short_img.transpose(2,0,1).astype(np.float32)) * 2 - 1
        long_img = torch.from_numpy(long_img.transpose(2,0,1).astype(np.float32)) * 2 - 1

        return {'short_img': short_img, 'long_img': long_img}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'SID Sony Train Dataset'