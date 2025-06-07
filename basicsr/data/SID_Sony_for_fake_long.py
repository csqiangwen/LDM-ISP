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
from natsort import natsorted


@DATASET_REGISTRY.register(suffix='basicsr')
class SonyDataset_train(data.Dataset):
    def __init__(self):
        super(SonyDataset_train, self).__init__()

        self.root_path = Path('/disk1/wenqiang/Documents/data/LLIE/SID/Sony/SID_val_crop512_stride512')
        self.folder_list = natsorted(os.listdir(self.root_path))
        self.img_num = len(self.folder_list)

    def __getitem__(self, index):
        short_img = np.load(os.path.join(self.root_path, self.folder_list[index], 'short.npy'))

        short_img = np.clip(short_img, 0, 1)

        short_img = torch.from_numpy(short_img.transpose(2,0,1).astype(np.float32)) * 2 - 1

        fake_long_npy = np.load(os.path.join(self.root_path, self.folder_list[index], 'fake_long.npy'))

        return {'short_img': short_img, 'short_path': os.path.join(self.root_path, self.folder_list[index], 'short.npy'),
                'fake_long_npy': fake_long_npy}

    def __len__(self):
        return self.img_num

    def name(self):
        return 'SID Sony Dataset'