import os
import cv2
import torch
import numpy as np
import skimage.metrics

from natsort import natsorted

import lpips
import pyiqa

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'

## For perceptual metrics
metric_psnr = pyiqa.create_metric('psnr', device=torch.device('cuda'))
metric_ssim = pyiqa.create_metric('ssim', device=torch.device('cuda'))
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
metric_nima = pyiqa.create_metric('nima-vgg16-ava', device=torch.device('cuda'))
metric_niqe = pyiqa.create_metric('niqe', device=torch.device('cuda'))
metric_clip = pyiqa.create_metric('clipiqa', device=torch.device('cuda'))


def get_psnr(im1, im2):
    im1_th = torch.from_numpy((im1.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    im2_th = torch.from_numpy((im2.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)

    return metric_psnr(im1_th, im2_th)


def get_ssim(im1, im2):
    im1_th = torch.from_numpy((im1.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    im2_th = torch.from_numpy((im2.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)

    return metric_ssim(im1_th, im2_th)


def get_lpips(im1, im2):
    im1_th = torch.from_numpy((im1.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    im2_th = torch.from_numpy((im2.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)

    im1_th = im1_th * 2 - 1
    im2_th = im2_th * 2 - 1

    return loss_fn_alex.forward(im1_th, im2_th)

def get_nima(im):
    im = torch.from_numpy((im.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    return metric_nima(im)

def get_clip(im):
    im = torch.from_numpy((im.astype(np.float32) / 255.).transpose(2,0,1)).unsqueeze(0)
    return metric_clip(im)

## The data folders (gt root) will like：
## gt root
##      |__ 
##          000000_250 (index_amplify ratio)
##                  |__ long.png (GT), short.npy
##          000001_100
##                  |__ long.png (GT), short.npy


gt_root = '' 
pred_root = '' # The output of the method
new_root = '' # During evaluation, the predicted result and its corresponding GT will be presented together along with the amplification ratio.

## If it is too laborious to execute this evaluation, dont worry, we have released all our predicted results on OneDrive.

gt_paths = natsorted(os.listdir(gt_root))
pred_paths = natsorted(os.listdir(pred_root))

PSNR = 0
SSIM = 0
LPIPS = 0
NIMA = 0
CLIP = 0

ratios = [100, 250, 300]

for ratio in ratios:
    num = 0
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        if '_%d' % ratio in gt_path:
            gt = cv2.cvtColor(cv2.imread(os.path.join(gt_root, gt_path, 'long.png')), cv2.COLOR_BGR2RGB)
            pred = cv2.cvtColor(cv2.imread(os.path.join(pred_root, pred_path)), cv2.COLOR_BGR2RGB)

            PSNR += get_psnr(pred, gt)
            SSIM += get_ssim(pred, gt)
            LPIPS += get_lpips(pred, gt)
            NIMA+= get_nima(pred)
            CLIP += get_clip(pred)

            num += 1

            os.mkdir(os.path.join(new_root, gt_path))
            cv2.imwrite(os.path.join(new_root, gt_path, pred_path.replace('.png', '_fake.png')), cv2.cvtColor(pred, cv2.COLOR_BGR2RGB))
            cv2.imwrite(os.path.join(new_root, gt_path, pred_path.replace('.png', '_real.png')), cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))

    PSNR = PSNR / num
    SSIM = SSIM / num
    LPIPS = LPIPS / num
    NIMA = NIMA / num
    CLIP = CLIP / num

    print('PSNR: %.4f'%PSNR)
    print('SSIM: %.4f'%SSIM)
    print('LPIPS: %.4f'%LPIPS)
    print('NIMA: %.4f'%NIMA)
    print('CLIP: %.4f'%CLIP)
    print('----------------Ratio: %d Finished----------------'%ratio)