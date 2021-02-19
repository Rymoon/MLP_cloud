import os
from tqdm import trange
from math import ceil
import time

import sys
from random import random
from time import sleep

from tqdm.auto import tqdm, trange

import torch
import numpy as np
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as computePsnr

import matplotlib
from matplotlib import pyplot as plt

from MLP_Log import Runner,loadConfig,getLogger,DBLogger,getDefaultRunSetting
from MLP_Log import GPU,CPU
from MLP_Data import BSDS500PatchDataset,StdTestPatchataset,RestoreDataset,ten2img
from MLP_Net import MLP0,MLP1,MLP2
from MLP_Task import TrainTask


# prefetch dataset
test_dataset = StdTestPatchataset((13,13),(6,6),0.1,GPU,n_max_patch_index=None,do_prefetch=True)

model = MLP0().to(GPU)
sd_path = "D:\\Workspace\\ADMM_Net\\MyHumbleADMM2\\MLP2\\MLP_Run_Seq3_MLP0\\16.sd"
sd= torch.load(sd_path)
model.load_state_dict(sd)

restore_dataset = RestoreDataset(model,test_dataset,sz_batch = 128)
restore_dataset.test()



for k in trange(len(restore_dataset.crop_dataset.image_dataset)):
    # BUG FIXME 只显示第一张图？？
    tr,sa,re = restore_dataset.showExample(k)

    img_pair=(ten2img(tr),ten2img(sa),ten2img(re))
    img_strip = np.concatenate(img_pair,axis = 1)

    psnr_noi = computePsnr(img_pair[0],img_pair[1])
    psnr_res = computePsnr(img_pair[0],img_pair[2])
    plt.title("psnr_noi,res = {:.3f},{:.3f}".format(psnr_noi,psnr_res))
    plt.imshow(img_strip )
    plt.savefig('{}.png'.format(k))
# plt.show()