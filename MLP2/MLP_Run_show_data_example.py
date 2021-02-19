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
from MLP_Data import BSDS500PatchDataset,StdTestPatchataset,ten2img
from MLP_Net import MLP0,MLP1,MLP2
from MLP_Task import TrainTask


# prefetch dataset
train_dataset = BSDS500PatchDataset((13,13),(6,6),0.1,GPU,do_prefetch=True)
test_dataset = StdTestPatchataset((13,13),(6,6),0.1,GPU,n_max_patch_index=None,do_prefetch=True)

for k in trange(8):
    # BUG FIXME 只显示第一张图？？
    trimage= train_dataset.showExample(0,0)
    saimage= train_dataset.showExample(0,1)


    img_pair=(ten2img(trimage),ten2img(saimage))
    img_strip = np.concatenate(img_pair,axis = 1)
    psnr_noi = computePsnr(img_pair[0],img_pair[1])
    plt.title("psnr_noi = {:.3f}".format(psnr_noi))
    plt.imshow(img_strip )
    #plt.show()
    plt.savefig('show_{}.png'.format(k))