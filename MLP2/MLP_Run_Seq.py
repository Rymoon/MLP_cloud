
"""
# NOTICE Finished Expr. Donnot modify this, ok?
Out:
    MLP_Run_Seq_xxx
        MLP0\\31.sd
        MLP1\\9.sd
        MLP2\\8.sd
            
"""
import os
from tqdm import trange
from math import ceil
import time

import sys
from random import random
from time import sleep

from tqdm.auto import tqdm, trange

from MLP_Log import Runner,loadConfig,getLogger,DBLogger,getDefaultRunSetting
from MLP_Log import GPU,CPU
from MLP_Data import BSDS500PatchDataset,StdTestPatchataset
from MLP_Net import MLP0,MLP1,MLP2
from MLP_Task import TrainTask


# prefetch dataset
train_dataset = BSDS500PatchDataset((13,13),(6,6),0.1,GPU,do_prefetch=True)
test_dataset = StdTestPatchataset((13,13),(6,6),0.1,GPU,n_max_patch_index=None,do_prefetch=True)


models = [MLP0().to(GPU),MLP1().to(GPU),MLP2().to(GPU)]
n_model = len(models)

# PLOT train setting
n_sample_per_epoch = len(test_dataset)
#n_sample_per_epoch = 128


n_max_epoch = ceil(len(train_dataset)/n_sample_per_epoch)
#n_max_epoch =2


for i in range(n_model):
    model = models[i]

    cfp,cfn = os.path.split(__file__)
    cfn = cfn.split('.')[0]
    cfn = "{}_{}.py".format(cfn,type(model).__name__)
    ss = os.path.join(cfp,cfn)
    runner = Runner(filestr=ss)

    log_path = os.path.join(runner.base_path,runner.base_name+'.log')
    logger = getLogger(log_path)

    db_path = os.path.join(runner.base_path,runner.base_name+'.db')
    dblogger = DBLogger(db_path)

    task = TrainTask(models[i],train_dataset,test_dataset,logger,dblogger,sz_batch = 64,n_max_epoch=n_max_epoch,n_sample_per_epoch = n_sample_per_epoch,n_epoch_per_save_state_dict=4,runner = runner)
    
    for i in range(8):
        task.run()

