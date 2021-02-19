import os
from tqdm import trange
from math import ceil

from MLP_Log import Runner,loadConfig,getLogger,DBLogger
from MLP_Log import GPU,CPU
from MLP_Data import BSDS500PatchDataset,StdTestPatchataset
from MLP_Net import MLP0
from MLP_Task import TrainTask

try:
    config = loadConfig(__file__)
except FileNotFoundError as e:
    print(e)
    config = {}
runner = Runner(filestr=__file__)

if "log_path" not in config:
    log_path = os.path.join(runner.base_path,runner.base_name+'.log')
else:
    log_path = config["log_path"]
logger = getLogger(log_path)

if "db_path" not in config:
    db_path = os.path.join(runner.base_path,runner.base_name+'.db')
else:
    db_path =config["db_path"]

dblogger = DBLogger(db_path)

train_dataset = BSDS500PatchDataset((13,13),(6,6),0.1,GPU)
test_dataset = StdTestPatchataset((13,13),(6,6),0.1,GPU)
model = MLP0().to(GPU)


# PLOT train setting
n_sample_per_epoch = len(test_dataset)
#n_sample_per_epoch = 128
n_max_epoch = ceil(len(train_dataset)/n_sample_per_epoch)
task = TrainTask(model,train_dataset,test_dataset,logger,dblogger,sz_batch = 128,n_max_epoch=n_max_epoch,n_sample_per_epoch = n_sample_per_epoch,n_epoch_per_save_state_dict=4,runner = runner)

for i in trange(10):
    task.run()
    
    