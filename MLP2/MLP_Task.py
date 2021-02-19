from MLP_Log import Runner,getTimestr,DBLogger,CPU,GPU
from MLP_Data import getPsnr

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import dataloader

from typing import Union

from pprint import pprint,pformat

import os

from math import sqrt,ceil,floor
from tqdm import tqdm
from RenNet.RyCore import getDebugOut

from RenNet.RyLog import DB
from torch import nn


class Task:
    pass

class TrainTask(Task):
    def __init__(self,model:nn.Module,train_dataset,test_dataset,logger:getDebugOut,dblogger:DBLogger,*,sz_batch,n_max_epoch,n_sample_per_epoch,n_epoch_per_save_state_dict=1,save_folder = None,startfrom_path = None,runner:Runner = None,taskstr_short = ''):
        ## task
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.logger = logger
        self.dblogger = dblogger
        self.n_sample_per_epoch = n_sample_per_epoch
        self.n_epoch_per_save_state_dict = n_epoch_per_save_state_dict
        self.sz_batch = sz_batch

        self.isFinished = False

        self.n_max_epoch = n_max_epoch

        self.pabr = None

        ## task description
        self.taskstr_short = taskstr_short

        ## output
        if save_folder is None:
            assert runner is not None
            save_folder = os.path.join(runner.base_path,runner.base_name)
            
        self.save_folder = save_folder
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

        ## model-optim
        self.model = model
        self.func_loss = nn.MSELoss()

        ## reset
        self._reset(startfrom_path)
    
    @property
    def startfrom_path(self):
        return self._startfrom_path
    
    @startfrom_path.setter
    def startfrom_path(self,v):
        self.startfrom_path_raw = v
        v2 = v
        if v2 is not None:
            runindex_path =  self.fromRunindexLink(v2)
            if runindex_path is not None:
                v2 = runindex_path
        self._startfrom_path = v2



    def forward(self):
        self.dblogger.forward()

    def reset(self,startfrom_path = None):
        """
        trained-model retained.

        self.startfrom = runindex_link

        runindex forward 1.

        reset counters.

        reset optimizer.

        isFinished <- False.

        """
        self._reset(startfrom_path)
        self.forward()
        self.isFinished = False

    def _reset(self,startfrom_path = None):
        """
        Now runindex is the same as the one during last-traing.
        """
        ## optim
        self.optimizer = SGD(self.model.parameters(),momentum=0.9,lr = 0.001)

        ## warmstart
        if startfrom_path is None:
            self.startfrom_path = "runindex:\\{}".format(self.current_runindex)
        else:
            self.startfrom_path = startfrom_path

        ## counter
        self.cnt_reported_epoch = 0

        ## last state
        self.last_sd_path = None
        self.last_epoch = None

    def fromRunindexLink(self,url:'runindex:\\')->Union[str,None]:
        """Ref self.save_folder.
        """
        prefix = "runindex:\\"
        n_prefix = len(prefix)
        if url[:n_prefix] == prefix:
            runindex_str = url[n_prefix:]
            sd_path = os.path.join(self.save_folder,'{}.sd'.format(runindex_str))
            return sd_path
        else:
            return None

    def run(self):
        """Call self.reset() first if self.isFinish == True
        """

        try:
            if self.isFinished:
                self.reset() # isFinished := False

            self.model.to(GPU)

            # start from trained state-dict
            if self.startfrom_path is not None:
                ### load sd-file to torch-Module
                self.startfrom = torch.load(self.startfrom_path)
                self.model.load_state_dict(self.startfrom)
            else:
                self.startfrom = None
            
            if self.startfrom_path_raw != self.startfrom_path:
                sf_fullpath = "({})".format(self.startfrom_path)
            else:
                sf_fullpath = "" 
            self.logger.send('Start from trained: {}{}'.format(self.startfrom_path_raw,sf_fullpath))

            setting = {
                'runindex':self.dblogger.current_runindex,
                'model':type(self.model).__name__,
                'logger':self.logger.out_path,
                'dblogger':self.dblogger.db_path,
                'train_dataset':type(self.train_dataset).__name__,
                'test_dataset':type(self.test_dataset).__name__,
                'sz_batch':self.sz_batch,
                'n_max_epoch':self.n_max_epoch,
                'n_sample_per_epoch':self.n_sample_per_epoch,
                'n_epoch_per_save_dict':self.n_epoch_per_save_state_dict,
                'startfrom_path':self.startfrom_path,
                'startfrom_path_raw':self.startfrom_path_raw,
            }
            self.logger.send('{}'.format(pformat(setting)))
            self.dblogger.send(timestr=getTimestr(),tag='config',data = setting)
            self.train()

            #
            self.isFinished = True
        except KeyboardInterrupt as e:
            self.logger.send("{}".format(e))
        

    @property
    def current_runindex(self):
        return self.dblogger.current_runindex

    def getPbarDescription(self,cnt_epoch,onReporting=False):
        flag_report = 'R' if onReporting else ' '
        runindex = self.current_runindex
        desc = "@{}#{}{}".format(runindex,cnt_epoch,flag_report)
        return desc

    def train(self):

        self.model.train()
        
        ## reset/prepare data
        self.train_dataloader = torch.utils.data.DataLoader(dataset =self.train_dataset,batch_size = self.sz_batch) 
        self.test_dataloader = torch.utils.data.DataLoader(dataset =self.test_dataset,batch_size = self.sz_batch) 

        ## predict n_sample to deal with
        self.n_max_sample = self.n_max_epoch*self.n_sample_per_epoch
        self.c_n_max_sample = min(self.n_max_sample,len(self.train_dataset))
        self.c_n_max_epoch = min(self.n_max_epoch,ceil(self.c_n_max_sample/self.n_sample_per_epoch))

        ## counter to zero
        cnt_sample = 0
        cnt_epoch = 0
        pbar_sample = tqdm(total = self.c_n_max_sample,ascii = True)  
        pbar_sample.set_description(self.getPbarDescription(cnt_epoch) )

        for truth,sample in self.train_dataloader:
            c_sz_batch = truth.shape[0]
            
            if cnt_epoch >=self.n_max_epoch:
                break

            restore = self.model(sample)
            loss = self.func_loss(truth,restore)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            #self.scheduler.step()

            
            cnt_sample +=c_sz_batch
            pbar_sample.update(c_sz_batch)
            
            if cnt_sample//self.n_sample_per_epoch != cnt_epoch:
                cnt_epoch +=1
                pbar_sample.set_description(self.getPbarDescription(cnt_epoch))


            self.func_after_epoch(cnt_sample,cnt_epoch)
            pbar_sample.set_description(self.getPbarDescription(cnt_epoch,onReporting=True))
            self.func_report(cnt_sample,cnt_epoch)
            pbar_sample.set_description(self.getPbarDescription(cnt_epoch))


            if self.func_achieve_loss(cnt_sample,cnt_epoch,loss):
                self.func_after_epoch(cnt_sample,cnt_epoch,'achieve_loss')
                self.func_report(cnt_sample,cnt_epoch,'achieve_loss')
                break
        pbar_sample.close()
        cnt_epoch +=1
        ppp = self.save_sd(cnt_epoch) # NOTICE a small epoch, or zero-epoch

        self.last_epoch = cnt_epoch
        self.last_sd = ppp

    def save_sd(self,cnt_epoch)->'url':
        sd = self.model.state_dict()
        runindex = self.dblogger.current_runindex
        sd_name = "{}_{}.sd".format(runindex, cnt_epoch)
        ppp = os.path.join(self.save_folder,sd_name)
        torch.save(sd,ppp)

        sd_name2 = "{}.sd".format(runindex)
        ppp2 = os.path.join(self.save_folder,sd_name2)
        if os.path.exists(ppp2):
            os.remove(ppp2)
        torch.save(sd,ppp2)
        return ppp2

    def func_report(self, cnt_sample, cnt_epoch, causestr='after_sample'):

        if causestr == 'achieve_loss':
            pass

        if self.cnt_reported_epoch!=cnt_epoch: # NOTICE epoch 0->1 will trigger it at first time!
            self.cnt_reported_epoch +=1

            # Test
            with torch.no_grad():
                aver_psnr = 0
                aver_loss = 0
                n_test_sample= 0

                for truth,sample in self.test_dataloader:
                    c_sz_batch = truth.shape[0]
                    self.model.eval()
                    restore= self.model(sample)
                    loss = self.func_loss(truth,restore)
                    # sqrt(loss.item())
                    psnr = getPsnr(truth,restore) # average on batch
                    aver_psnr +=psnr
                    aver_loss +=sqrt(loss.detach().to(CPU).item())
                    n_test_sample+=1 #  NOT c_sz_batch
                

                aver_psnr  =  aver_psnr/n_test_sample
                aver_loss = (aver_loss/n_test_sample)**2
                self.model.train()
            

            # Save
            timestr = getTimestr()
            # Save med test

            self.dblogger.send(timestr = timestr,tag = 'loss',data = {'loss':aver_loss})
            self.dblogger.send(timestr = timestr,tag = 'psnr',data = {'psnr':aver_psnr})

            # save state_dict
            if self.cnt_reported_epoch % self.n_epoch_per_save_state_dict == 0:
                self.save_sd(self.cnt_reported_epoch)

    def func_achieve_loss(self,cnt_sample,cnt_epoch,loss):
        return False

    def func_after_epoch(self,cnt_sample,cnt_epoch,causestr = 'after_sample'):
        pass
