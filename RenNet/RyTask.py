from typing import Callable
import torch
from torch import nn
from tqdm import tqdm

class ConstScheduler:
    def __init__(self,optimizer=None):
        super().__init__()
    def step(self,closure = None):
        pass


class Task:
    def __init__(self,model:nn.Module,dataloader,func_loss,optimizer,scheduler,*,taskstr,taskstr_short,n_max_epoch,n_sample_per_epoch):
        """
        `dataloader` ONLY used once, i.e. ONLY one for-loop called.
        """
        self.model= model
        self.dataloader = dataloader
        self.func_loss = func_loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.n_max_epoch = n_max_epoch
        self.n_sample_per_epoch = n_sample_per_epoch
        self.taskstr = taskstr
        self.taskstr_short = taskstr_short

    def func_achieve_loss(self,cnt_sample,cnt_epoch,loss):
        return False

    
    def func_after_epoch(self,cnt_sample,cnt_epoch,causestr = 'after_sample'):
        pass

    def func_report(self,cnt_sample,cnt_epoch,causestr='after_sample'):
        pass

    def train(self):
        self.model.train()
        n_max_sample = self.n_max_epoch*self.n_sample_per_epoch
        pbar_epoch = tqdm(desc=self.taskstr_short,total=self.n_max_epoch)
        pbar_sample = tqdm(total=n_max_sample)
        cnt_sample = 0
        cnt_epoch = 0
        for truth,sample in self.dataloader:
            
            if cnt_epoch >=self.n_max_epoch:
                break

            restore = self.model(sample)
            loss = self.func_loss(truth,restore)

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

            
            cnt_sample +=1
            pbar_sample.update(1)
            if cnt_sample%self.n_sample_per_epoch == 0:
                cnt_epoch +=1
                pbar_epoch.update(1)
            self.func_after_epoch(cnt_sample,cnt_epoch)
            pbar_sample.set_description('rpt')
            self.func_report(cnt_sample,cnt_epoch)
            pbar_sample.set_description(None)


            if self.func_achieve_loss(cnt_sample,cnt_epoch,loss):
                self.func_after_epoch(cnt_sample,cnt_epoch,'achieve_loss')
                self.func_report(cnt_sample,cnt_epoch,'achieve_loss')
                break
        pbar_epoch.close()
        pbar_sample.close()




