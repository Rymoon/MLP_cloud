
import json
import numpy as np
import os,sys,datetime
from math import sqrt,floor,ceil,inf,isinf
from numpy.lib import ufunclike
import torch
from tqdm import tqdm
from skimage import img_as_ubyte

from typing import List,Tuple,Union,Dict,Callable,Any
from torch.functional import Tensor
from torch.utils.data import Dataset,IterableDataset


from RenNet.RyData import getPsnr
from RenNet.RyLog import jsonDumpNotNone,check
from RenNet import RyLog,RyCore,RyData

from MLP_Log import GPU,CPU

def ten2img(t:torch.Tensor):
    if len(t.shape) == 2:
        t = t.detach().to(CPU)
        n = t.numpy()
        #
        n = (n-n.min())/(n.max()-n.min())
        #
        img = img_as_ubyte(n)
        return img



class CropDataset(Dataset):
    """
    # NOTICE
    self.image_dataset.getSize(i) == W,H (PIL,Image) 兼容
    self.image_dataset.getShape(i) == H,W 用此
    image.shape  == H,W (skimage,ndarray)
    """
    def __init__(self,url_set,crop_shape,offset_shape,std,device,*,n_max_patch_index=None,do_prefetch= False,verbose_patch_index = False):
        self.url_set = url_set
        self.std=std
        self.device = device

        self.image_dataset= RyData.ImageDataset(self.url_set,n_max_index = len(self.url_set),indexToKey=lambda env,i:i+1,device=self.device,do_prefetch=do_prefetch)

        self.verbose_patch_index = verbose_patch_index

        self.crop_shape = crop_shape
        self.offset_shape = offset_shape

        self.min_patch_index_each_image = [0]
        for i in range(len(self.image_dataset)):
            h,w = self.image_dataset.getShape(i)
            nnn = self.max_crop_index(h,w)
            last_nnn = self.min_patch_index_each_image[i]
            self.min_patch_index_each_image.append(nnn+last_nnn)
        
        if n_max_patch_index is not None:
            self._n_max_patch_index = min(n_max_patch_index,self.min_patch_index_each_image[-1])
        else:
            self._n_max_patch_index = self.min_patch_index_each_image[-1]
        
        

    @property
    def n_max_patch_index(self):
        
        return self._n_max_patch_index

    def __getitem__(self,patch_index=None,*,image_index=None,crop_index = None)->Tuple[Tensor,Tensor]:
        """(truth,sample), each crop_shape,2d
        """
        if patch_index is not None:
            for i,nnn in enumerate(self.min_patch_index_each_image):
                if patch_index<nnn:
                    image_index = i-1
                    break
            crop_index = patch_index-self.min_patch_index_each_image[image_index]
        
        image = self.image_dataset[image_index].to(GPU)
        H,W = self.image_dataset.getShape(image_index)
        box = self.crop(H,W,crop_index)
        r,c,h,w = box
        patch = image[r:r+h,c:c+w]

        noise = self.getNoise((h,w),self.std,self.device)

        truth = patch.to(self.device)
        sample = (noise+patch).to(self.device)

        if self.verbose_patch_index:
            return truth,sample,patch_index
        else:
            return truth,sample

    def __len__(self):
        return self.n_max_patch_index

    def verbosePatchIndex(self,flag= True):
        self.verbose_patch_index = flag


    def max_crop_index(self,H,W):
        h,w = self.crop_shape
        dh,dw = self.offset_shape
        n_h = floor((H-h)/dh)+1
        n_w = floor((W-w)/dw)+1
        return n_h*n_w

    def crop(self,H,W,index:int = None):
        """
        # NOTICE H:row,W:col 
        tensor2d-shape = (row,col)
        drop last
        """
        h,w = self.crop_shape
        dh,dw = self.offset_shape
        n_h = floor((H-h)/dh)+1
        n_w = floor((W-w)/dw)+1

        i = index//n_w # row
        j = index%n_w # col

        assert i<n_h

        r = i*dh
        c = j*dw
        box = r,c,h,w
        return box

    @staticmethod
    def getNoise(shape,std,device):
        noi = std*np.random.randn(*shape) # np.float64
        noi = torch.from_numpy(noi).to(torch.float32)
        noi = noi.to(device)
        return noi

    def showExample(self,image_index,item_index:int=None,patch_dict=None)->Tuple:
        """(truth,sample), call __getitem__ to splice the origin image.

        assert patch_dict[image_index][item_index]
        """

        # BUG FIXME 只显示第一张图？？
        if patch_dict is None:
            patch_dict=  self

        def addPatch(image,patch,patch_i):
            H,W = image.shape
            h,w = self.crop_shape
            dh,dw = self.offset_shape
            n_h = floor((H-h)/dh)+1
            n_w = floor((W-w)/dw)+1

            i = patch_i//n_w # row
            j = patch_i%n_w # col

            r = i*dh
            c = j*dw
            assert dw<w
            assert 2*dw <=w
            assert dh<h
            assert 2*dh<=h

            def _addPatch(coef):
                image[r:r+dh,       c:c+dw]     +=coef[0][0]*patch[:dh,      :dw] # left top
                image[r+dh:r+2*dh,  c:c+dw]     +=coef[1][0]*patch[dh:2*dh,  :dw] # left bottom
                image[r:r+dh,       c+dw:c+2*dw]+=coef[0][1]*patch[:dh,      dw:2*dw] # right top
                image[r+dh:r+2*dh,  c+dw:c+2*dw]+=coef[1][1]*patch[dh:2*dh,  dw:2*dw] #right bottom


            # dicard [:,2*dw:w] part ,and [2*dh:h,:] part
            # image,patch are not sparse, so do addtion-opr as below.
            if i ==0 and j == 0:
                _addPatch([ [1,1/2],
                            [1/2,1/4]])
            elif i ==0 and j==n_w-1:
                _addPatch([ [1/2,1],
                            [1/4,1/2]])
            elif i ==n_h-1 and j==0:
                _addPatch([ [1/2,1/4],
                            [1,1/2]])
            elif i ==n_h-1 and j==n_w-1:
                _addPatch([ [1/4,1/2],
                            [1/2,1]])
            elif i == 0 :
                _addPatch([ [1/2,1/2],
                            [1/4,1/4]])
            elif i == n_h-1 :
                _addPatch([ [1/4,1/4],
                            [1/2,1/2]])
            elif j == 0 :
                _addPatch([ [1/2,1/4],
                            [1/2,1/4]])
            elif j == n_w-1 :
                _addPatch([ [1/4,1/2],
                            [1/4,1/2]])
            else:
                _addPatch([ [1/4,1/4],
                            [1/4,1/4]])

        min_pindex = self.min_patch_index_each_image[image_index]
        max_pindex = self.min_patch_index_each_image[image_index+1]
        image = torch.zeros_like(self.image_dataset[image_index]) # on GPU , if self._image_dataset on GPU

        if item_index is not None:
            try:
                patch_dict[min_pindex][item_index]
            except Exception as e:
                raise e

            
            for patch_i in range(max_pindex-min_pindex):
                patches = patch_dict[patch_i+min_pindex]
                patch = patches[item_index]
                addPatch(image,patch,patch_i)

            return image
        else:
            try:
                patch_dict[min_pindex]
            except Exception as e:
                raise e

            for patch_i in range(max_pindex-min_pindex):
                patch = patch_dict[patch_i+min_pindex]
                addPatch(image,patch,patch_i)
            return image



class RestoreDataset:
    def __init__(self,model,crop_dataset:CropDataset,*,sz_batch=1):

        self.model = model
        self.crop_dataset = crop_dataset
        self.crop_dataset.verbosePatchIndex()
        self.data = {}

        self.sz_batch = sz_batch

    def test(self):
        self.test_dataloader = torch.utils.data.DataLoader(dataset =self.crop_dataset,batch_size = self.sz_batch) 

        # Test
        with torch.no_grad():
            n_test_sample= 0
            pbar = tqdm(total = len(self.test_dataloader),desc = "batchT",ascii=True)
            for truth,sample,patch_indices in self.test_dataloader:
                c_sz_batch = truth.shape[0]

                self.model.eval()

                restore= self.model(sample)
                restore = restore.detach() # on GPU, if sample on GPU

                n_test_sample+=c_sz_batch 
                for i,pindex in enumerate(patch_indices):
                    self.data[pindex.item()] = restore[i]
                pbar.update(1)
            pbar.close()
            



    def showExample(self,image_index):
        truth= self.crop_dataset.showExample(image_index,0)
        sample= self.crop_dataset.showExample(image_index,1)
        restore= self.crop_dataset.showExample(image_index,None,self.data)

        return truth,sample,restore


class StdTestPatchataset(CropDataset):
    def __init__(self,crop_shape,offset_shape,std,device,*,n_max_patch_index = None,do_prefetch = False):
        url_set = RyData.ImageUrlSet("D:\\Workspace\\ADMM_Net\\MyHumbleADMM2\\MLP\\RenNetData\\BM3D\\BM3D.db",'IMAGEDB','uid','url') # index uid from 1

        super().__init__(url_set,crop_shape,offset_shape,std,device,n_max_patch_index=n_max_patch_index,do_prefetch= do_prefetch)
    

class BSDS500PatchDataset(CropDataset):
    def __init__(self,crop_shape,offset_shape,std,device,*,n_max_patch_index=None,do_prefetch = False):
        url_set = RyData.ImageUrlSet("D:\\Workspace\\ADMM_Net\\MyHumbleADMM2\\MLP\\RenNetData\\BSDS500\\BSDS500.db",'IMAGEDB','uid','url') # index uid from 1

        super().__init__(url_set,crop_shape,offset_shape,std,device=device,n_max_patch_index=n_max_patch_index,do_prefetch=do_prefetch)