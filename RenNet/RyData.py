
import json
import numpy as np
import os,sys,datetime
from math import sqrt,floor,ceil,inf,isinf

from typing import List,Tuple,Union,Dict,Callable,Any
import PIL
from PIL import Image

# from bm3d_demos.experiment_funcs import get_experiment_noise
from  skimage.util import random_noise

import torch
from torch.utils.data import Dataset,IterableDataset

import PySide2
from PySide2.QtSql import QSql
from PySide2.QtSql import QSqlDatabase, QSqlQuery,QSqlDriver

import skimage
from skimage.io import imread
from skimage import img_as_float32,img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio

from RenNet.RyCore import checkFolder,Runner,loadConfig,getcfp
from RenNet.RyCore import CPU,GPU
from RenNet.RyCore import COLOR
from RenNet.RyCore import getTimestr

from RenNet.RyLog import jsonDumpNotNone,check
from RenNet import RyLog,RyCore

import warnings 


# image trans

def getPsnr(truth:torch.Tensor,sample:torch.Tensor):
    w,h = truth.shape[-2],truth.shape[-1]

    def do(t):
        t= t.detach()
        t=  t.to(CPU)
        t= t.reshape(-1,w,h)
        t= t.numpy()
        return t

    truth = do(truth)
    sample = do(sample)
    n=  truth.shape[0]
    v= 0
    for i in range(n):
        tr = truth[i]
        sa = sample[i]
        v += peak_signal_noise_ratio(tr,sa)
    v = v/n
    return v

# PLOT Database

class ImageDB(RyLog.MainHandler):
    ''' All images here, main-uid/refs here.

    A record  === A stored image, or a virtual image-group.
    '''

    base_table_name = "IMAGEDB"
    base_columns = ["timestr","url","meta","md5"]
    base_cols_sql = "uid integer PRIMARY KEY AUTOINCREMENT, timestr text, url text, meta json, md5 text"
    uid_name = 'uid'
    extra_table_name = []

    def __init__(self,config:dict = None):
        super().__init__(config)

    def send(self,values = None,*,timestr=None, url = None, meta = None, md5 = None):
        return self._send(values,[timestr,url,meta,md5])

    def scan(self,image_folder_path,image_suffix:list):
        """
        Using PIL
        """
        n_sended_image = 0
        try:
            for root,dirs,files in os.walk(image_folder_path):
                for name in files:
                    url = os.path.join(root,name)
                    fname = os.path.split(url)[1]
                    if fname.split('.')[1]  in image_suffix:
                        # check md5
                        md5_str = self.md5(filepath = url)
                        search_result = self.find("md5",md5_str)
                        if len(search_result) == 0:
                            # send
                            im_tmp = Image.open(url)
                            size = im_tmp.size
                            v = getTimestr(),url,{'size':list(size),'noise_level':0},md5_str
                            im_id = self.send(v)
                            n_sended_image+=1
        except Exception as e:
            print('Sended image: {}'.format(n_sended_image))
            raise e
        finally:
            pass
        
        return n_sended_image

    @staticmethod
    def md5(*,filepath=None,content=None):
        return RyCore.md5(filepath=filepath,content=content)


class ImageAlbum(RyLog.SubHandler):
    '''
    A Album is an image-set.

    A page is an image-group, for example, `{'label':23,'sample':24}`, or a single image `{'VALUE':235}`.
    '''
    table_name = 'ALBUM'
    columns = ["page"]
    extra_columns_sql = "page json"

    def __init__(self,imagedb:ImageDB):
        super().__init__(imagedb)

    def send(self,values= None,* , timestr=None,url = None,meta = None, md5=None, page:dict = None):
        values = list(values)

        page = jsonDumpNotNone(page)
        values[-1] = jsonDumpNotNone(values[-1])
        return self._send(values,[timestr,url,meta,md5],[page])
    
    @staticmethod
    def md5(*,filepath=None,content=None):
        return RyCore.md5(filepath=filepath,content=content)



# PLOT Dataset
class ImageUrlSet:
    """Wrapper for url-databse. 
    Not any file-IO Here.
    """
    def __init__(self,db_path,table_name,index_column,url_column):
        """
        qt database name == db_path
        """
        self.db_path = db_path
        self.table_name = table_name
        self.index_column = index_column
        self.url_column =url_column

        self.query = None

        self.open()
    
    def open(self):
        assert os.path.exists(self.db_path)
        db = QSqlDatabase.addDatabase("QSQLITE",self.db_path)
        db.setDatabaseName(self.db_path)
        check(db.open)
        self.query = QSqlQuery(db)

    @property
    def db(self):
        return QSqlDatabase.database(self.db_path)

    def __getitem__(self,key):
        order = "select {} from {} where {} = ?".format(self.url_column,self.table_name,self.index_column)
        q = self.query
        check(q.prepare,order)
        q.addBindValue(key)
        check(q.exec_)
        result = []
        while q.next():
            result.append(q.value(0))
        assert len(result)>0
        assert len(result)==1

        return result[0]

    def __len__(self):
        order = "select COUNT(*) from {}".format(self.table_name)
        q = self.query
        q.prepare(order)
        check(q.exec_)
        if q.next():
            l = q.value(0)
        else:
            raise Exception()

        return l

class ImageDataset(Dataset):
    """
    A map from keys to data samples.

    File-IO here.
    Int index, from 0 to n_max_index-1.

    As gray, torch.Tensor(w,h) , float32, range[0,1] image.


    """
    def __init__(self,url_set:Union[list,dict,ImageUrlSet],n_max_index:int,indexToKey:Callable[[dict,int],Any] ,device = RyCore.CPU,do_prefetch = False):
        self.url_set = url_set
        
        self.env = {}
        self._n_max_index = n_max_index
        self.trans = indexToKey

        self.cache = [None]*self.n_max_index
        self.device = device

        self.do_prefetch = do_prefetch
        if self.do_prefetch:
            self.prefetch()


    @property
    def n_max_index(self):
        return self._n_max_index

    def indexToKey(self,index:int):
        key = self.trans(self.env,index)
        return key

    def getShape(self,index)->Tuple[int,int]:
        '''[Discard]Not load image-file, very fast.

        Using PIL. 

        return Row=h,Col=w
        '''
        assert 0<=index
        assert index<self.n_max_index
        
        key = self.indexToKey(index)
        url = self.url_set[key]

        import sys
        from PIL import Image
        try:
            with Image.open(url) as im:
                w,h = im.size
        except OSError:
            pass

        return (h,w) # NOTICE

    def getSize(self,index)->Tuple[int,int]:
        '''[Discard]Not load image-file, very fast.

        Using PIL. 
        兼容用。
        # TODO Remove this.

        return Width=row,Height=col
        '''
        assert 0<=index
        assert index<self.n_max_index
        
        key = self.indexToKey(index)
        url = self.url_set[key]

        import sys
        from PIL import Image
        try:
            with Image.open(url) as im:
                w,h = im.size
        except OSError:
            pass

        return (w,h)

    def prefetch(self):
        for i in range(len(self)):
            self.loadImage(i)

    def loadImage(self,index):
        key = self.indexToKey(index)
        url = self.url_set[key]
        img = imread(url,as_gray =True)
        img = img_as_float32(img)
        img = torch.from_numpy(img)
        img = img.to(self.device)
        self.cache[index] = img


    def __getitem__(self,index):
        assert 0<=index
        assert index<self.n_max_index

        if self.cache[index] is not None:
            return self.cache[index]
        else:
            self.loadImage(index)

            return img

    def __len__(self):
        return self.n_max_index


class OnlineGaussAdditionalNoise(IterableDataset):
    """
    Yield batched 2d-Gauss-white-noise(additional gauss).
    """
    def __init__(self,shape,std,n_image =-1,*,n_cache=1,device = RyCore.CPU):
        '''
        shape = (...,w,h)
        std respect to image-range[0,1]
        '''
        self.shape = shape
        self.shape_2d = shape[-2:]
        self.batch_shape = shape[:-2]
        self.std = std
        self._n_cache=n_cache
        self.cache= [None]*self.n_cache
        self.n_image = n_image
        self.cnt = 0
        self.device = device

    @property
    def n_cache(self):
        return self._n_cache

    def getNoise(self):
        noi = self.std*np.random.randn(*self.shape) # np.float64
        noi = torch.from_numpy(noi).to(torch.float32)
        noi = noi.to(self.device)
        return noi
        

    def fillCache(self):
        for i in range(len(self.cache)):
            self.cache[i] = self.getNoise()

    def __iter__(self):
        self.fillCache()
        while self.n_image==-1 or self.cnt<self.n_image:
            for v in self.cache:
                self.cnt +=1
                yield v
            self.fillCache()


class OnlineImageDataset(IterableDataset):
    """
    
    """
    def __init__(self,image_dataset:ImageDataset,n_patch,sz_batch,device = RyCore.CPU,*,crop_shape,offset_shape):
        '''
        crop_shape = (w,h)
        coef_crop_per_image = x : crop x*(w/W*h/H) patches from one image
        noise_sigma = σ 
        duplicate_crop = x : apply x different noises on the same crop.
        refresh_interval , discard all cached picture

        '''
        super().__init__()
        self.n_patch = n_patch
        self.image_dataset = image_dataset

        self.sz_batch = sz_batch
        self.crop_shape = crop_shape
        self.offset_shape = offset_shape

        self.device = device
    
    def splitImageDataset(self):
        """ # TODO For Dalaloader co_works>0
        """
        pass

    def crop(self,image:torch.Tensor):
        """
    
        """
        assert len(image.shape) == 2
        W,H=  image.shape
        w,h = self.crop_shape
        dw,dh = self.offset_shape
        n_w = floor((W-w)/dw)+1
        n_h = floor((H-h)/dh)+1
        for i in range(n_w):
            for j in range(n_h):
                l = i*dw
                t = j*dh
                box = l,t,w,h
                yield box


    def __iter__(self):
        # NOTICE not suit dataloader co_worker
        # TODO split workload with image.
        # Be careful! In NumPy indexing, the first dimension (camera.shape[0]) corresponds to rows, while the second (camera.shape[1]) corresponds to columns, with the origin (camera[0, 0]) at the top-left corner!
        sz_batch = self.sz_batch
        cnt_batched_tensor = 0
        for image in self.image_dataset:
            for boxes in RyCore.batch(self.crop(image),sz_batch):
                if cnt_batched_tensor>self.n_patch:
                    return
                if len(boxes) == sz_batch:
                    patches = [image[l:l+w,t:t+h].unsqueeze(0) for l,t,w,h in boxes ]
                    batched_tensor = torch.stack(patches,0) # 4d-batch
                    batched_tensor = batched_tensor.to(self.device)
                    cnt_batched_tensor +=1
                    
                    
                    yield batched_tensor
                else:
                    continue # discard if len<sz_batch
                
class zipIter(IterableDataset):
    def __init__(self,*iter_list,func_after):
        super().__init__()
        self.iter_list = iter_list
        self.func_after = func_after

    def __iter__(self):
        for p in zip(*self.iter_list):
            pp = self.func_after(*p)
            yield pp

    def __len__(self):
        min_len = inf
        for it in self.iter_list:
            try:
                l = len(it)
                if l<min_len:
                    min_len = l
            except Exception as e:
                pass
        if isinf(min_len ):
            raise Exception('zipIter: Not any iter support __len__')
        
        return min_len
            



if __name__=='__main__':
    runner = Runner(getcfp(__file__))
    dbpath = os.path.join(runner.base_path,"RyData_test.db")
    myconfig = {"db_path":dbpath}
    #config = loadConfig(__file__)
    imageDB = ImageDB(myconfig)

    album = ImageAlbum(imageDB)

    url = "D:\Workspace\\ADMM_Net\\MyHumbleADMM2\\MLP\\RenNet\\patch0.png"
    v = getTimestr(),url,{'size':[128,128],'noise_level':0},ImageAlbum.md5(filepath = url)
    label_id = imageDB.send(v)

    url = "D:\Workspace\\ADMM_Net\\MyHumbleADMM2\\MLP\\RenNet\\noised0.png"
    v = getTimestr(),url,{'size':128,'noise_level':0.1},ImageAlbum.md5(filepath = url)
    sample_id = imageDB.send(v)

    v = getTimestr(),None,{'size':128,'noise_level':0.1},None,{'label':label_id,'sample':sample_id}
    album.send(v)

    


