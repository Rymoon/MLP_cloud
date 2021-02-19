'''
2021-2-5
'''
import RenNet
import os,sys,hashlib
import datetime
import json
from typing import Iterable
from PySide2.QtCore import QDateTime,Qt
import torch

# PLOT Const

CPU = torch.device('cpu')
GPU = torch.device('cuda')


# PLOT Time

def getTimestr(format=Qt.ISODateWithMs):
    '''
    Qt.ISODateWithMs
        '2021-02-08T11:56:15.762'
    Qt.ISODate
        '2021-02-08T11:56:15'
    '''
    
    
    now = QDateTime.currentDateTime()
    
    return now.toString(format)

def getTimestrAsFilename():
    now = QDateTime.currentDateTime()
    #print(now.toString("yyyy-MM-dd-hh-mm-ss-zzz"))
    return now.toString("yyyy_MM_dd_hh_mm_ss_zzz")

# PLOT File

def getcfp(s:"__file__"):
    p = os.path.split(s)[0]
    return p


def checkFolder(p,build_if_not_exist=False):
    flag = os.path.exists(p)
    if build_if_not_exist and not flag:
        os.mkdir(p)
        return True
    else:
        return flag


def loadConfig(s:"__file__"=None,*,config_path = None):
    '''Load config file. Usage: `loadConfig(__file__)`.

    If called in `xxx.py`, this will load `xxx.json` and return a `dict`.

    Parameters
    ----------
    s : str
        __file__

    Returns
    -------
    dict
        Data in JSON file(config).

    '''
    if s is not None:
        cfp,cfn = os.path.split(s)
        cfn = cfn.split('.')[0]
        config_path = os.path.join(cfp,'.'.join([cfn,'json']))
    
    
    f = open(config_path)
    data = f.read()
    f.close()
    data = json.loads(data)
    assert type(data) is dict
    return data


class Runner:
    '''Manage info about the script on runing.
    '''
    def __init__(self,base_path=None,base_name = None,*,filestr:'__file__'=None):
        if filestr is None:
            self.base_path = base_path
            self.base_name = base_name
        else:
            cfp,cfn = os.path.split(filestr)
            cfn = cfn.split('.')[0]
            self.base_name = cfn
            self.base_path  = cfp

    def open(self,base_path):
        self.base_path = base_path


def md5(*,filepath=None,content:str=None)->str:
    if filepath is not None:
        m = hashlib.md5()   #创建md5对象
        with open(filepath,'rb') as fobj:
            while True:
                data = fobj.read(4096)
                if not data:
                    break
                m.update(data)  #更新md5对象

        result = m.hexdigest()    #返回md5对象
    else:
        m = hashlib.md5(content) #创建md5对象
        result =  m.hexdigest()
    return result


# PLOT Console

def COLOR(s:str,color:'r/g'='r'):
    '''
    r   red
    g   green
    '''
    if color == 'r':
        r= '\033[5;31;40m{}\033[0m'.format(s)
    elif color =='g':
        r = '\033[4;32;40m{}\033[0m'.format(s)
    else:
        r = s
    return r
import logging
class getDebugOut:
    fmt_default = "%(asctime)s %(levelname)s %(filename)s %(lineno)d %(process)d :: %(message)s"
    datefmt_default = "%a %d %b %Y %H:%M:%S"

    def __init__(self,out_path,mode='cf',fmt = None,datefmt=None):
        self.fmt = fmt
        self.datefmt = datefmt
        self.mode = mode
        self._out_path = out_path

        olg = logging.getLogger(self.out_path)
        olg.setLevel(logging.DEBUG)
        olg0 = None
        olg1 = None

        if 'c' in self.mode:
            olg0 = logging.StreamHandler(sys.stdout)
            olg0.setLevel(logging.DEBUG)
            olg.addHandler(olg0)
        if 'f' in self.mode:
            olg1 = logging.FileHandler(self.out_path,encoding='utf-8')
            olg1.setLevel(logging.DEBUG)
            olg.addHandler(olg1)

        if self.fmt is not None:
            formatter = logging.Formatter(self.fmt,self.datefmt) # fileformatter
            #olg0.setFormatter(fmt)
            olg1.setFormatter(formatter)

        self.olg = olg
    
    @property
    def out_path(self):
        return self._out_path

    def send(self,msg):
        return self.olg.debug(msg)

# PLOT itertools

class batch:
    """Iterator.
    Yield tuple of length sz_batch.
    The last yielded-value may smaller than sz_btach.
    """
    def __init__(self,it:Iterable,sz_batch:int):
        self.it = it
        self.sz_batch = sz_batch

    def __iter__(self)->tuple:
        while True:
            cache = tuple()
            for i,obj in zip(range(self.sz_batch),self.it):
                cache+=(obj,)
            if len(cache)==0:
                #raise StopIteration()
                return
            yield cache


