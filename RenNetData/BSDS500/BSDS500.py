from RenNet.RyData import ImageAlbum
import json
import numpy as np
import os,sys,datetime

from RenNet.RyCore import checkFolder,Runner,loadConfig,getcfp
from RenNet.RyCore import CPU,GPU
from RenNet.RyCore import COLOR
from RenNet.RyCore import getTimestr

from RenNet.RyLog import jsonDumpNotNone
from RenNet import RyLog,RyCore,RyData

from typing import List,Tuple,Union,Dict

from tqdm import tqdm,trange
import PIL
from PIL import Image


config = loadConfig(__file__)
runner = Runner(getcfp(__file__))
image_folder_path = config["image_folder_path"]
assert os.path.exists(image_folder_path)

# Regist picture into imageDB
imageDB = RyData.ImageDB(config)

n_sended_image = imageDB.scan(image_folder_path,config['image_suffix'])

print('Sended image: {}'.format(n_sended_image))

# Establish Album,  clean - noisy

album0 = ImageAlbum(imageDB)


