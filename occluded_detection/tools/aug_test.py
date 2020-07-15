import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from tqdm import tqdm
import json

anno_path = "/home/zhangming/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_train1_20191223/annotations_without_bg.json"
anno_o = open(anno_path,'r')
anno = json.load(anno_o)
print(anno['annotations'][1])