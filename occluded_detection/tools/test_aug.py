import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import json
import os
import cv2
import copy
from easydict import EasyDict as edict
import uuid
file = json.load(open('/home/zhangming/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_train1_20191223/annotations.json', 'r'))
print(file.keys(),file['annotations'][1:10])
bbs_new=[]
iamge_new=[]
img_path=[]
image=file['images']
for i in range(len(image)):
    img_path.append(os.path.join('/home/zhangming/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_train1_20191223/images', image[i]['file_name']))
annotation=file['annotations']
for j in range(len(image)):
    bbs_new.append([])
for k in range(len(annotation)):
    for l in range(len(image)):
        if annotation[k]['image_id']==l:
            bbs_new[l].append(ia.BoundingBox(annotation[k]['bbox'][0],annotation[k]['bbox'][1],annotation[k]['bbox'][2],annotation[k]['bbox'][3],label=annotation[k]['category_id']))
seq = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=0.05 * 255),
            iaa.Affine(translate_px={"x": (1, 5)}),
            iaa.Crop(px=(1, 16), keep_size=False),
            iaa.Fliplr(0.5),
            iaa.GaussianBlur(sigma=(0, 3.0)),
            iaa.CoarseDropout(p=0.1, size_percent=0.1)
        ])
images_aug=copy.deepcopy(bbs_new)
bbs_aug=copy.deepcopy(bbs_new)
# for i in range(len(images_aug)):
#     images_aug[i], bbs_aug[i] = seq(images=cv2.imread(img_path[i])[np.newaxis,:], bounding_boxes=bbs_new[i])
sum=0
for i in range(len(annotation)-1):
    if annotation[i]['category_id']!=annotation[i+1]['category_id'] and annotation[i]['image_id']!=annotation[i+1]['image_id']:
        sum+=1
print(sum)
# uuid_str = uuid.uuid4().hex
# image_name=
# cv2.imwrite('test.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70])
# meta={}
# meta['image']=
# meta['annotations']=
# meta['info']=file['info']
# meta['license']=file['license']
# meta['category']=file['category']




