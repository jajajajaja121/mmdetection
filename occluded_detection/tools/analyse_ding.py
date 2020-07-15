#-*-coding:utf-8-*-
from easydict import EasyDict as edict
import sys
# sys.path.append('/home/dingmingxu/work/work/code/mmdetection_ding/mmcv-master')
sys.path.append('/e/work/envs/imgaug-master')
# import mmcv
# from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
# from mmdet.apis import inference_detector, show_result, init_detector,crop_inference_detector
import time
import os
import numpy as np
from tqdm import tqdm
import copy
from easydict import EasyDict as edict
import copy
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib
#import seaborn as sns
import os
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug as ia
import random
import math
from bbox_utils import py_cpu_nms as nms
from bbox_utils import box_voting,box_voting_ad_iou
import torch
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class Analyse():
    def __init__(self, json_paths, img_path, flag_test=False):
        # pass
        self.flag_test = flag_test
        self.all_instance, self.cla_instance, self.img_instance, self.defect,self.categories = self._create_data_dict(json_paths, img_path)
        self.classes=  {
                        1: '酒液杂质',
                        2: '瓶身破损',
                        3: '瓶身气泡',
                        }

    def vis_result(self,models, configs, pic_dir, flag_traindata = False):
        # index = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
        # index = {1: 8, 2: 6, 3: 7}
        index = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 10}
        init_detectors = []
        for i in range(len(models)):
            # build the model from a config file and a checkpoint file
            init_detectors.append(init_detector(configs[i], models[i], device='cuda:0'))
        pics = os.listdir(pic_dir)
        for im in tqdm(pics):
            img_path = os.path.join(pic_dir, im)
            image = cv2.imread(img_path)

            for num in range(len(init_detectors)):
                img_temp = copy.deepcopy(image)
                if flag_traindata:
                    if img_path not in self.img_instance.keys():
                        print("Error pleaase check if you use test data not train data")
                    annos = self.img_instance[img_path]
                    if len(annos['bbox']):
                        for anno in annos['bbox']:
                            cv2.rectangle(img_temp, (int(anno['x1']), int(anno['y1'])), (int(anno['x1']+anno['w']), int(anno['y1']+anno['h'])), (255, 0, 0),
                                          2)
                            cv2.putText(img_temp,'%i'%anno['category_id'],(int(anno['x1']), int(anno['y1'])), cv2.FONT_HERSHEY_COMPLEX,0.8,(255, 0, 0), 1)
                result = inference_detector(init_detectors[num], img_temp)
                for i, boxes in enumerate(result, 1):
                    if len(boxes):
                        defect_label = index[i]
                        for box in boxes:
                            cv2.rectangle(img_temp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                                          2)
                            cv2.putText(img_temp, '%d %0.3f' % (defect_label, float(box[4])),
                                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX,
                                        0.8,
                                        (0, 255, 0), 1)
                cv2.namedWindow('im',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('im',(1000,600))
                cv2.imshow('im',img_temp)
            cv2.waitKey(0)

    def _create_data_dict(self,json_paths, img_path):
        '''
        :return:
        ins
            {'im_name': abs_path/xxx.jpg, 'bbox': {'x1': 165.14,'y1': 53.71,'w': 39.860000000000014,'h': 63.29,'category_id': 2},'width': 20, 'height': 20}

        all_instance
            [ ins1,ins2,...]

        cla_instance
            {'1':[ins1,ins2,...], '2'[ins1,ins2,...]}

        instance
            {
            'im_name': '/home/dingmingxu/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_train1_20191223/images/img_0017151.jpg',
             'bbox': [{'x1': 165.14,'y1': 53.71,'w': 39.860000000000014,'h': 63.29,'category_id': 2}],
             'width': 492,
             'height': 658
             }

        img_instance
            {'xx1.jpg':instance  ,'xxx.jpg':instance}

        '''
        '''
        json format
        {
            "images":
                [
                    {"file_name":"cat.jpg", "id":1, "height":1000, "width":1000},
                    {"file_name":"dog.jpg", "id":2, "height":1000, "width":1000},
                    ...
                ]
            "annotations":
                [
                    {"image_id":1, "bbox":[100.00, 200.00, 10.00, 10.00], "category_id": 1}
                    {"image_id":2, "bbox":[150.00, 250.00, 20.00, 20.00], "category_id": 2}
                    ...
                ]
            "categories":
                [
                    {"id":0, "name":"bg"}
                    {"id":1, "name":"cat"}
                    {"id":1, "name":"dog"}
                    ...
                ]
        }
        '''
        all_instance = []
        cla_instance = edict()
        img_instance = edict()
        defect = edict()
        info_temp = {}
        im_name_to_index = {}
        categories = ""
        for i in range(len(json_paths)):
            anno_path = json_paths[i]
            anno_o = open(anno_path,'r')
            anno = json.load(anno_o)
            # get image info
            '''
            [1:'image1',
             2:'imag2',
             ...
            ]
            '''
            for im_info in anno['images']:
                file_name = im_info['file_name']
                id = im_info['id']
                if not self.flag_test:
                    height = im_info['height']
                    width = im_info['width']
                else:
                    height = 0
                    width = 0
                file_path = os.path.join(img_path[i],file_name)
                im_name_to_index[id] = [file_path,height,width]
            # get anno info
            '''
            {
                1:[{x1: 1,y1: 2,w: 3,h: 4,category_id:3}, {x1: 1,y1: 2,w: 3,h: 4,category_id:3}, ...],
                2:[{x1: 1,y1: 2,w: 3,h: 4,category_id:3}, ...]
                ...
            }
            '''
            for im_info in anno['annotations']:
                image_id = im_info['image_id']
                bbox = im_info['bbox']
                category_id = im_info['category_id']
                bbox.append(category_id)
                ins = edict()
                ins.x1 = bbox[0]
                ins.y1 = bbox[1]
                ins.w = bbox[2]
                ins.h = bbox[3]
                if self.flag_test:
                    ins.score = im_info['score']
                ins.category_id = category_id
                if image_id in info_temp.keys():
                    info_temp[image_id].append(ins)
                else:
                    info_temp[image_id] = [ins]
            # get defect info
            if not self.flag_test:
                categories = anno['categories']
                for im_info in anno['categories']:
                    defect[str(im_info['id'])] = im_info['name']
            else:
                categories = {}
            # for im_info in anno['categories']:
            #     defect[str(im_info['id'])] = im_info['name']
            for index in im_name_to_index.keys():
                if index not in info_temp.keys():
                    continue
                im_name = im_name_to_index[index][0]
                w = im_name_to_index[index][2]
                h = im_name_to_index[index][1]
                img_instance[im_name] = {}
                img_instance[im_name].bbox = info_temp[index]
                img_instance[im_name].width = w
                img_instance[im_name].height = h
                for box in img_instance[im_name].bbox:
                    ins = edict()
                    ins.im_name = im_name
                    ins.bbox = box
                    ins.width = img_instance[im_name].width
                    ins.height = img_instance[im_name].height
                    cat = box['category_id']
                    if str(cat) in cla_instance.keys():
                        cla_instance[str(cat)].append(ins)
                    else:
                        cla_instance[str(cat)] = [ins]
                    all_instance.append(ins)
        return all_instance, cla_instance, img_instance, defect,categories

    def adjust_gamma(self, imgs, gamma=0.3):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        new_imgs = cv2.LUT(np.array(imgs, dtype=np.uint8), table)
        return new_imgs

    def gamma(self,image_dir):
        save_dir = image_dir+'_gamma'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for im in tqdm(os.listdir(image_dir)):
            image = cv2.imread(os.path.join(image_dir,im))
            image_gamma = self.adjust_gamma(image)
            cv2.imwrite(os.path.join(save_dir,im),image_gamma)

    def draw_box(self,im,image):
        anno = self.img_instance[im]
        bbox = anno['bbox']
        for box in bbox:
            x1 = box['x1']
            y1 = box['y1']
            x2 = x1 + box['w']
            y2 = y1 + box['h']
            cat = box['category_id']
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(image, '%d' % cat,
                        (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (0, 255, 0), 2)
        return image

    def analyse_group(self,image_dir):
        ims = os.listdir(image_dir)
        done = []
        for im in ims:
            sp = im.split('.')[0].split('_')
            if len(sp)==3 and sp[0] not in done:
                im_prefix = sp[0]+'_'+sp[1]+'_'
                im0_name = os.path.join(image_dir,im_prefix+'0.jpg')
                im1_name = os.path.join(image_dir,im_prefix+'1.jpg')
                im2_name = os.path.join(image_dir,im_prefix+'2.jpg')
                im3_name = os.path.join(image_dir,im_prefix+'3.jpg')
                im4_name = os.path.join(image_dir,im_prefix+'4.jpg')
                im0 = cv2.imread(im0_name).astype(np.int16)
                im0 = self.draw_box(im0_name, im0)
                im1 = cv2.imread(im1_name).astype(np.int16)
                im1 = self.draw_box(im0_name, im1)
                im2 = cv2.imread(im2_name).astype(np.int16)
                im2 = self.draw_box(im0_name, im2)
                im3 = cv2.imread(im3_name).astype(np.int16)
                im3 = self.draw_box(im0_name, im3)
                im4 = cv2.imread(im4_name).astype(np.int16)
                im4 = self.draw_box(im0_name, im4)
                im_all_r = (im4+im1+im3+im2+im0)
                im_all_g = im0-(im4+im1+im3+im2)/4
                im_all_b = im0
                im_all = im_all_r
                im_all[:,:,1] = im_all_g[:,:,1]
                im_all[:,:,2] = im_all_b[:,:,2]
                # im_all = im_all.astype(np.float32)
                im_all = self.draw_box(im0_name, im_all)
                im_all = self.draw_box(im1_name, im_all)
                im_all = self.draw_box(im2_name, im_all)
                im_all = self.draw_box(im3_name, im_all)
                im_all = self.draw_box(im4_name, im_all)
                # cv2.imshow('im00',im0-im1)
                # cv2.imshow('im01',im0[:,:,1])
                # cv2.imshow('im02',im0[:,:,2])
                cv2.imshow('im0',im0)
                # cv2.imshow('im1',im1)
                # cv2.imshow('im2',im2)
                # cv2.imshow('im3',im3)
                # cv2.imshow('im4',im4)
                cv2.imshow('im',im_all)
                cv2.waitKey(0)

    def compute_mean_std(self,im_dir):
        sum_mean = 0
        sum_std = 0
        ims = os.listdir(im_dir)
        for i, im in tqdm(enumerate(ims)):
            sp = im.split('.')[0].split('_')
            if len(sp) != 3:
                continue
            im_prefix = sp[0] + '_' + sp[1] + '_'
            im0_name = os.path.join(im_dir, im_prefix + '0.jpg')
            im1_name = os.path.join(im_dir, im_prefix + '1.jpg')
            im2_name = os.path.join(im_dir, im_prefix + '2.jpg')
            im3_name = os.path.join(im_dir, im_prefix + '3.jpg')
            im4_name = os.path.join(im_dir, im_prefix + '4.jpg')
            im0 = cv2.imread(im0_name)
            im1 = cv2.imread(im1_name)
            im2 = cv2.imread(im2_name)
            im3 = cv2.imread(im3_name)
            im4 = cv2.imread(im4_name)
            im_all_r = (im4 + im1 + im3 + im2)+im0
            im_all_g = im4 + im1 + im3 + im2
            im_all_b = im0
            im_all = im_all_r
            im_all[:, :, 1] = im_all_g[:, :, 1]
            im_all[:, :, 2] = im_all_b[:, :, 2]
            pic = torch.from_numpy(im_all).cuda()
            mean = torch.mean(pic, dim=(0, 1))
            std = torch.std(pic, dim=(0, 1))
            sum_mean = (i * sum_mean + mean) / (i + 1)
            sum_std = (i * sum_std + std) / (i + 1)
        print(sum_mean)
        print(sum_std)

    def modify_json(self,json_path,remove_label,stay_label,bad_image_path=None):
        r = json.load(open(json_path, 'r'))
        bad_id = []
        if bad_image_path is not None:
            bad_ims = os.listdir(bad_image_path)
            images = r['images']
            for image in images:
                if image['file_name'] in bad_ims:
                    bad_id.append(image['id'])
        new_r = {}
        categories = r['categories']
        # new_categories = categories
        new_categories = []
        for categories in categories:
            if categories['id'] in stay_label:
                new_categories.append(categories)
        print('new_categories', [categories['id'] for categories in new_categories])
        new_r['categories'] = new_categories
        new_annotations = []
        remove_im_id = []
        annotations = r['annotations']
        for annotation in annotations:
            if annotation['category_id'] in remove_label:
                remove_im_id.append(annotation['image_id'])
                continue
            new_annotations.append(annotation)
        remove_im_id+=bad_id
        print('remove %d 6 7 8 pics' % len(remove_im_id))
        new_r['annotations'] = new_annotations
        images = r['images']
        new_images = []
        for image in images:
            if image['id'] in remove_im_id:
                continue
            new_images.append(image)
        new_r['images'] = new_images
        prefix = ""
        for label in stay_label:
            prefix+='_'+str(label)
        prefix+='.json'
        save_json = json_path.replace('.json',prefix)
        with open(save_json, 'w')as f:
            json.dump(new_r, f)

    def vis_label(self,analyse = None,label=[],image_name=[],flag_compare = False):
        for im in self.img_instance.keys():
           flag_show = False
           im_name_ = im.split('/')[-1]
           im_name = im_name_.split('.')[0].split('_')[1]
           if im_name not in image_name and len(image_name):
               continue
           image = cv2.imread(im)
           # image_gamma = self.adjust_gamma(image)
           anno = self.img_instance[im]
           bbox = anno['bbox']
           if len(label):
               for box in bbox:
                   cat = box['category_id']
                   if cat in label:
                      flag_show = True
           else:
               flag_show = True
           color = {9:(255,0,0),10:(0,255,0),1:(255,0,0),3:(0,255,0)}
           if flag_show:
               for box in bbox:
                   x1 = box['x1']
                   y1 = box['y1']
                   # if y1>100:
                   #     flag_show = False
                   #     break
                   x2 = x1+box['w']
                   y2 = y1+box['h']
                   cat = box['category_id']
                   loc = {1:  (int(x1), int(y2)),3:  (int(x1), int(y2)),9:  (int(x1), int(y2)), 10:  (int(x2), int(y1))}
                   cv2.rectangle(image, (int(x1), int(y1)), (int(x2),int(y2)), (255, 0, 0),2)
                   cv2.putText(image, '%d' % cat,
                               (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,
                               0.8,
                               (255, 0, 0), 2)
                   # if cat in [1,3,9,10]:
                   #     cv2.putText(image, '%f' % round(box['score'],2), loc[cat], cv2.FONT_HERSHEY_COMPLEX, 0.5, color[cat],
                   #                 1)
                   # cv2.rectangle(image_gamma, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                   # cv2.putText(image_gamma, '%d' % cat,(int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,0.8,(0, 255, 0), 1)
               if flag_compare:
                   # if im not in analyse.img_instance.keys():
                   #     print(im)
                   #     continue
                   anno = analyse.img_instance[im]
                   bbox = anno['bbox']
                   print('compare',bbox)
                   for box in bbox:
                       x1 = box['x1']
                       y1 = box['y1']
                       x2 = x1 + box['w']
                       y2 = y1 + box['h']
                       cat = box['category_id']
                       cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                       cv2.putText(image, '%d' % cat,
                                   (int(x2), int(y2)), cv2.FONT_HERSHEY_COMPLEX,
                                   0.8,
                                   (0, 255, 0), 2)
               cv2.namedWindow("im",cv2.WINDOW_NORMAL)
               # cv2.namedWindow("image_gamma",cv2.WINDOW_NORMAL)
               # print(image.size())
               cv2.resizeWindow('im',(1000,800))
               # cv2.resizeWindow('image_gamma',(1000,800))
               # image = cv2.resize(image,(int(image.shape[1]/3),int(image.shape[0]/3)))
               cv2.imshow('im', image)
               # cv2.imshow('image_gamma', image_gamma)
               # cv2.waitKey(0)
               key = cv2.waitKey(0)
               if key == ord('s'):
                   cv2.imwrite('/Users/urwlee/personal/Mystudy/datasets/bottle_flaw/chongqing1_round1_train1_20191223/image_clean3/'+im_name_, image)

    def distinguish_9and10(self,json_path):
        file = json.load(open(json_path,'r'))
        pass

    def cut_pinggai(self,bg_path):
        ims = os.listdir(bg_path)
        save_path = bg_path+'_without_pinggai'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for im in tqdm(ims):
            name_s = im.split('_')
            if len(name_s) == 2 and im[-3:]=='jpg':
                image = cv2.imread(bg_path+'/'+im)
                image = image[300:2700,930:4090,:]
                image = cv2.resize(image,(4096,3000))
                cv2.imwrite(os.path.join(save_path,im),image)
                # cv2.imshow('im',image)
                # cv2.waitKey(0)

    def duck_label11(self,json_path,fg_path,bg_path,save_path,json_out_path):
        # self.cut_pinggai(bg_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        fg_json = json.load(open(json_path,'r'))
        categories = fg_json['categories']
        annotations = []
        images = []
        meta = {}
        for cls in self.cla_instance['11']:
            file_name = cls['im_name'].split('/')[-1].split('.')[0].split('_')[1]
            index = cls['im_name'].split('/')[-1].split('.')[0].split('_')[-1]
            if file_name not in meta.keys():
                meta[file_name] = {index:cls['bbox']}
            else:
                meta[file_name][index] = cls['bbox']
        bg_ims = os.listdir(bg_path)
        id = 0
        for im in tqdm(meta.keys()):
            bg_name= random.choices(bg_ims)[0]
            for i in range(5):
                id += 1
                box_anno = {}
                image_anno = {}
                fg_name = os.path.join(fg_path,'imgs_'+str(im)+'_'+str(i)+'.jpg')
                fg_im = cv2.imread(fg_name)
                bg_im = cv2.imread(os.path.join(bg_path,bg_name))
                if len(meta[im])!=5:
                    print(im)
                    continue
                anno = meta[im][str(i)]
                x1 = int(anno['x1'])
                y1 = int(anno['y1'])
                x2 = int(anno['x1']+anno['w'])
                y2 = int(anno['y1']+anno['h'])
                fg_crop = fg_im[y1:y2,x1:x2,:]
                bg_im[y1:y2,x1:x2,:] = fg_crop
                save_name = 'imgs_'+im+'_'+str(i)+'.jpg'
                # cv2.imwrite(os.path.join(save_path,save_name),bg_im)
                image_anno['file_name'] = save_name
                image_anno['id'] = id
                image_anno['height'] = 3000
                image_anno['width'] = 4096
                images.append(image_anno)
                box_anno['bbox'] = [anno['x1'],anno['y1'],anno['w'],anno['h']]
                box_anno['area'] = anno['w']*anno['h']
                box_anno['category_id'] = 11
                box_anno['image_id'] = id
                box_anno['iscrowd'] = 0
                box_anno['id'] = id
                annotations.append(box_anno)
                cv2.imwrite(os.path.join(save_path,save_name),bg_im)
                # cv2.rectangle(bg_im, (int(x1), int(y1)),
                #               (int(x2), int(y2)),
                #               (0, 0, 255),
                #               2)
                # cv2.imshow(save_name,bg_im)
                # cv2.waitKey(0)
        meta = {}
        meta['images'] = images
        meta['annotations'] = annotations
        meta['categories'] = categories

        with open(json_out_path, 'w')as f:
            json.dump(meta, f)



    def draw_cls_colum(self):
        # myfont = matplotlib.font_manager.FontProperties(fname='/home/dingmingxu/Downloads/simkai_downcc/simkai.ttf')
        cls = [self.classes[key] for key in self.classes.keys()][1:]
        cls_each = [len(self.cla_instance[str(i)]) for i in range(1,len(cls)+1)]
        plt.xticks(range(1, len(cls)), cls, rotation=45)
        plt.xlabel("classes")
        plt.ylabel("numbers")
        plt.bar(cls, cls_each, color='rgb')
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_cls_colum_part(self):
        myfont = matplotlib.font_manager.FontProperties(fname='/home/dingmingxu/Downloads/simkai_downcc/simkai.ttf')
        cls = ["<40","40--60","120--420",">420"]
        cls_each = [0,0,0,0]
        for ins in self.all_instance:
            w = ins.bbox.w
            h = ins.bbox.h
            short = min(w,h)
            if short<=40:
                cls_each[0]+=1
            elif short>40 and short<=60:
                cls_each[1]+=1
            elif short>120 and short<=420:
                cls_each[2]+=1
            if short>420:
                cls_each[3]+=1
        plt.xticks(range(0, len(cls)), cls, font_properties=myfont, rotation=0)
        plt.xlabel("short line")
        plt.ylabel("numbers")
        plt.bar(cls, cls_each, color='rgb')
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_cls_colum_area(self):
        myfont = matplotlib.font_manager.FontProperties(fname='/home/dingmingxu/Downloads/simkai_downcc/simkai.ttf')
        cls = ["<40","40--60","120--420",">420"]
        cls_each = [0,0,0,0]
        for ins in self.all_instance:
            w = ins.bbox.w
            h = ins.bbox.h
            short = min(w,h)
            if short<=40:
                cls_each[0]+=1
            elif short>40 and short<=60:
                cls_each[1]+=1
            elif short>120 and short<=420:
                cls_each[2]+=1
            if short>420:
                cls_each[3]+=1
        plt.xticks(range(0, len(cls)), cls, font_properties=myfont, rotation=0)
        plt.xlabel("short line")
        plt.ylabel("numbers")
        plt.bar(cls, cls_each, color='rgb')
        plt.legend()
        plt.grid(True)
        plt.show()

    def result_from_dir(self,config2make_json,model2make_json,json_out_path,pic_path):
        index = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
        config_file = config2make_json
        checkpoint_file = model2make_json

        # build the model from a config file and a checkpoint file
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        pics = os.listdir(pic_path)
        meta = {}
        images = []
        annotations = []
        num = 0
        for im in tqdm(pics):
            num += 1
            t1 = time.time()
            img = os.path.join(pic_path, im)
            result_ = inference_detector(model, img)
            images_anno = {}
            images_anno['file_name'] = im
            images_anno['id'] = str(num)
            images.append(images_anno)
            for i, boxes in enumerate(result_, 1):
                if len(boxes):
                    defect_label = index[i]
                    for box in boxes:
                        anno = {}
                        anno['image_id'] = str(num)
                        anno['category_id'] = defect_label
                        anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                        anno['bbox'][2] = anno['bbox'][2] - anno['bbox'][0]
                        anno['bbox'][3] = anno['bbox'][3] - anno['bbox'][1]
                        anno['score'] = float(box[4])
                        annotations.append(anno)
            t2 = time.time()
            # print("time one im ",str(t2-t1))
            meta['images'] = images
            meta['annotations'] = annotations
        with open(json_out_path,'w')as f:
            json.dump(meta, f)

    def result_from_dir_multi(self,small_config,big_config,small_model,big_model,json_out_path,pic_path):
        index_small = {1: 1, 2: 9, 3: 5, 4: 3, 5: 4, 6: 2, 7: 10}
        index_big = {1: 8, 2: 6, 3: 7}
        # build the model from a config file and a checkpoint file
        small_model_inference = init_detector(small_config, small_model, device='cuda:0')
        big_model_inference = init_detector(big_config, big_model, device='cuda:0')
        pics = os.listdir(pic_path)
        meta = {}
        images = []
        annotations = []
        num = 0
        flag_big = False
        for im in tqdm(pics):
            num += 1
            img = os.path.join(pic_path, im)
            image = cv2.imread(img)
            shape = image.shape
            if image.shape[0] > 1500 and image.shape[1] > 1500:
                flag_big = True
                result_ = inference_detector(big_model_inference, image)
            else:
                result_ = inference_detector(small_model_inference, image)
            images_anno = {}
            images_anno['file_name'] = im
            images_anno['id'] = num
            images.append(images_anno)
            for i, boxes in enumerate(result_, 1):
                if len(boxes):
                    if flag_big:
                        defect_label = index_big[i]
                    else:
                        defect_label = index_small[i]
                    for box in boxes:
                        anno = {}
                        anno['image_id'] = num
                        anno['category_id'] = defect_label
                        anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                        anno['bbox'][2] = anno['bbox'][2] - anno['bbox'][0]
                        anno['bbox'][3] = anno['bbox'][3] - anno['bbox'][1]
                        anno['score'] = float(box[4])
                        if shape[1] > 700:
                            if defect_label > 5 and defect_label < 9:
                                annotations.append(anno)
                        else:
                            if defect_label < 6 or defect_label > 8:
                                annotations.append(anno)
            flag_big = False
        meta['images'] = images
        meta['annotations'] = annotations
        with open(json_out_path, 'w') as fp:
            json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))

    def ana_boxes(self):
        # style set 这里只是一些简单的style设置
        # sns.set_palette('deep', desat=.6)
        # sns.set_context(rc={'figure.figsize': (8, 5)})
        sns.set(color_codes=True)
        # data = np.random.multivariate_normal([0, 0], [[1, 2], [2, 20]], size=1000)
        # data = pd.DataFrame(data, columns=["X", "Y"])
        # mpl.rc("figure", figsize=(6, 6))
        # # sns.kdeplot(data.X, data.Y, shade=True, bw="silverman", gridsize=50, clip=(-11, 11))
        # with sns.axes_style('white'):
        #     sns.jointplot('X', 'Y', data, kind='kde')
        # plt.show()

        # x = stats.gamma(2).rvs(5000)
        # y = stats.gamma(50).rvs(5000)
        # del_instance = self.all_instance
        for i in range(7,11):
            del_instance = self.cla_instance[str(i)]
            ws = []
            hs = []
            areas = []
            ratiowh = []
            for ins in del_instance:
                ws.append(ins.bbox.w)
                hs.append(ins.bbox.h)
                areas.append(ins.bbox.h*(ins.bbox.w))
                ratiowh.append(ins.bbox.w / ins.bbox.h)
            # with sns.axes_style("dark"):
            #     sns.jointplot(ws, hs, kind="hex", )

            # bins = np.linspace(0, 2500)
            # bins = 500
            # plt.hist(areas, bins, normed=False, color="#FF0000", alpha=.9, )
            # plt.hist(ratiowh, bins, density=False, color="#C1F320", alpha=.9, )
            # plt.hist(ws, bins, normed=False, color="#FF0000", alpha=.9, )
            # plt.hist(hs, bins, normed=False, color="#C1F320", alpha=.5)

            # sns.jointplot(ws, hs, kind="reg", )
            # sns.jointplot(ws, hs, kind="hex", )

            # ---------------
            #  2维分布图
            g = sns.jointplot(x=ws, y=hs, kind="kde", color="m", cbar=True)
            g.plot_joint(plt.scatter, c="b", s=20, linewidth=0.5, marker="+")
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels("$X$", "$Y$")
            # plt.savefig('/home/remo/Desktop/cloth_flaw_detection/Round2/Analyses/%d.png' % i)
            plt.show()

    def split_json(self,json_path,s_out_path, b_out_path):
        file = json.load(open(json_path, 'r'))
        annotations = file['annotations']
        categories = file['categories']
        images = file['images']
        s_anno = {}
        b_anno = {}
        s_images = []
        b_images = []
        s_annotations = []
        b_annotations = []
        s_categories = []
        b_categories = []
        b_categories_index = []
        s_categories_index = []
        for im in images:
            width = im['width']
            height = im['height']
            id = im['id']
            if width>700 and height >500 and id not in b_anno.keys():
                b_anno[id] = {"images":im,'annotations':[]}
            if width==658 and height==492 and id not in s_anno.keys():
                s_anno[id] = {"images":im,'annotations':[]}
        b_anno_keys = b_anno.keys()
        for anno in annotations:
            image_id = anno['image_id']
            if image_id in b_anno_keys:
                b_anno[image_id]['annotations'].append(anno)
                b_categories_index.append(anno['category_id'])
            else:
                s_anno[image_id]['annotations'].append(anno)
                s_categories_index.append(anno['category_id'])
        for cat in categories:
            if cat['id'] in b_categories_index:
                b_categories.append(cat)
            else:
                s_categories.append(cat)
        index_gt = 1
        for i,key in enumerate(s_anno.keys()):
            s_anno[key]['images']['id'] = i
            s_images.append(s_anno[key]['images'])
            for anno in s_anno[key]['annotations']:
                anno['image_id'] = i
                anno['id'] = index_gt
                index_gt+=1
                s_annotations.append(anno)
        s_meta = {}
        s_meta['images'] = s_images
        s_meta['annotations'] = s_annotations
        s_meta['categories'] = s_categories
        index_gt = 1
        for i,key in enumerate(b_anno.keys()):
            b_anno[key]['images']['id'] = i
            b_images.append(b_anno[key]['images'])
            for anno in b_anno[key]['annotations']:
                anno['image_id'] = i
                anno['id'] = index_gt
                index_gt+=1
                b_annotations.append(anno)
        b_meta = {}
        b_meta['images'] = b_images
        b_meta['annotations'] = b_annotations
        b_meta['categories'] = b_categories
        with open(s_out_path,'w')as f:
            json.dump(s_meta, f)
        f.close()
        with open(b_out_path,'w')as f:
            json.dump(b_meta, f)
        f.close()

    def mergecoco2coco(self,json_file_list,result_file):
        bbox_dict = {}
        images = []
        for index, json_file in enumerate(json_file_list):
            reserve_count = 0
            with open(json_file) as f:
                dataset = json.load(f)
                images = dataset['images']
                for x in dataset['annotations']:
                    reserve_count += 1
                    x['bbox'][2] = x['bbox'][2] + x['bbox'][0] - 1
                    x['bbox'][3] = x['bbox'][3] + x['bbox'][1] - 1
                    x['bbox'].append(x['score'])
                    image_id = x['image_id']
                    category_id = x['category_id']

                    if bbox_dict.get(image_id) is None:
                        bbox_dict[image_id] = {}
                    if bbox_dict[image_id].get(category_id) is None:
                        bbox_dict[image_id][category_id] = []
                    bbox_dict[image_id][category_id].append(x['bbox'])
                print('length of dataset:', json_file, len(dataset), ' resvered box:', reserve_count)

        json_results = []
        box_count = 0
        for key in bbox_dict:
            for label in bbox_dict[key]:
                if len(bbox_dict[key][label]) > 0:
                    nms_in = np.array(bbox_dict[key][label], dtype=np.float32, copy=True)
                    keep = nms(nms_in, 0.5)
                    nms_out = nms_in[keep, :]
                    # nms_out=soft_nms(nms_in,0.5,method='linear')
                    vote_out = box_voting_ad_iou(nms_out, nms_in, thresh=0.8, scoring_method='TEMP_AVG')
                    for box in vote_out:
                        w = box[2] - box[0] + 1
                        h = box[3] - box[1] + 1
                        data = dict()
                        data['image_id'] = key
                        data['bbox'] = [round(float(box[0]), 4), round(float(box[1]), 4), round(float(w), 4),
                                        round(float(h), 4)]
                        data['score'] = float(box[4])  # /test_count
                        data['category_id'] = label
                        # data['category_id'] = label+1
                        json_results.append(data)
                        box_count += 1
        print('out box count', box_count)

        print('writing results to {}'.format(result_file))
        # mmcv.dump(outputs, args.out)
        # results2json(dataset, outputs, result_file)
        result = {}
        result['images'] = images
        result['annotations'] = json_results
        with open(result_file, 'w') as f:
            # json.dump(json_results, f)
            json_str = json.dumps(result)
            f.write(json_str)

    def add_aug_data(self,transformer, add_num=500, aug_save_path=None, json_file_path=None,label = []):
        '''
        1. 设定补充的数据量
        2. 低于这些类的才需要补充
        3. 补充增广函数
            1. 每张图片增广多少张
        :return:
        '''
        if json_file_path is None:
            raise NameError
        if not os.path.exists(aug_save_path):
            os.mkdir(aug_save_path)
        aug_json_list = []
        for cla_name, bboxes_list in self.cla_instance.items():
            if len(label)!=0 and int(cla_name) not in label:
                continue
            cla_num = len(bboxes_list)
            # 按需增广
            if cla_num >= add_num:
                continue
            # 补充数据
            cla_add_num = add_num - cla_num  #

            # 每张图进行增广
            each_num = int(math.ceil(cla_add_num / cla_num * 1.0))  # 向上取整 每张图片都进行增广
            if float(cla_add_num)/float(cla_num)<0.1:
                continue
            # 每张图进行增广扩充
            for instance in tqdm(bboxes_list, desc='cla %s ;process %d: ' % (cla_name, each_num)):  # img_box edict
                for ind in range(each_num):  # 循环多次进行增广保存
                    instance_aug,save_name = transformer.aug_img(instance['im_name'],ind,cla_name, instance = self.img_instance[instance['im_name']],aug_save_path = aug_save_path)
                    if instance_aug is not None:
                        instance_aug['im_name'] = save_name
                        aug_json_list.append(instance_aug)
        print('一共%d张图片'%len(aug_json_list))
        # # 保存aug_json 文件
        random.shuffle(aug_json_list)
        coco_json = self.create2coco(aug_json_list)
        with open(json_file_path, 'w') as f:
            json.dump(coco_json, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def create2coco(self,aug_json_list):
        meta = {}
        images = []
        categories = self.categories
        annotations = []
        num_box = 1
        for value,anno in enumerate(aug_json_list):
            image = {}
            im_name = anno['im_name']
            width = anno['width']
            height = anno['height']
            bboxes = anno['bbox']
            image['file_name'] = im_name
            image['width'] = width
            image['height'] = height
            image['id'] = value
            images.append(image)
            for boxs in bboxes:
                annotation = {}
                box = []
                box.append(boxs['x1'])
                box.append(boxs['y1'])
                box.append(boxs['w'])
                box.append(boxs['h'])
                annotation['bbox'] = box
                annotation['area'] = boxs['w']*boxs['h']
                annotation['iscrowd'] = 0
                annotation['image_id'] = value
                annotation['category_id'] = boxs['category_id']
                annotation['id'] = num_box
                num_box+=1
                annotations.append(annotation)
        meta['images'] = images
        meta['annotations'] = annotations
        meta['categories'] = categories
        return meta

    def wash(self,DATASET_PATH):
        with  open(os.path.join(DATASET_PATH, 'annotations.json'))  as f:
            json_file = json.load(f)
        print('所有图片的数量：', len(json_file['images']))
        print('所有标注的数量：', len(json_file['annotations']))
        bg_imgs = set()  # 所有标注中包含背景的图片 id

        #找到和背景有关的图片
        for c in json_file['annotations']:
            if c['category_id'] == 0:
                bg_imgs.add(c['image_id'])
        print('所有标注中包含背景的图片数量：', len(bg_imgs))
        bg_only_imgs = set()  # 只有背景的图片的 id
        for img_id in bg_imgs:
            co = 0
            for c in json_file['annotations']:
                if c['image_id'] == img_id:
                    co += 1
            if co == 1:
                bg_only_imgs.add(img_id)
        print('只包含背景的图片数量：', len(bg_only_imgs))

        #删除只有背景标注的图片
        images_to_be_deleted = []
        for img in json_file['images']:
            if img['id'] in bg_only_imgs:
                images_to_be_deleted.append(img)
                # 删除的是只有一个标注，且为 background 的的图片
        print('待删除图片的数量：', len(images_to_be_deleted))
        for img in images_to_be_deleted:
            json_file['images'].remove(img)
        print('处理之后图片的数量：', len(json_file['images']))

        #删除所有关于背景的标注
        ann_to_be_deleted = []
        for c in json_file['annotations']:
            if c['category_id'] == 0:
                ann_to_be_deleted.append(c)
        print('待删除标注的数量：', len(ann_to_be_deleted))
        for img in ann_to_be_deleted:
            json_file['annotations'].remove(img)
        print('处理之后标注的数量：', len(json_file['annotations']))

        #删除categories中关于背景的部分
        bg_cate = {'supercategory': '背景', 'id': 0, 'name': '背景'}
        json_file['categories'].remove(bg_cate)

        #标注的 id 有重复的，这里重新标号
        for idx in range(len(json_file['annotations'])):
            json_file['annotations'][idx]['id'] = idx
        with  open(os.path.join(DATASET_PATH, 'annotations_washed.json'), 'w')  as f:
            json.dump(json_file, f)
            
    def crop_img(self,json_path,image_dir,save_json):
        file = json.load(open(json_path,'r'))
        new_annotations = []
        annotations = file['annotations']
        for anno in annotations:
            new_anno = anno
            category_id = anno['category_id']
            bbox = anno['bbox']
            if category_id<6 or category_id>8:
                bbox[0] = bbox[0]-15
                bbox[1] = bbox[1]-5
            else:
                bbox[0] = bbox[0] - 1200
                bbox[1] = bbox[1] - 400
            new_anno['bbox'] = bbox
            new_annotations.append(new_anno)
        file['annotations'] = new_annotations
        new_images = []
        images = file['images']
        save_dir = image_dir + '_crop'
        for image in tqdm(images):
            image_name = image['file_name']
            im_path = os.path.join(image_dir,image_name)
            img = cv2.imread(im_path)
            if image['height']<500:
                image['width']=585
                image['height']=455
                img = img[5:460,15:600,:]
                cv2.imwrite(os.path.join(save_dir, image_name), img)
            else:
                image['width'] = 2800
                image['height'] = 2300
                img = img[400:2700,1200:4000,:]
                cv2.imwrite(os.path.join(save_dir, image_name), img)
            new_images.append(image)

        file['images'] = new_images
        with open(save_json,'w')as f:
            json.dump(file, f)

    def check_json_with_error_msg(self,pred_json, num_classes=10):
        '''
        Args:
            pred_json (str): Json path
            num_classes (int): number of foreground categories
        Returns:
            Message (str)
        Example:
            msg = check_json_with_error_msg('./submittion.json')
            print(msg)
        '''
        if not pred_json.endswith('.json'):
            return "the prediction file should ends with .json"
        with open(pred_json) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return "the prediction data should be a dict"
        if not 'images' in data:
            return "missing key \"images\""
        if not 'annotations' in data:
            return "missing key \"annotations\""
        images = data['images']
        annotations = data['annotations']
        if not isinstance(images, (list, tuple)):
            return "\"images\" format error"
        if not isinstance(annotations, (list, tuple)):
            return "\"annotations\" format error"
        for image in images:
            if not 'file_name' in image:
                return "missing key \"file_name\" in \"images\""
            if not 'id' in image:
                return "missing key \"id\" in \"images\""
        for annotation in annotations:
            if not 'image_id' in annotation:
                return "missing key \"image_id\" in \"annotations\""
            if not 'category_id' in annotation:
                return "missing key \"category_id\" in \"annotations\""
            if not 'bbox' in annotation:
                return "missing key \"bbox\" in \"annotations\""
            if not 'score' in annotation:
                return "missing key \"score\" in \"annotations\""
            if not isinstance(annotation['bbox'], (tuple, list)):
                return "bbox format error"
            if len(annotation['bbox'])==0:
                return "empty bbox"
            if annotation['category_id'] > num_classes or annotation['category_id'] < 0:
                return "category_id out of range"
        return ""

    def mk_round2_json(self,round1_json,round2_json,save_json1,save_json2):
        r1 = json.load(open(round1_json,'r'))
        r2 = json.load(open(round2_json,'r'))
        r2_categories = r2['categories']
        new_categories = []
        for categories in r2_categories:
            if categories['id'] not in [6,7,8]:
                new_categories.append(categories)
        print('new_categories',[categories['id'] for categories in new_categories])
        r2['categories'] = new_categories
        r1['categories'] = new_categories
        new_r1_annotations = []
        remove_im_id = []
        r1_annotations = r1['annotations']
        for annotations in r1_annotations:
            if annotations['category_id'] in [6,7,8]:
                remove_im_id.append(annotations['image_id'])
                continue
            new_r1_annotations.append(annotations)
        print('remove %d 6 7 8 pics'%len(remove_im_id))
        r1['annotations'] = new_r1_annotations
        r1_images = r1['images']
        new_r1_images = []
        for image in r1_images:
            if image['id'] in remove_im_id:
                continue
            new_r1_images.append(image)
        r1['images'] = new_r1_images
        with open(save_json1,'w')as f:
            json.dump(r1, f)
        with open(save_json2,'w')as f1:
            json.dump(r2, f1)

    def split_train_val(self,jsons,pic_path,flag_water=True, ratio=0.2):
        f = json.load(open(jsons[0],'r'))
        categories = f['categories']
        _,_,img_instance,_,_ = self._create_data_dict(jsons, pic_path)
        all_images = img_instance.keys()
        if flag_water:
            water_images = []
            water_images_dict = {}
            no_water_images = []
            for im in all_images:
                im_name = im.split('/')[-1]
                im_prefix = im_name.split('.')[0].split('_')
                if len(im_prefix)==3:
                    if im_prefix[1] not in water_images:
                        water_images.append(im_prefix[1])
                        water_images_dict[im_prefix[1]] = [im]
                    else:
                        water_images_dict[im_prefix[1]].append(im)
                else:
                    no_water_images.append(im)
            val_water_images = random.sample(water_images,int(len(water_images)*ratio))
            val_images_split = random.sample(no_water_images,int(len(no_water_images)*ratio))
            for i in val_water_images:
                val_images_split += water_images_dict[i]
        else:
            val_images_split = random.sample(all_images, int(len(all_images) * ratio))
        train_annotations = []
        val_annotations = []
        train_images = []
        val_images = []
        train_id = 0
        val_id = 0
        train_box_id = 0
        val_box_id = 0
        for image_name in img_instance.keys():
            flag_val = False
            if image_name in val_images_split:
                flag_val = True
                val_id+=1
            else:
                train_id+=1
            image_anno = {}
            image_anno['file_name'] = image_name.split('/')[-1]
            image_anno['width'] = img_instance[image_name]['width']
            image_anno['height'] = img_instance[image_name]['height']
            if flag_val:
                image_anno['id'] = val_id
                val_images.append(image_anno)
            else:
                image_anno['id'] = train_id
                train_images.append(image_anno)
            bbox_anno = {}
            bboxes = img_instance[image_name]['bbox']
            for bbox in bboxes:
                if flag_val:
                    val_box_id+=1
                    bbox_anno['id'] = val_box_id
                    bbox_anno['image_id'] = val_id
                else:
                    train_box_id+=1
                    bbox_anno['id'] = train_box_id
                    bbox_anno['image_id'] = train_id
                x1 = bbox['x1']
                y1 = bbox['y1']
                w = bbox['w']
                h = bbox['h']
                category_id = bbox['category_id']
                bbox_anno['bbox'] = [x1,y1,w,h]
                bbox_anno['category_id'] = category_id
                bbox_anno['area'] = w*h
                bbox_anno['iscrowd'] = 0
                if flag_val:
                    val_annotations.append(bbox_anno)
                else:
                    train_annotations.append(bbox_anno)
        train_meta = {'categories':categories,'images':train_images,'annotations':train_annotations}
        val_meta = {'categories':categories,'images':val_images,'annotations':val_annotations}
        train_path = jsons[0].split('.')[0]+'_train.json'
        val_path = jsons[0].split('.')[0]+'_val.json'
        with open(train_path,'w')as f:
            json.dump(train_meta, f)
        with open(val_path,'w')as f1:
            json.dump(val_meta, f1)

class Transform():
    def __init__(self,Analyse = None, flag_det = True, flag_seg = False, flag_debug = False):
        '''
        定义增广方法
        '''
        self.Analyse = Analyse
        self.flag_det = flag_det
        self.flag_seg = flag_seg
        self.flag_debug = flag_debug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.aug_img_seq = iaa.Sequential([
            # iaa.Fliplr(0.5),
            # iaa.Flipud(0.5),
            # iaa.Invert(1.0),
            # iaa.Crop(px=(1, 50), keep_size=True),
            # iaa.GammaContrast(1.5),
            # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
            iaa.CropAndPad(
                percent=(-0.1, 0.1),
                pad_mode='constant',
                pad_cval=(0, 0)
            ),
            # iaa.CoarseDropout((0.03, 0.15), size_percent=(0.04, 0.05)),
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                # iaa.AverageBlur(k=(2, 7)),  # blur image using local means with kernel sizes between 2 and 7
                # iaa.MedianBlur(k=(3, 11)),  # blur image using local medians with kernel sizes between 2 and 7
            ]),
            # iaa.Dropout((0.01, 0.1), per_channel=0.5),
            # iaa.Affine(translate_px={"x": (-40, 40)}, name="Affine"),
            # sometimes(iaa.Affine(rotate=90)),
            # sometimes(iaa.Affine(rotate=270))
        ], random_order=True)
        self.save_name = {}

    def aug_img(self, img_path,num,cla_name, mask=None, instance=None,aug_save_path = None):
        if self.flag_det:
            img_aug,instance_aug,save_name,flag_success = self.aug_det(img_path,instance,num,cla_name,aug_save_path)
            if flag_success:
                return instance_aug,save_name
            else:
                return None,None
        if self.flag_seg:
            self.aug_mask(img_path,mask)

    def _mk_bbs(self, instance):
        BBox = [] #[ Bounding_box, Bounding_box,]
        w = instance['width']
        h = instance['height']
        for box in instance['bbox']:
            BBox.append(BoundingBox(box['x1'], box['y1'], box['x1']+box['w'], box['y1']+box['h'],label=box['category_id']))

        return BoundingBoxesOnImage(BBox,shape = (h,w,3))

    def aug_det(self, img_path,instance,num,cla_name,aug_save_path):
        img = cv2.imread(img_path)
        bbs = self._mk_bbs(instance)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgRGB_aug, bbs_aug = self.aug_img_seq(image=imgRGB, bounding_boxes=bbs)
        bbs_aug = bbs_aug.clip_out_of_image()
        img_aug = cv2.cvtColor(imgRGB_aug, cv2.COLOR_RGB2BGR)
        # print(len(bbs_aug.bounding_boxes))
        instance_aug = {}
        flag_success = True
        if len(bbs_aug.bounding_boxes) != 0:
            bbox = []
            for i in range(len(bbs_aug.bounding_boxes)):
                box = {}
                box['x1'] = (bbs_aug.bounding_boxes[i].x1)
                box['y1'] = (bbs_aug.bounding_boxes[i].y1)
                box['w'] = (bbs_aug.bounding_boxes[i].x2-box['x1'])
                box['h'] = (bbs_aug.bounding_boxes[i].y2-box['y1'])
                box['category_id'] = bbs_aug.bounding_boxes[i].label
                bbox.append(box)
            instance_aug['bbox'] = bbox
            instance_aug['width'] = instance['width']
            instance_aug['height'] = instance['height']
            path_split = img_path.split('/')
            save_dir = aug_save_path
            save_name = path_split[-1].split('.')[0]+'_'+str(cla_name)+'_'+str(num)+'_aug.jpg'
            save_path = os.path.join(save_dir,save_name)
            if save_name not in self.save_name.keys():
                self.save_name[save_name] = 1
            else:
                self.save_name[save_name] += 1
                save_name = str(self.save_name[save_name])+'_'+save_name
            cv2.imwrite(save_path, img_aug)
        else:
            flag_success = False

        if self.flag_debug and flag_success == True:
            self.show_det(img,instance,img_aug,instance_aug)

        if flag_success == True:
            flag_success = self.check_box(instance,instance_aug)
        return img_aug,instance_aug,save_name,flag_success

    def draw_box(self,img,instance):
        bboxes = instance['bbox']
        for box in bboxes:
            cv2.rectangle(img, (int(box['x1']), int(box['y1'])), (int(box['x1'] + box['w']), int(box['y1'] + box['h'])),
                          (0, 0, 255),
                          2)
            cv2.putText(img, '%d ' % box['category_id'],
                        (int(box['x1']), int(box['y1'])), cv2.FONT_HERSHEY_COMPLEX,
                        0.8,
                        (0, 255, 0), 1)
        return img

    def show_det(self,img,instance,img_aug,instance_aug):
        img_show = self.draw_box(img,instance)
        img_aug_show = self.draw_box(img_aug,instance_aug)
        cv2.imshow('im_show',img_show)
        cv2.imshow('img_aug_show',img_aug_show)
        cv2.waitKey(0)

    def show_seg(self,img,img_aug,mask,mask_aug):
        pass

    def aug_mask(self, img,mask):
        img_aug = img
        mask_aug = mask
        if self.flag_debug:
            self.show_seg(img,img_aug,mask,mask_aug)
        flag_success = self.check_seg_iou(mask,mask_aug)
        return img_aug,mask_aug,flag_success

    def check_box(self,ins,ins_aug):
        return True

    def check_seg_iou(self,mask,mask_aug):
        return True

    def compute_iou(self,rec1, rec2):
        """
        computing IoU
        :param rec1: (x0, y0, x1, y1), which reflects
                (top, left, bottom, right)
        :param rec2: (x0, y0, x1, y1)
        :return: scala value of IoU
        """
        # computing area of each rectangles
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        top_line = max(rec1[0], rec2[0])
        left_line = max(rec1[1], rec2[1])
        bottom_line = min(rec1[2], rec2[2])
        right_line = min(rec1[3], rec2[3])

        # judge if there is an intersect
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0

    def compute_list_iou(self,ins, ins_list):
        mix_matrix = np.zeros((1, len(ins_list)))
        for i in range(len(ins_list)):
            mix_matrix[0, i] = self.compute_iou(ins.bbox, ins_list[i].bbox)
        return mix_matrix

if __name__=='__main__':
    json_path = ["E:\work\datasets\chongqingbottleflaw\chongqing1_round1_train1_20191223\chongqing1_round1_train1_20191223\\annotations.json"]
    image_path = ['E:\work\datasets\chongqingbottleflaw\chongqing1_round1_train1_20191223\chongqing1_round1_train1_20191223\images']
    aug_save_path = 'E:\work\datasets\chongqingbottleflaw\chongqing1_round1_train1_20191223\chongqing1_round1_train1_20191223\images_aug'
    json_file_path = 'E:\work\datasets\chongqingbottleflaw\chongqing1_round1_train1_20191223\chongqing1_round1_train1_20191223\\annotations_aug.json'
    analyse = Analyse(json_path,image_path)
    trans = Transform(Analyse = analyse, flag_det = True,flag_seg = False, flag_debug = False)
    analyse.add_aug_data(trans, add_num=205, aug_save_path=aug_save_path, json_file_path=json_file_path)

    # for im_path in analyse.img_instance.keys():
    #     for time in range(10):
    #         instance_aug,save_name = trans.aug_img(im_path,time, instance = analyse.img_instance[im_path])