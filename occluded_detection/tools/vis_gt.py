# encoding:utf/8
import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import os
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xml.dom.minidom
from xml.dom.minidom import Document
from tqdm import tqdm
from easydict import EasyDict as edict
import os.path as osp
import math
from tqdm import tqdm

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import getpass  # 获取用户名
import random
import copy

USER = getpass.getuser()


class Config:
    def __init__(self):
        self.json_paths = ['']  # train json
        self.val_json_paths = ['']     # val json

        self.allimg_path = ''   # 训练图片集
        self.val_img_path = ''  # 验证图片集
        self.add_num = 0        # add_aug_data 扩增数据数量

        self.result_json = '' # 模型对val 的输出结果
        self.divide_json = ''

        self.submit_json = ['']
        self.submit_path = ''
class DataAnalyze:
    '''
    bbox 分析类，
        1. 每一类的bbox 尺寸统计
        2.
    '''

    def __init__(self, cfg: Config,flag_coco=False):
        self.cfg = cfg
        self.category = {
            '破洞': 1, '水渍': 2, '油渍': 2, '污渍': 2, '三丝': 3, '结头': 4, '花板跳': 5, '百脚': 6, '毛粒': 7,
            '粗经': 8, '松经': 9, '断经': 10, '吊经': 11, '粗维': 12, '纬缩': 13, '浆斑': 14, '整经结': 15, '星跳': 16, '跳花': 16,
            '断氨纶': 17, '稀密档': 18, '浪纹档': 18, '色差档': 18, '磨痕': 19, '轧痕': 19, '修痕': 19, '烧毛痕': 19, '死皱': 20, '云织': 20,
            '双纬': 20, '双经': 20, '跳纱': 20, '筘路': 20, '纬纱不良': 20,
        }
        self.reverse_category = {
            1:'破洞', 2:'水渍',3: '三丝', 4:'结头', 5:'花板跳', 6:'百脚', 7:'毛粒',
            8:'粗经', 9:'松经', 10:'断经', 11:'吊经', 12:'粗维', 13:'纬缩', 14:'浆斑', 15:'整经结', 16:'星跳',
            17:'断氨纶', 18:'稀密档', 19:'磨痕', 20:'死皱'
        }
        self.num_classes = 20  # 前景类别

        self.all_instance, self.cla_instance, self.img_instance = self._create_data_dict(cfg.json_paths, cfg.allimg_path)
        # if hasattr(cfg, 'val_json_paths') and not  cfg.val_json_paths == '' :
        #     self.val_all_instance, self.val_cla_instance, self.val_img_instance = self._create_data_dict(cfg.val_json_paths, cfg.val_img_path)
        # else:
        #     self.val_all_instance, self.val_cla_instance, self.val_img_instance = (None, None, None)

        '''
        all_instance 
        [
            {'bbox': [2000.66, 326.38, 2029.87, 355.59],
             'defect_name': '结头',
             'name': 'd6718a7129af0ecf0827157752.jpg',
             'abs_path' : 'xxx/xxx.jpg',
             'classes': 1
             'w':1,
             'h':1,
             'area':1,
             }
        ]

        cla_instance 
            {'1':[], '2':[] }
        '''

        self.num_data = len(self.all_instance)

    def _create_data_dict(self, json_path, data_file, flag_ins_list=False):
        '''
        flag_ins_list: True 传入的json_path 为 [instance1, instance2, ] 不用json 文件读取
        :return:
            instance:
                {'bbox': [2000.66, 326.38, 2029.87, 355.59],
                 'defect_name': '结头',
                 'name': 'd6718a7129af0ecf0827157752.jpg',
                 'abs_path' : 'xxx/xxx.jpg',
                 'w':1,
                 'h':1,
                 'area':1,
                 'im_w':1
                 'im_h':2
                 }

        all_instance
            [instance1, instance2, instance3]

        cla_instance
            {'1':[instance, instances2], '2'[instance, ]}

        img_instance
            {'xx1.jpg': [instance]  'xxx.jpg':[instance, instance]}

        '''
        if flag_ins_list:
            json_path = [json_path ]# 为 [ [ins1, ins2] ] 2维数组

        all_instance = []
        key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes

        cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
        img_instance = edict()
        if isinstance(json_path, str):
            json_path = [json_path]

        if isinstance(json_path, list):
            for path in json_path:
                if flag_ins_list:
                    gt_list = path
                else:
                    gt_list = json.load(open(path, 'r'))

                for instance in tqdm(gt_list):
                    instance = edict(instance)
                    instance.classes = int(self.category[instance.defect_name])  # add classes int
                    w, h = compute_wh(instance.bbox)
                    instance.w = round(w, 2)  # add w
                    instance.h = round(h, 2)  # add h
                    instance.area = round(w * h, 2)  # add area
                    instance.abs_path = osp.join(data_file, instance.name)  # add 绝对路径
                    # im = cv2.imread(instance.abs_path)
                    # instance.im_w = im.shape[2]
                    # instance.im_h = im.shape[1]
                    instance.im_w = 2446
                    instance.im_h = 1000
                    all_instance.append(instance)  # 所有instance

                    cla_instance[str(instance.classes)].append(instance)  # 每类的instance

                    if instance.name not in img_instance.keys():  # 每张图片的instance
                        img_instance[instance.name] = [instance]
                    else:
                        img_instance[instance.name].append(instance)

        return all_instance, cla_instance, img_instance

    # def load_coco_format(self):
    #     all_instance = []
    #     key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes
    #
    #     cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
    #     img_instance = edict()
    #     if isinstance(self.cfg.json_paths, str):
    #         self.cfg.json_paths = [self.cfg.json_paths]
    #     if isinstance(self.cfg.json_paths, list):
    #         for path in self.cfg.json_paths:
    #             gt_list = json.load(open(path, 'r'))

    def ana_classes(self):
        ws_all = []
        hs_all = []
        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            ws = []
            hs = []
            for instance in bboxes_list:
                ws.append(instance.w)
                hs.append(instance.h)
                ws_all.append(instance.w)
                hs_all.append(instance.h)
            # plt.title(cla_name, fontsize='large',fontweight = 'bold')
            # plt.scatter(ws, hs, marker='x', label=cla_name, s=30)
        plt.scatter(ws_all, hs_all, marker='x', s=30)

        plt.grid(True)
        plt.show()

    def draw_cls(self):
        myfont = matplotlib.font_manager.FontProperties(fname='/home/remo/Desktop/simkai_downcc/simkai.ttf')
        cls = [i for i in range(1,self.num_classes+1)]
        cls_each = [len(self.cla_instance[str(i)]) for i in range(1,self.num_classes+1)]
        plt.xticks(range(1, len(cls) + 1), cls, font_properties=myfont, rotation=0)
        plt.bar(cls, cls_each, color='rgb')
        plt.legend()
        plt.grid(True)
        plt.show()

    def add_aug_data(self, add_num=500, aug_save_path=None, json_file_path=None):
        '''
        1. 设定补充的数据量
        2. 低于这些类的才需要补充
        3. 补充增广函数
            1. 每张图片增广多少张
        :return:
        '''
        if aug_save_path is None or json_file_path is None:
            raise NameError

        if not osp.exists(aug_save_path):
            os.makedirs(aug_save_path)

        transformer = Transformer()

        aug_json_list = []
        auged_image_dict = {}
        for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
            cla_num = len(bboxes_list)
            # 按需增广
            if cla_num >= add_num:
                continue
            # 补充数据
            cla_add_num = add_num - cla_num  #

            # 每张图进行增广
            # cla_add_num = cla_num

            each_num = int(math.ceil(cla_add_num / cla_num * 1.0))  # 向上取整 每张图片都进行增广
            # 每张图进行增广扩充
            for instance in tqdm(bboxes_list, desc='cla %s ;process %d: ' % (cla_name, each_num)):  # img_box edict
                # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

                img = cv2.imread(instance.abs_path)
                try:  # 检测图片是否可用
                    h, w, c = img.shape
                    img_info = edict({'img_h':h, "img_w":w, 'name':instance.name, 'aug_save_path':aug_save_path})
                except:
                    print("%s is wrong " % instance.abs_path)
                import copy
                for ind in range(each_num):  # 循环多次进行增广保存
                    img_info.ind = ind
                    if instance.name not in auged_image_dict.keys():
                        aug_name = '%s_aug%d.jpg' % (osp.splitext(instance.name)[0],0)  # 6598413.jpg -> 6598413_aug0.jpg, 6598413_aug1.jpg
                        auged_image_dict[instance.name] = 1
                    else:
                        auged_image_dict[instance.name] += 1
                        aug_name = '%s_aug%d.jpg' % (osp.splitext(instance.name)[0], auged_image_dict[instance.name])
                    img_info.aug_name = aug_name
                    img_ins = copy.deepcopy(self.img_instance[instance.name])
                    aug_img, img_info_tmp = transformer.aug_img(img, img_ins, img_info = img_info) # list
                    if img_info_tmp is not None:
                        aug_json_list += img_info_tmp # 融合
                        # aug_json_list.append(img_info_tmp) # 融合
        print(auged_image_dict)

        # # 保存aug_json 文件
        random.shuffle(aug_json_list)
        with open(json_file_path, 'w') as f:
            json.dump(aug_json_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def vis_gt(self, flag_show_raw_img=False, test_img=None):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0
        set_img_name = list(self.img_instance.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            instances_list = self.img_instance[img_name]
            if test_img is not None:
                instances_list = self.img_instance[test_img]

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)
            print('num gt: ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            for instance in instances_list:
                box = instance.bbox

                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_aug, '%d' % instance.classes, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)
    def vis_ins_list(self, img, ins_list, flag_show_raw_img=False):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        transformer = Transformer()
        cur_node = 0

        while True:
            # img_name = set_img_name[cur_node]
            instances_list = ins_list

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            pad_size = 200
            cv2.resizeWindow('img', 1333 + pad_size * 2, 800 + pad_size * 2)
            print('num gt: ', len(instances_list))

            ins_init = instances_list[0]
            # img_resize = cv2.resize(img, (1333,800))
            img_aug, _ = transformer(img, ins_init)
            for instance in instances_list:
                box = instance.bbox
                w, h = compute_wh(box)
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_aug, '%d ' % instance.classes, (int(box[0]), int(box[1] )),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 0), 1)

                cv2.putText(img_aug, '%d x %d' % (w, h), (int(box[0]), int(box[3] + 20)), cv2.FONT_HERSHEY_COMPLEX,
                            0.8, (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)

    def vis_res(self, flag_show_raw_img=False, test_name=None):
        '''
        可视化gt， 按键盘d 向前， 按a 向后
        :return:
        '''
        res_ins_list = self.load_res_json(self.cfg.submit_json)
        all_instance, cla_instance, img_instance = self._create_data_dict(res_ins_list, self.cfg.submit_path, flag_ins_list=True)
        transformer = Transformer()
        cur_node = 0

        set_img_name = list(img_instance.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            instances_list = img_instance[img_name]
            if test_name is not None:
                instances_list = img_instance[test_name]

            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)
            print('num res : ', len(instances_list))
            print('img_name: %s ' % (img_name))

            ins_init = instances_list[0]
            img = cv2.imread(ins_init.abs_path)
            img_aug, _ = transformer(img, ins_init)
            for instance in instances_list:
                box = instance.bbox
                cv2.rectangle(img_aug, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(img_aug, '%d %0.3f' % (instance.classes, instance.score), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            cv2.imshow('img', img_aug)
            if flag_show_raw_img:
                cv2.imshow('raw', img)

    def visRes_gt(self, gt_img_ins, res_img_ins, gtline=0.8,resline=0.8):
        '''
        1. 可视化gt 和 result 效果
        :param gt_img_ins: gt 的img_instance
        :param res_img_ins:  result 的img_instance
        :return:
        '''
        empty_ins = [
            edict(
                {'abs_path': '',
                 'area': 1,
                 'bbox': [0,0,0,0],
                 'classes': -1,
                 'defect_name': '',
                 'h': 0,
                 'name': '',
                 'w': 0,
                 'score':0}
            )
        ]

        cur_node = 0
        set_img_name = list(gt_img_ins.keys())  # 所有图片的名称的 list
        while True:
            img_name = set_img_name[cur_node]
            gt_ins_list = gt_img_ins[img_name] # gt instance 列表
            if img_name in res_img_ins.keys():
                res_ins_list = res_img_ins[img_name] # result
                res_num = len(res_ins_list)
            else:
                res_ins_list = empty_ins
                res_num = 0
            # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area

            cv2.namedWindow('img', 0)
            cv2.resizeWindow('img', 1920, 1080)


            ins_init = gt_ins_list[0]
            img = cv2.imread(ins_init.abs_path)

            for gt_ins in gt_ins_list :
                gt_box = gt_ins.bbox

                # 绘制gt
                cv2.rectangle(img, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0, 0, 255), 2)
                cv2.putText(img, '%d' % gt_ins.classes, (int(gt_box[0]), int(gt_box[1])), cv2.FONT_HERSHEY_COMPLEX,gtline,
                            (0, 0, 255), 1)

            for res_ins in res_ins_list:
                res_box = res_ins.bbox

                # 绘制result
                cv2.rectangle(img, (int(res_box[0]), int(res_box[1])), (int(res_box[2]), int(res_box[3])), (0, 255, 0), 2)
                cv2.putText(img, '%d %0.3f' % (res_ins.classes, res_ins.score), (int(res_box[2]), int(res_box[1])), cv2.FONT_HERSHEY_COMPLEX, resline,
                            (0, 255, 0), 1)

            k = cv2.waitKey(0)
            if k == ord('d'):
                cur_node += 1
            if k == ord('a'):
                cur_node -= 1
                if cur_node <= 0:
                    cur_node = 0
            if k == ord('q'):
                cv2.destroyAllWindows()
                break
            print('img_name: %s ' % (img_name))
            print('num gt : ', len(gt_ins_list))
            print('num res: ', res_num)
            cv2.imshow('img', img)
        pass

    def val_analyze(self, flag_coco_json=False):
        '''
        1. val 作为gt的 instance
        2. 模型在val 上的输出 instace
        3. 可视化比较
        4. 指标比较
        :return:q
        '''
        if not (hasattr(self.cfg, 'result_json') and self.cfg.result_json != ''):
            raise(" no result_json ")

        if flag_coco_json :
            res_ins_list = self.load_res_json(self.cfg.result_json)
            valr_all_instance, valr_cla_instance, valr_img_instance = self._create_data_dict(res_ins_list, self.cfg.val_img_path, flag_ins_list=True)
        else:
            valr_all_instance, valr_cla_instance, valr_img_instance = self._create_data_dict(self.cfg.result_json,
                                                                                             self.cfg.val_img_path,
                                                                                             flag_ins_list=False)

        self.val_all_instance, self.val_cla_instance, self.val_img_instance = self.load_coco_format(self.cfg.val_json_paths,
                                                                                                    self.cfg.val_img_path)
        self.visRes_gt(self.val_img_instance, valr_img_instance)

    def load_res_json(self, path):
        '''
        结果json 转换为 raw json 方式，
        增加 defect_name
        :param path:
        :return:
        '''

        raw_ins_list = []
        for i in range(len(path)):
            ins_list = json.load(open(path[i], 'r'))
            for instance in ins_list:
                instance = edict(instance)
                instance.defect_name = self.reverse_category[instance.category]

                raw_ins_list.append(instance)

        return raw_ins_list

    def load_coco_format(self, json_path, data_file):
        all_instance = []
        key_classes = list(range(1, self.num_classes + 1))  # 1 ... num_classes

        cla_instance = edict({str(k): [] for k in key_classes})  # key 必须是字符串
        img_instance = edict()
        if isinstance(json_path, str):
            json_path = [json_path]

        if isinstance(json_path, list):
            for path in json_path:
                gt_dict = json.load(open(path, 'r'))
                im_anno = edict()
                bbox_anno = edict()
                for anno in gt_dict['images']:
                    anno = edict(anno)
                    im_name = anno['file_name']
                    image_id = anno['id']
                    width = anno['width']
                    height = anno['height']
                    if str(image_id) not in im_anno.keys():
                        im_anno[str(image_id)] = {'im_name':im_name,'width':width,'height':height}
                for anno in gt_dict['annotations']:
                    image_id = anno['image_id']
                    category_id = anno['category_id']
                    bbox = anno['bbox']
                    bbox[2] = bbox[0]+bbox[2]
                    bbox[3] = bbox[1]+bbox[3]
                    area = anno['area']
                    if str(image_id) not in bbox_anno.keys():
                        bbox_anno[str(image_id)] = [{'category_id':category_id,'bbox':bbox,'area':area}]
                    else:
                        bbox_anno[str(image_id)].append({'category_id': category_id, 'bbox': bbox, 'area': area})

                for image_id in im_anno.keys():
                    im_anno_temp = im_anno[image_id]
                    bbox_anno_temp = bbox_anno[image_id]
                    for bbox in bbox_anno_temp:
                        instance = edict()
                        instance.imgid = image_id
                        instance.name = im_anno_temp['im_name']
                        instance.im_w = im_anno_temp['width']
                        instance.im_h = im_anno_temp['height']
                        instance.area = bbox['area']
                        instance.bbox = bbox['bbox']
                        instance.w = bbox['bbox'][2] - bbox['bbox'][0]
                        instance.h = bbox['bbox'][3] - bbox['bbox'][1]
                        instance.classes = bbox['category_id']
                        instance.abs_path = osp.join(data_file, im_anno_temp['im_name'])  # add 绝对路径
                        instance.defect_name = self.reverse_category[bbox['category_id']]
                        all_instance.append(instance)
                        cla_instance[str(bbox['category_id'])].append(instance)  # 每类的instance

                        if instance.name not in img_instance.keys():  # 每张图片的instance
                            img_instance[instance.name] = [instance]
                        else:
                            img_instance[instance.name].append(instance)
                return all_instance, cla_instance, img_instance

    def divide_trainval(self, ratio=0.2, del_json=None, del_path=None, train_json='', val_json=''):
        import random

        train_ins_list = []
        val_ins_list = []
        if del_json is None:
            divide_jsons = self.cfg.divide_json
        if del_path is None:
            del_path = self.cfg.allimg_path

        if isinstance(divide_jsons, str):
            divide_jsons = [divide_jsons]
        if not isinstance(divide_jsons, list):
            raise ("divide_jsons error !!")

        # for divide_json in divide_jsons:
        all_instance, cla_instance, img_instance = self._create_data_dict(divide_jsons, del_path)
        all_ins_keys = set(img_instance.keys())
        num_ins = len(list(all_ins_keys))
        num_val = int(num_ins  * ratio)
        print("total num : " ,num_ins)
        print("val   num : " ,num_val)
        print("train num : " ,num_ins - num_val)

        val_ins_keys = random.sample(all_ins_keys, num_val)
        train_ins_keys = set(all_ins_keys) - set(val_ins_keys)
        for ins_key in all_ins_keys:
            if ins_key in val_ins_keys:
                val_ins_list += (img_instance[ins_key])
            elif ins_key in train_ins_keys:
                train_ins_list += (img_instance[ins_key])

        with open(train_json, 'w') as f:
            json.dump(train_ins_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)
            print('train json save : ', train_json)
        with open(val_json, 'w') as f:
            json.dump(val_ins_list, f, indent=4, separators=(',', ': '), cls=MyEncoder)
            print('val json save : ', val_json)

    def pick_gt(self, normal_Images, img_save, json_file_path, condition=None, transcfg=None):


        '''
        1. 判断图片中instance 进行抠gt的条件
        2. 抠取gt得到 新的 gt图， (gt 个数， flip 等操作)
        3. (根据条件替代原来gt图(gt太少))gt 图加入到batch 中进行训练

         all_instance
        [
            {'bbox': [2000.66, 326.38, 2029.87, 355.59],
             'defect_name': '结头',
             'name': 'd6718a7129af0ecf0827157752.jpg',
             'abs_path' : 'xxx/xxx.jpg',
             'classes': 1
             'w':1,
             'h':1,
             'area':1,
             }
        ]

        :param results:
        :return:
        '''
        def _get_gtroi(list_ins, condition):
            init_ins = list_ins[0]
            img = cv2.imread(init_ins.abs_path)
            h, w, c = img.shape
            gt_roi_list = []
            new_list_ins = _check_pick_gt(list_ins, condition=condition)

            for ind, instance in enumerate(new_list_ins ):
                pick_ins = edict()
                x1, y1, x2, y2 = [round(i) for i in instance.bbox]  # x1, y1, x2, y2
                pick_ins.im_w = int(w)
                pick_ins.im_h = int(h)
                pick_ins.roi = img[y1:y2, x1:x2, :]
                pick_ins.w = x2 - x1
                pick_ins.h = y2 - y1
                pick_ins.classes = instance.classes
                pick_ins.defect_name = instance.defect_name
                pick_ins.area = pick_ins.w * pick_ins.h

                gt_roi_list.append(pick_ins)
            return gt_roi_list

        def get_normal_img(imgs_path):
            imgs = []
            if isinstance(imgs_path, str):
                imgs_path = [imgs_path]
            if isinstance(imgs_path, list):

                for imgs_p in imgs_path:
                    imgs += [osp.join(imgs_p, l) for l in os.listdir(imgs_p)]
            return imgs

        def compute_iou(rec1, rec2):
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

        def judeg_big(instance, condition=None):
            flag = False
            img_scale = condition.img_scale
            if instance.w / instance.im_w >= img_scale or instance.h / instance.im_h > img_scale:
                flag = True
            return flag

        def _check_pick_gt(img_ins, condition):
            new_img_ins = []
            mix_matrix = np.zeros((len(img_ins), len(img_ins)))
            for i in range(len(img_ins)):
                instance = img_ins[i]
                for j in range(len(img_ins)):
                    instance_temp = img_ins[j]
                    iou = compute_iou(instance.bbox,instance_temp.bbox)
                    mix_matrix[i][j] = iou
                if judeg_big(instance,condition):
                    mix_matrix[i][i] = 1
                else:
                    mix_matrix[i][i] = 0

            for i in range(len(img_ins)):
                col = mix_matrix[:,i]
                row = mix_matrix[i,:]
                if sum(col) == 0 and sum(row) == 0:
                    new_img_ins.append(img_ins[i])

            return new_img_ins

        def _get_pick_gt(all_ins, condition, transform=None, transcfg=None):
            gt_ins_list = []
            num = random.randint(condition.min_num_per_image, condition.max_num_per_image)
            num = min(num, len(all_ins))
            for i in range(num):
                gt_ins_temp = transform.aug_ins(all_ins[0], transcfg)
                gt_ins_list.append(gt_ins_temp)
                all_ins.pop(0)
            return gt_ins_list


        def compute_list_iou(ins, ins_list):
            mix_matrix = np.zeros((1, len(ins_list)))
            for i in range(len(ins_list)):
                mix_matrix[0, i] = compute_iou(ins.bbox, ins_list[i].bbox)
            return mix_matrix

        def _changebg(img, ins_list, img_info, info=None):
            '''
            1. 可以粘贴区域
            2. 去除有 iou box ， 循环n次 删除改box
                pick_ins.roi = img[y1:y2, x1:x2, :]
                pick_ins.w = instance.w
                pick_ins.h = instance.h
                pick_ins.area = instance.area
            :param img:
            :param ins_list:
            :param img_info:
            :return:
            '''
            if info is None:
                info = edict()
                info.pick = 0.5
                info.ins = 0.5

            img_tmp = copy.deepcopy(img)
            img_info.im_h, img_info.im_w, c = img_tmp.shape


            loop_times = 10
            normal_ins = []
            for ins in ins_list:
                new_ins = edict()
                paste_roi = np.asarray([0, 0, img_info.im_w - ins.w, img_info.im_h - ins.h])  # 在paste roi 中选取一点进行贴图
                if ins.w > img_info.im_w or ins.h > img_info.im_h:
                    continue
                x1s = np.random.randint(paste_roi[0], paste_roi[2], loop_times)  # 产生 loop_times个 随机点，
                y1s = np.random.randint(paste_roi[1], paste_roi[3], loop_times)

                new_ins.im_w = ins.im_w
                new_ins.im_h = ins.im_h
                new_ins.w = ins.w
                new_ins.h = ins.h
                new_ins.classes = ins.classes
                new_ins.defect_name = ins.defect_name
                new_ins.area = ins.area
                new_ins.name = img_info.name
                new_ins.abs_path = img_info.abs_path
                # new_ins.h = ins.h
                # new_ins.area = ins.w * ins.h
                new_ins.bbox = [round(l) for l in [x1s[0], y1s[0], ins.w + x1s[0], ins.h + y1s[0]]]
                if len(normal_ins) == 0:
                    x1, y1, x2, y2 = new_ins.bbox
                    pick = img_tmp[y1:y2, x1:x2, :]
                    img_tmp[y1:y2, x1:x2, :] = cv2.addWeighted(pick, info.pick, ins.roi, info.ins, 0)
                    # ins.roi
                    normal_ins.append(new_ins)
                else:

                    for i in range(1, loop_times):
                        iou_m = compute_list_iou(new_ins, normal_ins)  # 计算iou
                        if iou_m.max() == 0:  # 有iou 交叠重新选择 ins
                            x1, y1, x2, y2 = new_ins.bbox
                            pick = img_tmp[y1:y2, x1:x2, :]
                            img_tmp[y1:y2, x1:x2, :] = cv2.addWeighted(pick, info.pick, ins.roi, info.ins, 0)
                            normal_ins.append(new_ins)
                            break
                        new_ins.bbox = [x1s[i], y1s[i], ins.w + x1s[i], ins.h + y1s[i]]
                    # 超过loop times 不append

            return img_tmp, normal_ins


        if not osp.exists(img_save):
            os.makedirs(img_save)

        transformer = Transformer()


        all_pickgt = []
        all_ins = self.img_instance
        normal_imgs_path =  get_normal_img(normal_Images)
        # 1. 是否要进行抠图
        i = 0
        for img_name, list_ins in tqdm(all_ins.items()):
            i += 1
            # if i > 700: break
            all_pickgt += _get_gtroi(list_ins, condition=condition.pickgt)

        # 2. 换背景
        bg_all_ins = []
        all_pickgt_temp = copy.deepcopy(all_pickgt)
        # all_pickgt_temp = all_pickgt
        for ind , back_img_p in enumerate(tqdm(normal_imgs_path)):
            img_info = edict()
            name = osp.basename(back_img_p).replace('.jpg', '_%d.jpg' % (ind))
            bg_name = osp.join(img_save, name)
            # change instance
            img_info.name = name
            img_info.abs_path = bg_name

            back_img = cv2.imread(back_img_p)
            if len(all_pickgt_temp):
                gt_ins_list = _get_pick_gt(all_pickgt_temp, condition=condition.pickgt,
                                           transform=transformer, transcfg=transcfg)  # 获得 instanddddddddqdqdqddce list
            else:
                random.shuffle(all_pickgt)
                all_pickgt_temp = copy.deepcopy(all_pickgt)
                gt_ins_list = _get_pick_gt(all_pickgt_temp, condition=condition.pickgt,
                                           transform=transformer, transcfg=transcfg)  # 获得 instance list
            # if len(all_pickgt_temp) == 0:
            #     break
            bg_img, instance = _changebg(back_img, gt_ins_list, img_info, condition.add_info)

            # self.vis_ins_list(bg_img, instance)
            bg_all_ins += instance


            cv2.imwrite(bg_name, bg_img)

        random.shuffle(bg_all_ins)
        with open(json_file_path, 'w') as f:
            json.dump(bg_all_ins, f, indent=4, separators=(',', ': '), cls=MyEncoder)




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


class Transformer:
    def __init__(self):
        self.sum=4516
        self.aug_img_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            # iaa.AdditiveGaussianNoise(scale=0.05 * 255),
            # iaa.Crop(px=(1, 16), keep_size=False),
            # iaa.GaussianBlur(sigma=(0, 3.0)),
            # iaa.CoarseDropout(p=0.1, size_percent=0.1)
            # iaa.Invert(1.0),
            # iaa.Crop(percent=0.1)
        ], random_order=True)
        # pass

    def __call__(self, imgBGR, instance=None):
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)

        # imgRGB = self.aug_img_seq.augment_images(imgRGB)
        imgBGR_aug = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)

        # save json format
        if instance is not None:

            img_info_tmp = edict()
            img_info_tmp.bbox = instance.bbox
            img_info_tmp.defect_name = instance.defect_name
            img_info_tmp.name = instance.name
            return imgBGR_aug, img_info_tmp
        else:
            return imgBGR_aug, None

    def aug_img(self, imgBGR, instance=None, img_info = None):
        bbs = self._mk_bbs(instance, img_info)
        imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        imgRGB_aug, bbs_aug = self.aug_img_seq(image = imgRGB, bounding_boxes = bbs)
        bbs_aug = bbs_aug.clip_out_of_image()
        imgBGR_aug = cv2.cvtColor(imgRGB_aug, cv2.COLOR_RGB2BGR)
        anno_anno=edict()
        anno_img=edict()
        # for debug to show
        # imgRGB_aug_with_box = bbs_aug.draw_on_image(imgRGB_aug,size = 2)
        # imgRGB_aug_with_box = cv2.cvtColor(imgRGB_aug_with_box, cv2.COLOR_RGB2BGR)
        # imgRGB_aug_with_box = cv2.resize(imgRGB_aug_with_box,(1333,800))
        # imgRGB_with_box = bbs.draw_on_image(imgRGB, size=2)
        # imgRGB_with_box = cv2.resize(imgRGB_with_box,(1333,800))
        # imgRGB_with_box = cv2.cvtColor(imgRGB_with_box, cv2.COLOR_RGB2BGR)
        # cv2.imshow('aug',imgRGB_aug_with_box)
        # cv2.imshow('raw',imgRGB_with_box)
        # cv2.waitKey(0)

        # save json format
        if len(bbs_aug.bounding_boxes) != 0:
            instance_aug = []
            img_instance_aug=[]
            aug_abs_path = osp.join(img_info.aug_save_path, img_info.aug_name)
            for i in range(len(bbs_aug.bounding_boxes)):
                anno = edict()
                box = []
                box.append(bbs_aug.bounding_boxes[i].x1)
                box.append(bbs_aug.bounding_boxes[i].y1)
                box.append(bbs_aug.bounding_boxes[i].x2-bbs_aug.bounding_boxes[i].x1)
                box.append(bbs_aug.bounding_boxes[i].y2-bbs_aug.bounding_boxes[i].y1)
                self.sum=self.sum+1
                if self._check_box(box, img_info):
                    continue
                anno_anno.bbox = box
                anno_anno.image_id = self.sum
                anno_anno.category_id = bbs_aug.bounding_boxes[i].label

                anno_img.file_name = img_info.aug_name
                anno_img.id = self.sum
                anno_img.height = imgRGB_aug.shape[0]
                anno_img.width = imgRGB_aug.shape[1]
                instance_aug.append(anno_anno)
                img_instance_aug.append(anno_img)
            cv2.imwrite(aug_abs_path, imgBGR_aug)
            return imgBGR_aug, instance_aug, img_instance_aug
        else:
            return imgBGR_aug, None

    def _mk_bbs(self, instance, img_info):
        BBox = [] #[ Bounding_box, Bounding_box,]
        w = img_info.width
        h = img_info.height
        for box in instance.bbox:
            #box = ins.bbox
            BBox.append(BoundingBox(x1 = box.x1, y1 = box.y1, x2 = box.x2, y2 = box.y2,label=box.category_id))

        return BoundingBoxesOnImage(BBox,shape = (h,w))

    def _check_box(self, box, img_info):
        '''
        img_info = edict({'img_h':h, "img_w":w, 'name':instance.name, 'aug_save_path':aug_save_path})
        :param box:
        :param img_info:
        :return:
        '''
        # img_w = img_info.img_w
        # img_h = img_info.img_h
        # if box[0] == 0 or box[1]== 0 or box[2] == img_w or box[3] == img_h:
        #     x, y = get_center(box)
        #     min_dis = min(abs(img_h - y), abs(img_w - x),x, y)
        #     if min_dis

        pass

    def aug_ins(self, ins, transcfg=None):
        if transcfg is None:
            transcfg = edict()
            transcfg.flipProb = 1
            transcfg.fx = [1, 1]
            transcfg.fy = [1, 1]
        fx = round(random.uniform(transcfg.fx[0], transcfg.fx[1]), 2)
        fy = round(random.uniform(transcfg.fy[0], transcfg.fy[1]), 2)

        imgBGR = ins.roi
        imgBGR_aug = cv2.resize(imgBGR, dsize=(0,0), fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)

        p = np.random.randint(0, 10, 1)
        if p > transcfg.flipProb:
            type_flip = random.choice([0, 1, -1])
            imgBGR_aug = cv2.flip(imgBGR_aug, type_flip)

        ins.roi = imgBGR_aug
        h, w, c = imgBGR_aug.shape
        ins.w = w
        ins.h = h

        return ins


def compute_wh(box):
    x1, y1, x2, y2 = box
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = max(x2, 0)
    y2 = max(y2, 0)
    w = x2 - x1
    h = y2 - y1
    return w, h
