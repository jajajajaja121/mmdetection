#-*-coding:utf-8-*-
from easydict import EasyDict as edict
import sys
# sys.path.append('/home/dingmingxu/work/work/code/mmdetection_ding/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
# from mmdet.models import build_detector
from mmdet.apis import inference_detector,init_detector
import time
import os
import numpy as np
# import argparse
from tqdm import tqdm
# from copy import deepcopy as dcopy
from easydict import EasyDict as edict
import copy
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os
# import imgaug as ia
# import imgaug.augmenters as iaa
# from vis_gt import Transformer
import os.path as osp
import math
import random
import csv
from bbox_utils import box_iou_stat_coco
# from numpy.random import randn
# import matplotlib as mpl
# from scipy import stats
# from scipy.stats import multivariate_normal
# from mpl_toolkits.mplot3d import Axes3D

class Analyse():

    def __init__(self,json_path, img_path):
        # self.tococo(json_path,json_path)
        self.imgpath=img_path
        self.json_path=json_path
        #下面五个参数instalce分别是将所有ins放在一个列表里,将instance按照类别分类放好,将instance按照所属图片放好
        #产生类别名与序号对应的一个字典,读取到的原本的json信息
        self.all_instance, self.cla_instance, self.img_instance, self.defect ,self.json_anno= self._create_data_dict(json_path, img_path)

    def delete_defect(self,json_save_path):
        json_file=self.json_anno
        for ann in json_file['annotations']:
            ann['bbox']=ann['bbox'][:4]
        with open(json_save_path, 'w') as f:
            json.dump(json_file, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def gen_coco_occluded(self,thr,json_save_path):
        result=box_iou_stat_coco(self.json_path)
        json_anno=self.json_anno.copy()
        images=json_anno['images']
        for im in images:
            im['occluded']=0
        for id,iou_result in result.items():
            if np.array(iou_result[thr:]).sum()>0:
                for im in images:
                    if im['id']==int(id):
                        im['occluded']=1
        with open(json_save_path, 'w') as f:
            json.dump(json_anno, f, indent=4, separators=(',', ': '), cls=MyEncoder)




    def cityperson2coco_train(self,ratio,city_path="gtBbox_cityPersons_trainval/gtBboxCityPersons",json_save_path="cityperson_train.json"):
        coco_train_json={}
        anna = {}
        ima = {}
        coco_train_json['info']=None
        coco_train_json['licenses']=None
        coco_train_json['categories']=set()
        name2category_id={"rider":1,"person (other)":2,"person group":3,"sitting person":4,"pedestrian":5,"ignore":6}
        images=[]
        annotations=[]
        train_list=os.listdir(osp.join(city_path,"train"))
        ann_id=0
        im_id=0
        for train_dic in train_list:
            train_file=os.listdir(osp.join(city_path,"train",train_dic))
            for file in train_file:
                json_path=osp.join(city_path,"train",train_dic,file)
                with open(json_path, 'r') as f:
                    anno=json.load(f)
                im_id=im_id+1
                fname, ext = osp.splitext(file)
                eff_len=len(fname)-len("gtBboxCityPersons")
                file_name = fname[:eff_len] +'leftImg8bit'+'.png'
                ima['file_name'] = osp.join(train_dic,file_name)
                ima['id'] = im_id
                ima['width'] = anno["imgWidth"]
                ima['height'] = anno["imgHeight"]
                images.append(ima.copy())
                objects=anno["objects"]
                for ele_id in range(len(objects)):
                    ann_id=ann_id+1
                    anna['image_id'] = im_id
                    anna['id']=ann_id
                    # if len(objects)>1:
                    #     anna['iscrowd']=1
                    # else:
                    #     anna['iscrowd'] = 0
                    anna['iscrowd'] = 0
                    anna['bbox']=[]
                    anna['bbox']=objects[ele_id]["bbox"]
                    anna['area']=anna['bbox'][2]*anna['bbox'][3]
                    anna['category_id'] = name2category_id[objects[ele_id]["label"]]
                    # coco_train_json['categories'].add(objects[ele_id]["label"])
                    anna['occation_ratio']=round(1-float(objects[ele_id]['bboxVis'][2]*objects[ele_id]['bboxVis'][3])/float(objects[ele_id]['bbox'][2]*objects[ele_id]['bbox'][3]),2)
                    if anna['occation_ratio']<ratio:
                        anna['occluded']=0
                    else:
                        anna['occluded']=1
                    annotations.append(anna.copy())
        coco_train_json['images']=images
        coco_train_json['annotations']=annotations
        coco_train_json['categories']=[{"id":1,"name":"rider"},{"id":2,"name":"person (other)"},{"id":3,"name":"person group"},{"id":4,"name":"sitting person"},{"id":5,"name":"pedestrian"},{"id":6,"name":"ignore"}]
        with open(json_save_path, 'w') as f:
            json.dump(coco_train_json, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def cal_occluded_num(self):
        occluded_img=0
        for ins in self.all_instance:
            if(ins['bbox']['occluded']==1):
                occluded_img=occluded_img+1
        ratio=occluded_img/len(self.all_instance)
        print(ratio)

    def cityperson2coco_val(self,city_path="gtBbox_cityPersons_trainval/gtBboxCityPersons",json_save_path="cityperson_val.json"):
        coco_train_json = {}
        anna = {}
        ima = {}
        coco_train_json['info'] = None
        coco_train_json['licenses'] = None
        coco_train_json['categories'] = set()
        name2category_id={"rider":1,"person (other)":2,"person group":3,"sitting person":4,"pedestrian":5,"ignore":6}
        images = []
        annotations = []
        val_list = os.listdir(osp.join(city_path, "val"))
        ann_id = 0
        im_id = 0
        for val_dic in val_list:
            val_file = os.listdir(osp.join(city_path,"val", val_dic))
            for file in val_file:
                json_path = osp.join(city_path,"val",val_dic, file)
                with open(json_path, 'r') as f:
                    anno = json.load(f)
                im_id = im_id + 1
                fname, ext = osp.splitext(file)
                eff_len = len(fname) - len("gtBboxCityPersons")
                file_name = fname[:eff_len] + 'leftImg8bit' + '.png'
                ima['file_name'] = osp.join(val_dic,file_name)
                ima['id'] = im_id
                ima['width'] = anno["imgWidth"]
                ima['height'] = anno["imgHeight"]
                images.append(ima.copy())
                objects = anno["objects"]
                for ele_id in range(len(objects)):
                    ann_id = ann_id + 1
                    anna['image_id'] = im_id
                    anna['id'] = ann_id
                    # if len(objects) > 1:
                    #     anna['iscrowd'] = 1
                    # else:
                    #     anna['iscrowd'] = 0
                    anna['iscrowd'] = 1
                    anna['bbox'] = []
                    anna['bbox'] = objects[ele_id]["bbox"]
                    anna['area'] = anna['bbox'][2] * anna['bbox'][3]
                    anna['category_id'] = name2category_id[objects[ele_id]["label"]]
                    # coco_train_json['categories'].add(objects[ele_id]["label"])
                    anna['occation_ratio'] = round(1 - float(objects[ele_id]['bboxVis'][2] * objects[ele_id]['bboxVis'][3])/float(objects[ele_id]['bbox'][2] * objects[ele_id]['bbox'][3]),2)
                    annotations.append(anna.copy())
        coco_train_json['images'] = images
        coco_train_json['annotations'] = annotations
        coco_train_json['categories'] = coco_train_json['categories']=[{"id":1,"name":"rider"},{"id":2,"name":"person (other)"},{"id":3,"name":"person group"},{"id":4,"name":"sitting person"},{"id":5,"name":"pedestrian"},{"id":6,"name":"ignore"}]
        with open(json_save_path, 'w') as f:
            json.dump(coco_train_json, f, indent=4, separators=(',', ': '), cls=MyEncoder)
    def gene_all_ratio_file(self):
        ratio_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        annotations=self.json_anno['annotations']
        for ratio in ratio_list:
            annotations_ratio=[]
            for ann in annotations:
                if ann['occation_ratio']>ratio:
                    ann['occluded']=1
                else:
                    ann['occluded']=0
                annotations_ratio.append(ann.copy())
            result=self.json_anno
            result['annotations']=annotations_ratio
            file_name='cityperson_train_'+str(ratio)+'.json'
            with open(file_name,'w') as f:
                json.dump(result, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def gen_occluded_img(self):
        for im in tqdm(self.img_instance.keys()):
            img_path = osp.join(self.imgpath, im)
            new_path="/home/dingmingxu/work/work/dataset/leftImg8bit_trainvaltest(occ)/leftImg8bit/train"
            img_new_path=osp.join(new_path,im)
            dir_path,img_name=os.path.split(img_new_path)
            if os.path.lexists(dir_path):
                pass
            else:
                os.makedirs(dir_path)
            # print(img_path)
            # flag_show = False
            image = cv2.imread(img_path)
            anno = self.img_instance[im]
            bbox = anno['bbox']
            for box in bbox:
                x1 = box['x1']
                y1 = box['y1']
                w=box['w']
                h=box['h']
                x2 = x1 + box['w']
                y2 = y1 + box['h']
                cat = box['category_id']
                new_x=random.randint(int(x1+0.25*(x2-x1)),int(x1+0.75*(x2-x1)))
                new_y=random.randint(int(y1+0.25*(y2-y1)),int(y1+0.75*(y2-y1)))
                new_w=random.randint(int(w/4),w)
                new_h=random.randint(int(h/4),h)
                flag=random.randint(0,1)
                if flag==0:
                    image[new_y:new_y+new_h,new_x:new_x+new_w,:]=0
                else:
                    image[new_y:new_y - new_h, new_x:new_x - new_w, :] = 0
            cv2.imwrite(img_new_path,image)
                # if label != -1:
                #     if cat != label:
                #         continue
                # flag_show = True
                # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # cv2.putText(image, '%d' % cat,
                #             (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,
                #             0.8,
                #             (0, 255, 0), 1)
            # if flag_show:

            # cv2.namedWindow("im", cv2.WINDOW_NORMAL)
            # # print(image.size())
            # cv2.resizeWindow('im', (1000, 800))
            # # image = cv2.resize(image,(int(image.shape[1]/3),int(image.shape[0]/3)))
            # cv2.imshow('im', image)
            # cv2.waitKey(0)

    def ana_occ(self,json_path):
        with open(json_path,'r') as f:
            ann=json.load(f)
        images=ann['images']
        sum=0
        for im in images:
            if im['occluded']==1:
                sum=sum+1
        print(sum)

    def val_devide(self,val_save_json,occluded_json,unoccluded_json):
        with open(val_save_json,'r') as f:
            ann=json.load(f)
        images=ann['images']
        annotations=ann['annotations']
        ann_occluded=ann.copy()
        ann_unoccluded=ann.copy()
        images_occluded=[]
        images_unoccluded=[]
        annotations_occluded=[]
        annotations_unoccluded=[]
        occluded_index=[]
        for im in images:
            if im['occluded']==1:
                occluded_index.append(im['id'])
                images_occluded.append(im.copy())
            else:
                images_unoccluded.append(im.copy())
        for anno in annotations:
            if anno['image_id'] in occluded_index:
                annotations_occluded.append(anno.copy())
            else:
                annotations_unoccluded.append(anno.copy())
        ann_occluded['images']=images_occluded
        ann_occluded['annotations']=annotations_occluded
        ann_unoccluded['images']=images_unoccluded
        ann_unoccluded['annotations']=annotations_unoccluded
        with open(occluded_json, 'w') as f:
            json.dump(ann_occluded, f, indent=4, separators=(',', ': '), cls=MyEncoder)
        with open(unoccluded_json, 'w') as f:
            json.dump(ann_unoccluded, f, indent=4, separators=(',', ': '), cls=MyEncoder)
    def check_result(self,ann_file,coco_file,val_path,ca2name):
        images=os.listdir(val_path)
        anns=json.load(open(ann_file,'r'))
        cocos=json.load(open(coco_file,'r'))
        images_ann=anns['images']
        file_name2id={im_index['file_name']:im_index['id'] for im_index in images_ann}
        ann_img={}
        coco_img={}
        for i in range(len(images)):
            ann_img[i]=[]
            coco_img[i]=[]
        for ann in anns['annotations']:
            ann_img[ann['image_id']].append(ann)
        for _coco in cocos['annotations']:
            coco_img[_coco['image_id']].append(_coco)
        for im in images:
            img_path = osp.join(val_path, im)
            image = cv2.imread(img_path)
            im_anno=ann_img.get(file_name2id[im])
            im_coco=coco_img.get(file_name2id[im])
            # im_anno = ann_img.get(0)
            # im_coco = coco_img.get(0)
            for anno in im_anno:
                defect_label=ca2name[anno['category_id']]
                cv2.rectangle(image, (int(anno['bbox'][0]), int(anno['bbox'][1])), (int(anno['bbox'][0]+anno['bbox'][2]), int(anno['bbox'][1]+anno['bbox'][3])), (0, 0, 255),
                              1)  # hua chu xiang ying de fang kuang
                cv2.putText(image, '%s %0.3f' % (defect_label, float(anno['score'])),
                            (int(anno['bbox'][0]), int(anno['bbox'][1])), cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0), 1)
            for coco in im_coco:
                # defect_label_coco=ca2name[coco['category_id']]
                cv2.rectangle(image, (int(coco['bbox'][0]), int(coco['bbox'][1])), (int(coco['bbox'][0]+coco['bbox'][2]), int(coco['bbox'][1]+coco['bbox'][3])), (255, 0, 0),
                              1)  # hua chu xiang ying de fang kuang
                # cv2.putText(image, '%s %0.3f' % (defect_label_coco, float(anno['score'])),
                #             (int(coco['bbox'][0]), int(anno['bbox'][1])), cv2.FONT_HERSHEY_COMPLEX,
                #             0.8,
                #             (0, 255, 0), 1)
            cv2.namedWindow('im', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('im', (1000, 600))
            cv2.imshow('im', image)
            cv2.waitKey(0)


    def vis_result(self,models, configs, pic_dir, index,cat2name,flag_traindata = False):
        # index = {1: 1, 2: 9, 3: 5, 4: 5, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
        # index = {1: 8, 2: 6, 3: 7}
        init_detectors = []
            # build the model from a config file and a checkpoint file
        init_detectors.append(init_detector(configs, models, device='cuda:0'))
        images = os.listdir(pic_dir)#fan hui pic_dir zhong baohan de wen jian ming de lie biao
        for img in tqdm(images):#tqdm ke yi zai xun huan zhong jia ru yi tiao jindu ti shi xin xi
            img_path=osp.join(pic_dir,img)
            image=cv2.imread(img_path)
            for num in range(len(init_detectors)):
                img_temp = copy.deepcopy(image)#deepcopy shen fu zhi hui jiang yuan shi dui xiang zhong suo you yuansu quan dou fang zai yi ge xin de di zhi
                # if flag_traindata:#ru guo xun lian shuju kai qi
                #     if img_path not in self.img_instance.keys():#ru guo tu xiang de lu jing bu zai tu pian lei bie zhong
                #         print("Error pleaase check if you use test data not train data")
                #     annos = self.img_instance[img_path]
                #     if len(annos['bbox']):#ru guo bbox de changdu bu wei 0
                #         for anno in annos['bbox']:
                #             cv2.rectangle(img_temp, (int(anno['x1']), int(anno['y1'])), (int(anno['x1']+anno['w']), int(anno['y1']+anno['h'])), (255, 0, 0),
                #                           2)#li yong opencv jiang jiance dedao de jie guo hua chu lai
                #             cv2.putText(img_temp,'%i'%anno['category_id'],(int(anno['x1']), int(anno['y1'])), cv2.FONT_HERSHEY_COMPLEX,0.8,(255, 0, 0), 1)#jiang xiang ying lei bie ming xian shi zai kuang shang mian
                result = inference_detector(init_detectors[num], img_temp)#detector ji mo xing
                for i, boxes in enumerate(result, 1):#enumerate you liang ge can shu di er ge shi bian li kai qi dian
                    if len(boxes):
                        defect_label = cat2name[index[i]]#xia ci biao qian
                        for box in boxes:
                            cv2.rectangle(img_temp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                                          2)#hua chu xiang ying de fang kuang
                            cv2.putText(img_temp, '%s %0.3f' % (defect_label, float(box[4])),
                                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_COMPLEX,
                                        0.8,
                                        (0, 255, 0), 1)
                cv2.namedWindow('im',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('im',(1000,600))
                cv2.imshow('im',img_temp)
                cv2.waitKey(0)


    def mergecoco2coco(self, json_file_list, result_file):
        from bbox_utils import py_cpu_nms as nms
        from bbox_utils import box_voting, box_voting_ad_iou
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
                    keep = nms(nms_in, 0.4)
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
                        data['id']=box_count
                        data['area']=data['bbox'][2]*data['bbox'][3]
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

    def vis_json(self,json_paths):
        anno_o = open(json_paths,'r')
        anno = json.load(anno_o)
        for key in anno.keys():
            if isinstance(anno[key],list):
                print(key,anno[key][0])
            else:
                print(anno[key])
            if key=="categories":
                print(anno[key])
        return anno['categories']
    def generate_class(self,json_paths):
        result={}
        categories=self.vis_json(json_paths)
        for cat in categories:
            result[cat["id"]]=cat["name"]
        print(result)
    def coco_eval_by_file(self,ann_file,result_file,iou_type='bbox'):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        metric = 'proposal'
        coco_GT=COCO(ann_file)
        coco_DT=COCO(result_file)
        cocoEval=COCOeval(coco_GT,coco_DT,iou_type)
        cocoEval.evaluate()
        accu=cocoEval.accumulate()
        suma=cocoEval.summarize()


    def make_result(self,config2make_json,model2make_json,pic_path,json_out_path,index,ann_file):
        # index = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        # build the model from a config file and a checkpoint file
        with open(ann_file,'r') as f:
            json_ann=json.load(f)
        model = init_detector(config2make_json, model2make_json,
                              device='cuda:0')  # 第一个参数是config file,第二个参数是checkpoint file
        meta = {}
        images = json_ann['images']
        annotations = []
        num = 0
        for image in tqdm(images):
            im=image['file_name']
            im_id=image["id"]
            img = os.path.join(pic_path, im)
            result_ = inference_detector(model, img)
            for i, boxes in enumerate(result_, 1):
                if len(boxes):
                    defect_label = index[i]
                    for box in boxes:
                        anno = {}
                        anno['category_id'] = defect_label
                        anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                        anno['area'] = anno['bbox'][2] * anno['bbox'][3]
                        anno['bbox'][2]=anno['bbox'][2]-anno['bbox'][0]
                        anno['bbox'][3]=anno['bbox'][3]-anno['bbox'][1]
                        anno['score'] = float(box[4])
                        anno['image_id']=im_id
                        anno['id']=num
                        num=num+1
                        annotations.append(anno.copy())
                # print("result: ",str(time.time()-t3))
        meta['images'] = images
        meta['annotations'] = annotations
        meta['categories']=json_ann['categories']
        # with open(json_out_path, 'w') as fp:
        #     json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))
        with open(json_out_path, 'w') as fp:
            json_str = json.dumps(meta)
            fp.write(json_str)



    def _create_data_dict(self,json_path, img_path):#chuang jian shu ju lie biao
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
        cla_instance = edict()#edict ke yi shi xian yi shu xing de fang shi lai cha kan zi dian de zhi
        img_instance = edict()
        defect = edict()
        info_temp = []
        im_name_to_index = {}
        anno_o = open(json_path,'r')
        anno = json.load(anno_o)
        # get image info
        '''
        [1:'image1',
         2:'imag2',
         ...
        ]
        '''
        for im_info in anno['images']:#dui mei yi zhang tu pian jin xing bian li
            info_temp.append([])#每一张图片创建一个列表，info_temo为一个嵌套的列表
            file_name = im_info['file_name']#du qu tu pian ming
            id = im_info['id']#du qu tu pian id
            height = im_info['height']#du qu tu pian gao du
            width = im_info['width']#du qu tu pian kuan du
            file_path = os.path.join(img_path,file_name)
            im_name_to_index[id] = [file_path,file_name,height,width,id]#这个字典中嵌套列
            # 表，列表中存储着文件的位置文件名，图片的长宽以及id，这里的id就是图片的id
        # get anno info
        '''
        {
            1:[{x1: 1,y1: 2,w: 3,h: 4,category_id:3}, {x1: 1,y1: 2,w: 3,h: 4,category_id:3}, ...],
            2:[{x1: 1,y1: 2,w: 3,h: 4,category_id:3}, ...]
            ...
        }
        '''
        for im_info in anno['annotations']:#bian li tu pian de biao zhu
            image_id = im_info['image_id']#image_id从1开始
            bbox = im_info['bbox']#huoqu tu pian bbox
            category_id = im_info['category_id']#huo qu mu lu id
            bbox.append(category_id)#jiang dui ying fen lei de xin xi jia ru dao bbox zhong
            ins = edict()#jiang bbox de si ge zuo biao jia ru dao ins zhong
            # ins.occluded=im_info['occluded']
            ins.x1 = bbox[0]
            ins.y1 = bbox[1]
            # ins.x2 = bbox[2]#+bbox[0]
            # ins.y2 = bbox[3]#+bbox[1]
            ins.w=bbox[2]
            ins.h=bbox[3]
            ins.category_id = category_id #jiang mu lu id jia ru dao ins zhong
            info_temp[image_id-1].append(ins)#将同一张图片的bbox以及分类加入同一个列表中
            #因为列表索引从0开始，而image_id从1开始，所以需要减一
        # get defect info
        for im_info in anno['categories']:
            defect[str(im_info['id'])] = im_info['name']
        for index in im_name_to_index.keys():
            abs_path = im_name_to_index[index][0]#图片路径
            name=im_name_to_index[index][1]#图片名
            w = im_name_to_index[index][2]#宽度
            h = im_name_to_index[index][3]#高度
            id=im_name_to_index[index][4]#图片id
            img_instance[name] = {}
            img_instance[name].bbox = info_temp[index-1]#将图片的分类以及bbox加入到相同图片名的字典中
            img_instance[name].width = w
            img_instance[name].height = h
            img_instance[name].id=id
            '''
                img_instance{
                image_name_1:{
                    bbox:[x0,y0,w,h,category_id]
                    width:w
                    height:h
                    id:id
                            }
                image_name_2:{
                    bbox:{
                            x1:左下角x
                            y1:左下角y
                            w:box宽度
                            h:box高度
                            category_id:分类
                            }
                    width:w
                    height:h
                    id:id
                            }
                ........
                }
            '''
            for box in img_instance[name].bbox:
                ins = edict()
                ins.abs_path = abs_path
                ins.name=name
                ins.bbox = box
                ins.width = img_instance[name].width
                ins.height = img_instance[name].height
                ins.id=id
                cat = box['category_id']
                if str(cat) in cla_instance.keys():
                    cla_instance[str(cat)].append(ins)
                else:
                    cla_instance[str(cat)] = [ins]
                all_instance.append(ins)
                '''
                1:[
                {
                abs_path:abs_path
                name:name
                bbox:
                    {
                    x1:左下角x
                    y1:左下角y
                    w:box宽度
                    h:box高度
                    category_id:分类
                    }
                width:图片宽度
                height:图片高度
                id:
                }
                ....
                ]
                2:[....]
                3:[.....]
                4:[....]
                
                '''
        return all_instance, cla_instance, img_instance, defect,anno
    #这里返回的四类分别是，annotation列表，images列表，
    # 根据类别为字典储存的annotations，defect是一个id到name的类别映射

    def vis_label(self,label = -1):
        for im in self.img_instance.keys():
           img_path=osp.join(self.imgpath,im)
           # print(img_path)
           # flag_show = False
           image = cv2.imread(img_path)
           anno = self.img_instance[im]
           bbox = anno['bbox']
           for box in bbox:
               x1 = box['x1']
               y1 = box['y1']
               x2 = x1+box['w']
               y2 = y1+box['h']
               cat = box['category_id']
               if label != -1:
                   if cat != label:
                       continue
               # flag_show = True
               cv2.rectangle(image, (int(x1), int(y1)), (int(x2),int(y2)), (0, 0, 255),2)
               cv2.putText(image, '%d' % cat,
                           (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX,
                           0.8,
                           (0, 255, 0), 1)
           # if flag_show:
           cv2.namedWindow("im",cv2.WINDOW_NORMAL)
           # print(image.size())
           cv2.resizeWindow('im',(1000,800))
           # image = cv2.resize(image,(int(image.shape[1]/3),int(image.shape[0]/3)))
           cv2.imshow('im', image)
           cv2.waitKey(0)

    def draw_cls_colum(self):
        myfont = matplotlib.font_manager.FontProperties(fname='/home/dingmingxu/Downloads/simkai_downcc/simkai.ttf')
        cls = [self.classes[key] for key in self.classes.keys()]
        cls_each = [len(self.cla_instance[str(i)]) for i in range(len(cls))]
        plt.xticks(range(0, len(cls)), cls, font_properties=myfont, rotation=45)
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

    # def result_from_dir(self,config2make_json,model2make_json,json_out_path,pic_path):
    #     index = {1: 1, 2: 9, 3: 5, 4: 5, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
    #     config_file = config2make_json
    #     checkpoint_file = model2make_json
    #
    #     # build the model from a config file and a checkpoint file
    #     model = init_detector(config_file, checkpoint_file, device='cuda:0')
    #     pics = os.listdir(pic_path)
    #     meta = {}
    #     images = []
    #     annotations = []
    #     num = 0
    #     for im in tqdm(pics):
    #         num += 1
    #         t1 = time.time()
    #         img = os.path.join(pic_path, im)
    #         result_ = inference_detector(model, img)
    #         images_anno = {}
    #         images_anno['file_name'] = im
    #         images_anno['id'] = str(num)
    #         images.append(images_anno)
    #         for i, boxes in enumerate(result_, 1):
    #             if len(boxes):
    #                 defect_label = index[i]
    #                 for box in boxes:
    #                     anno = {}
    #                     anno['image_id'] = str(num)
    #                     anno['category_id'] = defect_label
    #                     anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
    #                     anno['bbox'][2] = anno['bbox'][2] - anno['bbox'][0]
    #                     anno['bbox'][3] = anno['bbox'][3] - anno['bbox'][1]
    #                     anno['score'] = float(box[4])
    #                     annotations.append(anno)
    #         t2 = time.time()
    #         # print("time one im ",str(t2-t1))
    #         meta['images'] = images
    #         meta['annotations'] = annotations
    #     with open(json_out_path,'w')as f:
    #         json.dump(meta, f)#jiang zi dian zhuan hua wei zifu chuan

    def tococo(self,json_path,save_path):
        anno_o = open(json_path, 'r')
        anno = json.load(anno_o)
        # anno_new=anno
        # images_new=[]
        # annotations_new=[]
        for id,image in enumerate(anno['images']):
            print(id)
            for ann in anno['annotations']:
                if ann['image_id']==image['id']:
                    ann['image_id']=id
            image['id']=id
        with open(save_path, 'w') as f:
            json.dump(anno, f, indent=4, separators=(',', ': '), cls=MyEncoder)

    def remove_bg_and_tococo(self,json_path,save_path):
        file = json.load(open(json_path,'r'))
        new_meta = {}
        annotations = file['annotations']
        categories = file['categories']
        images = file['images']
        new_images = []
        new_annotations = []
        new_categories = []
        pic_index = []
        image_index = {}
        for anno in annotations:
            if anno['category_id'] != 0:
                new_annotations.append(anno)
                pic_index.append(anno['image_id'])
        for cat in categories:
            if cat['id']!=0:
                new_categories.append(cat)
        not_ignore = 0
        raw = 0
        for im in images:
            raw+=1
            if im['id'] in pic_index:
                not_ignore+=1
                image_index[im['id']] = not_ignore
                im['id'] = not_ignore
                new_images.append(im)
        for i,anno in enumerate(new_annotations,1):
            anno['image_id'] = image_index[anno['image_id']]
            anno['id'] = i
        print('There are %d images originally ,now there are %d images stay true'%(raw,not_ignore))
        new_meta['images'] = new_images
        new_meta['annotations'] = new_annotations
        new_meta['categories'] = new_categories
        with open(save_path,'w')as f:
            json.dump(new_meta, f)



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
        ws = []
        hs = []
        areas = []
        ratiowh = []
        for _,ann in self.cla_instance.items():

            for ins in ann:
                ws.append(ins.bbox.w)
                hs.append(ins.bbox.h)
                areas.append(ins.bbox.h*(ins.bbox.w))
                ratiowh.append(round(ins.bbox.w / ins.bbox.h))
        ratio_set=set(ratiowh)
        ratio_list=list(ratio_set)
        ratio_list.sort()
        ratio_count=[]
        for ratio in ratio_list:
            ratio_count.append(ratiowh.count(ratio))
        print(ratio_count)
            # with sns.axes_style("dark"):
            #     sns.jointplot(ws, hs, kind="hex", )

            # bins = np.linspace(0, 2500)
        # bins = list(range(len(ratio_count)))

        # plt.hist(areas, bins, normed=False, color="#FF0000", alpha=.9, )
        # plt.plot(bins,ratio_count)
        # plt.xlabel('ratio')
        ratio_set = set(ws)
        ratio_list = list(ratio_set)
        ratio_list.sort()
        ratio_count = []
        for ratio in ratio_list:
            ratio_count.append(ws.count(ratio))
        plt.xlabel('w')
        plt.ylabel('num')
        bins = list(range(len(ratio_count)))
        plt.plot(ratio_count,bins)
        plt.show()
        plt.savefig('/home/dingmingxu/work/work/code/mmdetection_ding/cityperson/analyse/ana_analyse.png')
        # plt.hist(ws, bins, normed=False, color="#FF0000", alpha=.9, )
        # plt.hist(hs, bins, normed=False, color="#C1F320", alpha=.5)

        # sns.jointplot(ws, hs, kind="reg", )
        # sns.jointplot(ws, hs, kind="hex", )

        # ---------------
        #  2维分布图
        # g = sns.jointplot(x=ws, y=hs, kind="kde", color="m", cbar=True)
        # g.plot_joint(plt.scatter, c="b", s=20, linewidth=0.5, marker="+")
        # g.ax_joint.collections[0].set_alpha(0)
        # g.set_axis_labels("$X$", "$Y$")
        # plt.savefig('/home/zhangming/work/work/code/mmdetection_xiong/underwater_detection/utils/ana_analyse.png')




    # def add_aug_data(self, add_num=1000, aug_save_path=None, json_save_path=None):
    #     '''
    #            1. 设定补充的数据量
    #            2. 低于这些类的才需要补充
    #            3. 补充增广函数
    #                1. 每张图片增广多少张
    #            :return:
    #            '''
    #     if aug_save_path is None or json_save_path is None:#
    #         raise NameError
    #
    #     if not osp.exists(aug_save_path):
    #         os.makedirs(aug_save_path)
    #
    #     transformer = Transformer()
    #
    #     img_list=self.anno_image
    #     annotations_list=self.anno_anno
    #     json_all=edict()
    #     auged_image_dict = {}
    #     for cla_name, bboxes_list in self.cla_instance.items():  # str '1': [edict()]
    #         cla_num = len(bboxes_list)
    #         # 按需增广
    #         if cla_num >= add_num:
    #             continue
    #         # 补充数据
    #         cla_add_num = add_num - cla_num  #
    #
    #         # 每张图进行增广
    #         # cla_add_num = cla_num
    #
    #         each_num = int(math.ceil(cla_add_num / cla_num * 1.0))  # 向上取整 每张图片都进行增广
    #         # 每张图进行增广扩充
    #         for instance in tqdm(bboxes_list, desc='cla %s ;process %d: ' % (cla_name, each_num)):  # img_box edict
    #             # instance attr: bbox, defect_name, name, abs_path, classes, w, h, area
    #
    #             img = cv2.imread(instance.abs_path)
    #             try:  # 检测图片是否可用
    #                 h, w, c = img.shape
    #                 img_info = edict({'height': h, "width": w, 'file_name': instance.name, 'id': instance.id,'aug_save_path':aug_save_path})
    #                 #annotations_info=edict({'area':(instance.bbox.w * instance.bbox.h) ,'iscrowd': 0,'image_id':instance.id ,'bbox':instance.bbox})
    #             except:
    #                 print("%s is wrong " % instance.abs_path)
    #             import copy
    #             for ind in range(each_num):  # 循环多次进行增广保存
    #                 img_info.id = ind
    #                 if instance.name not in auged_image_dict.keys():
    #                     aug_name = '%s_aug%d.jpg' % (
    #                     osp.splitext(instance.name)[0], 0)  # 6598413.jpg -> 6598413_aug0.jpg, 6598413_aug1.jpg
    #                     auged_image_dict[instance.name] = 1
    #                 else:
    #                     auged_image_dict[instance.name] += 1
    #                     aug_name = '%s_aug%d.jpg' % (osp.splitext(instance.name)[0], auged_image_dict[instance.name])
    #                 img_info.aug_name = aug_name
    #                 img_ins = copy.deepcopy(self.img_instance[instance.name])
    #                 aug_img, img_instance_info_tmp,img_info_tmp = transformer.aug_img(imgBGR=img, instance=img_ins, img_info=img_info)  # list
    #                 if img_info_tmp is not None:
    #                     annotations_list += img_instance_info_tmp# 融合
    #                     img_list+=img_info_tmp
    #                     # aug_json_list.append(img_info_tmp) # 融合
    #     random.shuffle(img_list)
    #     random.shuffle(annotations_list)
    #     json_all.images=img_list
    #     json_all.annotations=annotations_list
    #     json_all.categories=self.anno_categories
    #     print(img_list[1])
    #     # # 保存aug_json 文件
    #     with open(json_save_path, 'w') as f:
    #         json.dump(json_all, f, indent=4, separators=(',', ': '), cls=MyEncoder)



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

def check_json_with_error_msg(pred_json, num_classes=10):
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