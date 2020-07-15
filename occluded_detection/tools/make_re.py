#encoding:utf/8
import sys
# sys.path.append('/home/zhangming/work/work/code/mmdetection_ding/mmcv-master')
# import mmcv
from mmdet.apis import inference_detector, init_detector
import time
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
import cv2

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


# generate result sigle model
def result_from_dir():
    index = {1:1,2:2,3:3,4:4,5:5}
    # build the model from a config file and a checkpoint file
    model = init_detector(config2make_json, model2make_json, device='cuda:0')#第一个参数是config file,第二个参数是checkpoint file
    pics = os.listdir(pic_path)
    meta = {}
    images = []
    annotations = []
    num=0
    for im in tqdm(pics):
        print(1)
        num += 1
        fname,ext=os.path.splitext(im)
        img = os.path.join(pic_path,im)
        result_ = inference_detector(model, img)
        for i, boxes in enumerate(result_, 1):
            if len(boxes):
                defect_label = index[i]
                for box in boxes:
                    anno = {}
                    anno['category_id'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['score'] = float(box[4])
                    annotations.append(anno.copy())
            # print("result: ",str(time.time()-t3))
        meta['images'] = images
        meta['annotations'] = annotations
        # with open(json_out_path, 'w') as fp:
        #     json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))
        with open(json_out_path,'w') as fp:
            json_str=json.dumps(meta)
            fp.write(json_str)
    #     for i ,boxes in enumerate(result_,1):
    #         print(boxes)
    #         # if len(boxes)!=0:
    #         defect_label = index[i]
    #         for box in boxes:
    #             print(3)
    #             anno=[]
    #             anno.append(label2name[defect_label])
    #             anno.append(file_name)
    #             anno.append(float(box[4]))  # confidence
    #             anno.append(box[0])#[round(float(i), 2) for i in box[0:4]]
    #             anno.append(box[1])
    #             anno.append(box[2])
    #             anno.append(box[3])
    #             meta.append(anno)
    #     t2 = time.time()
    #     print("time one im ",str(t2-t1))
    # with open(json_out_path, 'w') as fp:
    #     json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))


# generate result split model
def result_from_dir_multi():
    index_small = {1: 1, 2: 9, 3: 5, 4: 5, 5: 4, 6: 2, 7: 10}
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
        t1 = time.time()
        img = os.path.join(pic_path,im)
        image = cv2.imread(img)
        # print("read: ", str(time.time() - t1))
        t2 = time.time()
        if image.shape[0]>1500 and image.shape[1]>1500:
            flag_big = True
            result_ = inference_detector(big_model_inference, image)
        else:
            result_ = inference_detector(small_model_inference, image)
        # print("inference: ", str(time.time() - t2))
        # t3 = time.time()
        images_anno = {}
        images_anno['file_name'] = im
        images_anno['id'] = num
        images.append(images_anno.copy())
        for i ,boxes in enumerate(result_,1):
            if len(boxes):
                if flag_big:
                    defect_label = index_big[i]
                else:
                    defect_label = index_small[i]
                for box in boxes:
                    anno = {}
                    anno['category_id'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['score'] = float(box[4])
                    print(anno)
                    annotations.append()
                    print(annotations)
        # print("result: ",str(time.time()-t3))
        flag_big = False
    meta['images'] = images
    meta['annotations'] = annotations
    with open(json_out_path, 'w') as fp:
        json.dump(meta, fp, cls=MyEncoder, indent=4, separators=(',', ': '))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result")
    parser.add_argument("-p", "--phase", default="test", help="Test val data or test data", type=str, )
    parser.add_argument("-m", "--model", default=" ",help="Model path", type=str, )
    parser.add_argument("-c", "--config",default=" ", help="Config path", type=str, )
    parser.add_argument("-im", "--im_dir", help="Image path", type=str, )
    parser.add_argument('-o', "--out", help="Save path", type=str, )

    parser.add_argument('-s', "--split", help="Multi models", type=bool, )
    parser.add_argument('-ms', "--model_small", help="Small model", type=str, )
    parser.add_argument('-mb', "--model_big", help="Bmall model", type=str, )
    parser.add_argument('-cs', "--config_small", help="Small config", type=str, )
    parser.add_argument('-cb', "--config_big", help="Bmall config", type=str, )
    args = parser.parse_args()
    model2make_json = args.model
    config2make_json = args.config
    json_out_path = args.out
    flag_split = args.split
    if flag_split:
        small_model = args.model_small
        big_model = args.model_big
        small_config = args.config_small
        big_config = args.config_big
    if args.phase == 'test':
        pic_path = "/home/zhangming/work/work/dataset/underwater_object/val/test-A-image"
    if args.phase == "val":
        pic_path = "/home/zhangming/work/work/dataset/underwater_object/val/test-A-image"
    if flag_split:
        result_from_dir_multi()
    else:
        result_from_dir()

# python my_util/make_re.py -p val -m workdir/XXX -c round2/XXX -o results/result_XXX.json
#python bottle_flaw/round1/utils/make_re.py \
# -o=bottle_flaw/round1/results/result_2.json \
# -s=True
# -ms bottle_flaw/round1/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_small_2/epoch_12.pth \
# -mb bottle_flaw/round1/work_dirs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_big_2/epoch_12.pth \
# -cs bottle_flaw/round1/configs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_small_2.py \
# -cb bottle_flaw/round1/configs/faster_rcnn_dconv_c3-c5_x101_32x4d_fpn_1x_big_2.py