#encoding:utf/8
import sys
sys.path.append('/home/zhangming/work/work/code/mmdetection_ding/mmcv-master')
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result, init_detector,crop_inference_detector
import time
import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from copy import deepcopy as dcopy
from easydict import EasyDict as edict
from multiprocessing import Pool,Process

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


# generate result
def result_from_dir(im_dir):
    index = {1: 1, 2: 9, 3: 5, 4: 5, 5: 4, 6: 2, 7: 8, 8: 6, 9: 10, 10: 7}
    config_file = config2make_json
    checkpoint_file = model2make_json
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    pics = im_dir
    images = []
    annotations = []
    num = 0
    for im in tqdm(pics):
        num += 1
        t1 = time.time()
        img = im
        result_ = inference_detector(model, img)
        images_anno = {}
        images_anno['file_name'] = im
        images_anno['id'] = str(num)
        images.append(images_anno)
        for i ,boxes in enumerate(result_,1):
            if len(boxes):
                defect_label = index[i]
                for box in boxes:
                    anno = {}
                    anno['image_id'] = str(num)
                    anno['category_id'] = defect_label
                    anno['bbox'] = [round(float(i), 2) for i in box[0:4]]
                    anno['bbox'][2] = anno['bbox'][2]-anno['bbox'][0]
                    anno['bbox'][3] = anno['bbox'][3]-anno['bbox'][1]
                    anno['score'] = float(box[4])
                    annotations.append(anno)
        t2 = time.time()
        print("time one im ",str(t2-t1))
        annos['images'] += images
        annos['annotations'] += annotations

def devide():
    ims = os.listdir(pic_path)
    ims = [os.path.join(pic_path,im_name) for im_name in ims]
    num_images = len(ims)
    mean = int(num_images/num_process)
    ims_list = []
    for i in range(0,num_images,mean):
        ims_list.append(ims[i:i+mean])
    return ims_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result")
    parser.add_argument("-p", "--phase",default="test",help="Test val data or test data",type=str,)
    parser.add_argument("-m", "--model",help="Model path",type=str,)
    parser.add_argument("-c", "--config",help="Config path",type=str,)
    parser.add_argument("-im", "--im_dir",help="Image path",type=str,)
    parser.add_argument('-o', "--out",help="Save path", type=str,)
    parser.add_argument('-n', "--num_pro",help="num process", type=int)
    args = parser.parse_args()
    model2make_json = args.model
    config2make_json = args.config
    json_out_path = args.out
    num_process = args.num_pro
    if args.phase == 'test':
        pic_path = "/home/zhangming/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_testA_20191223/images/"
    if args.phase == "val":
        pic_path = "/home/zhangming/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_testA_20191223/images/"

    annos = {'images':[],'annotations':[]}
    ims_list = devide()
    p1 = Process(target=result_from_dir, args=(ims_list[0],))  # 必须加,号
    p2 = Process(target=result_from_dir, args=(ims_list[1],))
    # p3 = Process(target=result_from_dir, args=(ims_list[2],))  # 必须加,号
    # p4 = Process(target=result_from_dir, args=(ims_list[3],))
    # p5 = Process(target=result_from_dir, args=(ims_list[4],))  # 必须加,号
    # p6 = Process(target=result_from_dir, args=(ims_list[5],))

    p1.start()
    p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p = Pool(num_process)
    # for i in range(num_process):
    #     p.apply_async(result_from_dir,args=('ims_list[i]'))
    # p.close()
    # p.join()
    with open(json_out_path, 'w') as fp:
        json.dump(annos, fp, cls=MyEncoder, indent=4, separators=(',', ': '))
# python my_util/make_re.py -p val -m workdir/XXX -c round2/XXX -o results/result_XXX.json