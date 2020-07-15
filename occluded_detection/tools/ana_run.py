from analyse import Analyse
from analyse import check_json_with_error_msg
if __name__ == "__main__":
    train_json = "/home/dingmingxu/work/work/dataset/COCO/annotations/instances_train5000_right.json"
    val_json = "/home/dingmingxu/work/work/dataset/COCO/annotations/instances_val5000_right.json"
    img_path = "/home/dingmingxu/work/work/dataset/COCO/val"
    checkpoin_file = '/home/dingmingxu/work/work/code/mmdetection_ding/cityperson/work_dirs/retinanet_r50_fpn_1x/epoch_12.pth'
    config_ori_path = '/home/dingmingxu/work/work/code/mmdetection_ding/cityperson/configs/retinanet_r50_fpn_1x_ori.py'
    # anno_path = ["/home/dingmingxu/work/work/dataset/chongqing_bottle_flaw/chongqing1_round1_train1_20191223/annotations_without_bg.json"]
    # pic_path = "/home/zhangming/work/work/dataset/underwater_object/train/train/image"
    checkpoint='/home/dingmingxu/work/work/code/mmdetection_ding/pretrainmodels/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
    train_save_json="/home/dingmingxu/work/work/dataset/COCO/annotations/instances_train5000_right_occluded.json"
    save_path="/home/dingmingxu/work/work/dataset/COCO/annotations/val_occluded_nodefe.json"
    val_save_json="/home/dingmingxu/work/work/dataset/COCO/annotations/instances_val5000_right_occluded.json"
    occluded_val_json="/home/dingmingxu/work/work/dataset/COCO/annotations/val_occluded.json"
    unoccluded_val_json = "/home/dingmingxu/work/work/dataset/COCO/annotations/val_unoccluded.json"
    analyse = Analyse(val_json,img_path)
    flag=7
    if flag == 0:#产生标签文件
        analyse.cityperson2coco_train(0.9,json_save_path="cityperson_train_0.9.json")
        # analyse.cityperson2coco_val()
    elif flag==1:
        analyse.ana_occ(val_save_json)
    elif flag==2:
        analyse.vis_label()
    elif flag==3:
        analyse.cal_occluded_num()
    elif flag==4:
        analyse.gene_all_ratio_file()
    elif flag==5:#查看标注文件是否标注正确
        analyse.vis_label()
    elif flag==6:
        analyse.gen_occluded_img()
    elif flag==7:#查看结果
        # checkpoin_ori_file='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/work_dirs/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco_mixhead/epoch_4.pth'
        config_path='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco_mixhead.py'
        checkpoin_ori_file=load_from = ['/home/dingmingxu/work/work/code/mmdetection/pretrainmodels/fcos_r50_caffe_fpn_gn_2x_4gpu_20200218-8ceb5c76.pth',
                                        '/home/dingmingxu/work/work/code/mmdetection/pretrainmodels/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth']
        img_val_path='/home/dingmingxu/work/work/dataset/COCO/val'
        index = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13,
                 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27,
                 26: 28, 27: 31, 28: 32, 29: 33,
                 30: 34, 31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47,
                 43: 48, 44: 49, 45: 50, 46: 51,
                 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63, 59: 64,
                 60: 65, 61: 67, 62: 70, 63: 72,
                 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84, 75: 85, 76: 86,
                 77: 87, 78: 88, 79: 89, 80: 90}
        ca2name = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                   7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                   13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
                   19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
                   25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                   34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
                   39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
                   43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                   49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
                   55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
        # analyse.vis_result(checkpoin_ori_file,config_ori_path,img_val_path)#可视化训练结果
        analyse.vis_result(checkpoin_ori_file, config_path, img_val_path,index,ca2name)  # 可视化训练结果
    elif flag==8:
        analyse.tococo(val_json,val_save_json)
    elif flag==9:
        analyse.gen_coco_occluded(5,train_save_json)
    elif flag==10:
        analyse.val_devide(val_save_json,occluded_val_json,unoccluded_val_json)
    elif flag==11:
        analyse.ana_boxes()
    elif flag==12:
        analyse.delete_defect(save_path)
    elif flag==13:#模型融合
        json_file_list=[
            '/home/dingmingxu/work/work/code/mmdetection/occluded_detection/results/retinanet_r50_fpn_2x_coco/result.json',
            "/home/dingmingxu/work/work/code/mmdetection/occluded_detection/results/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco/result.json",
        ]
        result_file = "/home/dingmingxu/work/work/code/mmdetection/occluded_detection/results/fuse/result.json"
        analyse.mergecoco2coco(json_file_list,result_file)
    elif flag==14:#生成结果文件
        #配置文件
        config2make_json='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco.py'
        #训练好的模型
        model2make_json='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/work_dirs/fcos_r50_mixhead_improve_0.2/epoch_1.pth'
        #测试图片地址
        pic_path='/home/dingmingxu/work/work/dataset/COCO/val'
        #结果保存位置
        json_out_path='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/results/fcos_r50_mixhead_improve_0.2/result.json'
        #配置中类别与输出类别映射
        index={1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:13,
13:14,14:15,15:16,16:17,17:18,18:19,19:20,20:21,21:22,22:23,23:24,24:25,25:27,26:28,27:31,28:32,29:33,
30:34,31:35,32:36,33:37,34:38,35:39,36:40,37:41,38:42,39:43,40:44,41:46,42:47,43:48,44:49,45:50,46:51,
47:52,48:53,49:54,50:55,51:56,52:57,53:58,54:59,55:60,56:61,57:62,58:63,59:64,60:65,61:67,62:70,63:72,
64:73,65:74,66:75,67:76,68:77,69:78,70:79,71:80,72:81,73:82,74:84,75:85,76:86,77:87,78:88,79:89,80:90}
        ann_file='/home/dingmingxu/work/work/dataset/COCO/annotations/instances_val5000_right.json'
        analyse.make_result(config2make_json, model2make_json, pic_path, json_out_path,index,ann_file)
    elif flag==15:#模型评估
        ann_file='/home/dingmingxu/work/work/dataset/COCO/annotations/instances_val5000_right.json'
        result_file='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/results/fcos_r50_mixhead_improve_0.2/result.json'
        # result_file = '/home/dingmingxu/work/work/code/mmdetection/occluded_detection/results/fuse/result.json'
        analyse.coco_eval_by_file(ann_file,result_file)

    elif flag==16:#检查结果中annotation与images是否匹配
        ann_file='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/results/fcos_r50_mixhead_improve_0.2/result.json'
        coco_file='/home/dingmingxu/work/work/dataset/COCO/annotations/instances_val5000_right.json'
        val_path='/home/dingmingxu/work/work/dataset/COCO/val'
        ca2name={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
                 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
                 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog',
                 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
                 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
                 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
                 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
                 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
                 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich',
                 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
        analyse.check_result(ann_file,coco_file,val_path,ca2name)