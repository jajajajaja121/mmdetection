_base_ = './fcos_r50_caffe_fpn_gn-head_4x4_1x_coco_mixhead.py'

# learning policy
load_from = ['/home/dingmingxu/work/work/code/mmdetection/pretrainmodels/fcos_r50_caffe_fpn_gn_2x_4gpu_20200218-8ceb5c76.pth',
'/home/dingmingxu/work/work/code/mmdetection/pretrainmodels/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth']
# load_from='/home/dingmingxu/work/work/code/mmdetection/occluded_detection/work_dirs/fcos_r50_mixhead_fuse_result/epoch_1.pth'
lr_config = dict(step=[16, 22])
total_epochs = 12
