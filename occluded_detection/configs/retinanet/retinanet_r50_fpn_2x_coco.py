_base_ = './retinanet_r50_fpn_1x_coco.py'
# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 12
load_from = '/home/dingmingxu/work/work/code/mmdetection/pretrainmodels/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth'
resume_from = None