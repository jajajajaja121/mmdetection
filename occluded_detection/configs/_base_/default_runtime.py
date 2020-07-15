checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = ['/home/dingmingxu/work/work/code/mmdetection/pretrainmodels/fcos_r50_caffe_fpn_gn_2x_4gpu_20200218-8ceb5c76.pth',
             '/home/dingmingxu/work/work/code/mmdetection/pretrainmodels/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth']
resume_from = None
workflow = [('train', 1)]
