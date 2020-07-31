import torch.nn as nn
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
import torch
from mmdet.core import (anchor_inside_flags,
                        force_fp32, images_to_levels, multi_apply,
                        multiclass_nms, unmap,distance2bbox)
from .anchor_head import AnchorHead
from ..builder import HEADS
from .fcos_head import FCOSHead
INF = 1e8
class FCOS_Anchor(FCOSHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 strides=(4, 8, 16, 32, 64),
                 conv_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(FCOS_Anchor, self).__init__(
            num_classes,
            in_channels,
            center_sampling=False,
            conv_cfg=conv_cfg,
            strides=strides,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_anchor(self,
                   bbox_preds,):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)

        flatten_bbox_preds_anchor = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_points_anchor = [points.repeat(1, 1) for points in all_level_points]
        anchors = []
        for level in range(len(flatten_bbox_preds_anchor)):
            anchor = distance2bbox(flatten_points_anchor[level], flatten_bbox_preds_anchor[level])
            anchors.append(anchor)
        return anchors
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_anchor_train(self,
             cls_scores,
             bbox_preds,
             centernesses):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        num_imgs = cls_scores[0].size(0)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        flatten_bbox_preds_anchor = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_points_anchor = [points.repeat(1, 1) for points in all_level_points]
        anchor_level = [torch.chunk(bbox_anchor, chunks=bbox_anchor.shape[0], dim=0) for bbox_anchor in
                        flatten_bbox_preds_anchor]
        anchor_img = []
        for level in range(len(anchor_level)):
            for img_num in range(len(anchor_level[0])):
                if len(anchor_img) - 1 < img_num:
                    anchor_img.append([])
                if len(anchor_img[img_num]) - 1 < level:
                    anchor_img[img_num].append([])
                anchor_img[img_num][level] = anchor_level[level][img_num].squeeze()
        anchors = []
        for img_num in range(len(anchor_img)):
            for level in range(len(anchor_img[0])):
                if (len(anchors) - 1) < img_num:
                    anchors.append([])
                if len(anchors[img_num]) - 1 < level:
                    anchors[img_num].append([])
                anchor = distance2bbox(flatten_points_anchor[level], anchor_img[img_num][level])
                anchors[img_num][level] = anchor
        return anchors

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             device,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in bbox_preds]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)

        num_imgs = bbox_preds[0].size(0)
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_points = [points.repeat(num_imgs, 1) for points in all_level_points]
        flatten_bbox_preds_anchor= [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs,-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_points_anchor = [points.repeat(1, 1) for points in all_level_points]
        anchor_img=[]
        anchor_level=[torch.chunk(bbox_anchor,chunks=bbox_anchor.shape[0],dim=0) for bbox_anchor in flatten_bbox_preds_anchor]
        for level in range(len(anchor_level)):
            for img_num in range(len(anchor_level[0])):
                if len(anchor_img)-1<img_num:
                    anchor_img.append([])
                if len(anchor_img[img_num])-1<level:
                    anchor_img[img_num].append([])
                anchor_img[img_num][level]=anchor_level[level][img_num].squeeze()
        anchors = []
        for img_num in range(len(anchor_img)):
            # img_shape=img_metas[img_num]['img_shape']
            # img_h, img_w = img_shape[:2]
            for level in range(len(anchor_img[0])):
                if (len(anchors)-1)<img_num:
                    anchors.append([])
                if len(anchors[img_num])-1<level:
                    anchors[img_num].append([])
                anchor = distance2bbox(flatten_points_anchor[level], anchor_img[img_num][level])
                # anchor[:,0].clamp(0,img_w)
                # anchor[:,1].clamp(0,img_h)
                # anchor[:,2].clamp(0,img_w)
                # anchor[:,3].clamp(0,img_h)
                anchors[img_num][level]=anchor
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_points=torch.cat(flatten_points)
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
        else:
            loss_bbox = pos_bbox_preds.sum()

        return loss_bbox,anchors

@HEADS.register_module()
class Retina_mixHead(AnchorHead):
    """An anchor-based head used in
    `RetinaNet <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     # ratios=[0.5, 1.0, 2.0],
                     ratios=[1.0, 1.0, 1.0],
                     strides=[8, 16, 32, 64, 128]),
                 train_cfg=None,
                 test_cfg=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(Retina_mixHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)
        self.FCOS_anchor=FCOS_Anchor(
            num_classes,
            in_channels,
            strides=anchor_generator['strides'],
            conv_cfg=conv_cfg,
            train_cfg=train_cfg,
            test_cfg=test_cfg
        )

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)
    def forward(self, feats):
        return multi_apply(self.forward_single, feats,self.FCOS_anchor.scales)
    def forward_single(self, x,scale):
        cls_feat_AB = x
        reg_feat_AB = x
        # cls_feat_AF = x
        reg_feat_AF = x
        for cls_conv in self.cls_convs:
            cls_feat_AB = cls_conv(cls_feat_AB)
        for reg_conv in self.reg_convs:
            reg_feat_AB = reg_conv(reg_feat_AB)
        cls_score_AB = self.retina_cls(cls_feat_AB)
        bbox_pred_AB = self.retina_reg(reg_feat_AB)

        # for cls_layer in self.FCOS_anchor.cls_convs:#anchor free的分类分支
        #     cls_feat_AF = cls_layer(cls_feat_AF)
        # cls_score_AF = self.FCOS_anchor.fcos_cls(cls_feat_AF)#得到anchor free的分类结果
        # centerness = self.FCOS_anchor.fcos_centerness(cls_feat_AF)#得到centerness的预测结果,这个参数越靠近GT的中心越接近1,
        #越远离GT的中心越接近0,用于在测试阶段抑制GT边缘的预测框
        for reg_layer in self.FCOS_anchor.reg_convs:
            reg_feat_AF = reg_layer(reg_feat_AF)
            # scale the bbox_pred of different level
            # float to avoid overflow when enabling FP16
        bbox_pred_AF = scale(self.FCOS_anchor.fcos_reg(reg_feat_AF)).float().exp()  # 得到anchor free分支的预测结果
        cls_score = cls_score_AB  # 将两种head的分类得分打包
        bbox_pred = [bbox_pred_AB, bbox_pred_AF]  # 将两种head的回归得分打包
        return cls_score, bbox_pred
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        bbox_preds_AB = [bbox_pred[0] for bbox_pred in bbox_preds]  # anchor base回归得分
        bbox_preds_AF = [bbox_pred[1] for bbox_pred in bbox_preds]  # anchor free回归得分
        # centerness = [cls_score[2] for cls_score in cls_scores]
        device = cls_scores[0][0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        loss_FCOS_bbox, anchor_list = self.FCOS_anchor.loss(
            bbox_preds_AF,
            gt_bboxes,
            gt_labels,
            img_metas,
            device)
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        assert len(featmap_sizes) == len(anchor_list[0])
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds_AB,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox,
        #             loss_cls_AF=loss_AF['loss_cls'],loss_bbox_AF=loss_AF['loss_bbox'],
        #             loss_centerness=loss_AF['loss_centerness'])
        # return dict(loss_cls_AF=loss_AF['loss_cls'],loss_bbox_AF=loss_AF['loss_bbox'],
        #                    loss_centerness=loss_AF['loss_centerness'])
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox,loss_FCOS_bbox=loss_FCOS_bbox)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes_fcos(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        """
        Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Size / scale info for each image
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        cls_scores_AF = [cls_score[1] for cls_score in cls_scores]  # anchor free分类得分
        bbox_preds_AF = [bbox_pred[1] for bbox_pred in bbox_preds]  # anchor free回归得分
        centerness = [cls_score[2] for cls_score in cls_scores]
        result_list=self.FCOS_anchor.get_bboxes(cls_scores_AF,
                                                bbox_preds_AF,
                                                centerness,
                                                img_metas,
                                                cfg,
                                                rescale
                                                )
        return result_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False):
        """
        Transform network output for a batch into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            img_metas (list[dict]): Size / scale info for each image
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the class index of the
                corresponding box.

        Example:
            >>> import mmcv
            >>> self = AnchorHead(
            >>>     num_classes=9,
            >>>     in_channels=1,
            >>>     anchor_generator=dict(
            >>>         type='AnchorGenerator',
            >>>         scales=[8],
            >>>         ratios=[0.5, 1.0, 2.0],
            >>>         strides=[4,]))
            >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
            >>> cfg = mmcv.Config(dict(
            >>>     score_thr=0.00,
            >>>     nms=dict(type='nms', iou_thr=1.0),
            >>>     max_per_img=10))
            >>> feat = torch.rand(1, 1, 3, 3)
            >>> cls_score, bbox_pred = self.forward_single(feat)
            >>> # note the input lists are over different levels, not images
            >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
            >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
            >>>                               img_metas, cfg)
            >>> det_bboxes, det_labels = result_list[0]
            >>> assert len(result_list) == 1
            >>> assert det_bboxes.shape[1] == 5
            >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
        """
        # cls_scores_AB = [cls_score[0] for cls_score in cls_scores]  # anchor base分类得分
        # cls_scores_AF = [cls_score[1] for cls_score in cls_scores]  # anchor free分类得分
        bbox_preds_AB = [bbox_pred[0] for bbox_pred in bbox_preds]  # anchor base回归得分
        bbox_preds_AF = [bbox_pred[1] for bbox_pred in bbox_preds]  # anchor free回归得分
        # centerness = [cls_score[2] for cls_score in cls_scores]
        device = cls_scores[0][0].device
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        anchor_list = self.FCOS_anchor.get_anchor(
            bbox_preds_AF)
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)
        assert len(cls_scores) == len(bbox_preds_AB)
        num_levels = len(cls_scores)
        mlvl_anchors = anchor_list
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_AB[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                mlvl_anchors, img_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list
    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    # remind that we set FG labels to [0, num_class-1]
                    # since mmdet v2.0
                    # BG cat_id: num_class
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = self.bbox_coder.decode(
                anchors, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the backend when using sigmoid
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in
            a single image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)

        if not inside_flags.any():
            # return (None, ) * 6
            inside_flags = valid_flags
            # print('all value are None')
        # assign gt and sample anchors
        inside_flags=valid_flags
        anchors = flat_anchors[inside_flags, :]
        # print(anchors.shape[0])
        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors,),
                                  self.background_label,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        print(pos_inds.shape[0])
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # only rpn gives gt_labels as None, this time FG is 1
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)