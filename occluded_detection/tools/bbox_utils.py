import numpy as np
#from chainer.backends import cuda
import torch
import json

def box_iou(box1, box2, order='xyxy'):

    box1=torch.from_numpy(box1).float()
    box2=torch.from_numpy(box2).float()
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    wh = (rb-lt+1).clamp(min=0)      # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]+1) * (box1[:,3]-box1[:,1]+1)  # [N,]
    area2 = (box2[:,2]-box2[:,0]+1) * (box2[:,3]-box2[:,1]+1)  # [M,]
    iou = inter / (area1[:,None] + area2 - inter)
    return iou.numpy()
def box_distance(box1, box2, order='xyxy'):
    box1=torch.from_numpy(box1).float()
    box2=torch.from_numpy(box2).float()
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(box1[:,None,:2], box2[:,:2])  # [N,M,2]
    rb = torch.min(box1[:,None,2:], box2[:,2:])  # [N,M,2]

    #wh = (rb-lt).clamp(min=0)      # [N,M,2]
    wh = (lt-rb).clamp(min=0)      # [N,M,2]
    distance= torch.sqrt(wh[:,:,0]* wh[:,:,0]+wh[:,:,1]*wh[:,:,1])
    
    return distance.numpy()
def update_box_score_base_distance(box_list):
    boxes=np.array(box_list)
    w=boxes[:,2]-boxes[:,0]
    h=boxes[:,3]-boxes[:,1]
    avg_length=(np.mean(w)+np.mean(h))/2
    distance=box_distance(boxes[:,:4],boxes[:,:4])
    update_num=0
    for k in range(distance.shape[0]):
        distance[k][k]=-1
        #min_distance=np.min(distance[k][distance[k]>=0])
        ov_num=len(distance[k][distance[k]<5])
        box_w=boxes[k,2]-boxes[k,0]
        box_h=boxes[k,3]-boxes[k,1]
        shape_ratio=box_w/box_h if box_w>=box_h else box_h/box_w
        box_area=box_w*box_h
        avg_area=avg_length*avg_length
        
 
        if ov_num>2:
            boxes[k,4]=boxes[k,4]*1.5 if boxes[k,4]*1.5<=1 else 1
            update_num+=1
            
    return boxes.tolist(),update_num
def box_iou_stat(box_file,step_count=10):
    file_dict={}
    result=[0 for i in range(step_count)]  
    #file1 to dict
    with open(box_file) as f:
        lines=f.readlines()
    for line in lines[1:]:
        line=line.replace('\n','')
        split=line.split(',')
        filename=split[0]
        box=split[1]
        if box!='':
            box=box.split(' ')
        box=[int(i) for i in box]
        if file_dict.get(filename) is not None:
            file_dict[filename].append(box)
        else:
            file_dict[filename]=[box]
    print(len(file_dict))
    for filename,boxes in file_dict.items():
        iou=box_iou(np.array(boxes), np.array(boxes))
        for k in range(iou.shape[0]):
            max_iou=np.max(iou[k][iou[k]!=1])
            max_iou=max_iou*100
            result[int(max_iou//step_count)]+=1
    print(result)
def box_iou_stat_coco(box_file,step_count=10):
    file_dict={}
    result=[0 for i in range(step_count)]  
    #file1 to dict
    with open(box_file) as f:
        ds=json.load(f)
    for ann in ds['annotations']:
        
        image_id=ann['image_id']
        box=ann['bbox']
        if box[2]*box[3]==0:
            print(box)
            continue
        box[2]=box[0]+box[2]-1
        box[3]=box[1]+box[3]-1
        if file_dict.get(image_id) is not None:
            file_dict[image_id].append(box)
        else:
            file_dict[image_id]=[box]
    print(len(file_dict),len(ds['annotations']))
    result_fi={}
    for id,boxes in file_dict.items():
        if len(boxes)<2:
            continue
        iou=box_iou(np.array(boxes), np.array(boxes))
        result=[0 for i in range(step_count)]
        for k in range(iou.shape[0]):
            max_iou=np.max(iou[k][iou[k]!=1])
            max_iou=max_iou*100 if max_iou >0 else 0
            #print(max_iou)
            result[int(max_iou//step_count)]+=1
        result_fi[id]=result.copy()
    return result_fi


'''
def bbox_iou_chainer(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    xp = cuda.get_array_module(bbox_a)

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)
'''
def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = box_iou(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        np.where(ws==0,0.001,ws)
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        #top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'SCORE_SUM':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.sum()
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

def box_voting_ad_iou(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    w = top_dets_out[:,2]-top_dets_out[:,0]
    w = np.array([int(i) for i in w])

    h = top_dets_out[:,3]-top_dets_out[:,1]
    h = np.array([int(i) for i in h])
    min_side = np.minimum(w,h)
    top_to_all_overlaps = box_iou(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        # if min_side[k]<40:
        #     thresh=1.0
        # elif min_side[k]>=40 and min_side[k]<120:
        #     thresh = min_side[k]/500+0.76
        # elif min_side[k]>=120 and min_side[k]<420:
        #     thresh = min_side[k]/1500+0.52
        # else:
        #     thresh = 0.8
        thresh=1.0
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]#提取出与nms保留结果iou大于阈值的bbox
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        np.where(ws==0,0.001,ws)#找出参与投票的框中得分为0的框,并把他们的得分设置为0.001
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)#用选出来的框的均值来代替原本的nms输出,并且以他们的得分为权重
        #top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'SCORE_SUM':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.sum()
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1+1) * (y2 - y1+1)
    order = scores.argsort()[::-1]#将得分按照从大到小的顺序排列

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1+1)
        h = np.maximum(0.0, yy2 - yy1+1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)#计算得分最高的框与各个框的iou

        inds = np.where(ovr <= thresh)[0]#取出与最高得分框相交低于百分之五十的框
        order = order[inds + 1]

    return keep

def py_cpu_softnms(dets,iou_thr=0.3, method='linear',  sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)



if __name__ == '__main__':
    box_iou_stat_coco('data/jinnan2_round2_train_20190401/train_restriction.json',step_count=10)