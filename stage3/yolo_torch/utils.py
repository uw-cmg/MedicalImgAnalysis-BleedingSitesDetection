from __future__ import division
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(rcll, prcn):
    """ Compute the average prcn, given the rcll and prcn curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        rcll:    The rcll curve (list).
        prcn: The prcn curve (list).
    # Returns
        The average prcn as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], rcll, [1.0]))
    mpre = np.concatenate(([0.0], prcn, [0.0]))

    # compute the prcn envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (rcll) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta rcll) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou_min(box1, box2, x1y1x2y2=True):
    # print(box1, box2)
    """
    Returns the small-iou of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    # iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    # print(b1_area, b2_area)
    small_area = min(b1_area, b2_area)

    return inter_area/(small_area+1e-16)


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

##### def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    
    ### bbox corner is a temp holder for conversion
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):

        conf_mask = (image_pred[:, 4] >= conf_thres)

        image_pred = image_pred[conf_mask]

        if not image_pred.size(0):
            continue
        _, conf_sort_idx = torch.sort(image_pred[:,4], descending=True)
        det_sorted = image_pred[conf_sort_idx]
        max_detections = []
        while det_sorted.size(0) != 0:
            # Get detection with highest confidence and save as max detection
            max_detections.append(det_sorted[0].unsqueeze(0))
            # Stop if we're at the last detection
            if len(det_sorted) == 1:
                break
            # Get the IOUs for all boxes with lower confidence
            ious = bbox_iou(max_detections[-1], det_sorted[1:])
            # Remove detections with IoU >= NMS threshold
            det_sorted = det_sorted[1:][ious < nms_thres]

        max_detections = torch.cat(max_detections).data
        # Add max detections to outputs
        output[image_i] = (max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections)))
    return output


##### def build_targets(pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
def build_targets(pred_boxes, pred_conf, target, anchors, num_anchors, grid_size, ignore_thres, img_dim):
    batch_size = target.size(0) #
    num_anchs = num_anchors
    ##### nC = num_classes
    mask = torch.zeros(batch_size, num_anchs, grid_size, grid_size)
    conf_mask = torch.ones(batch_size, num_anchs, grid_size, grid_size) ### anchor box intersection max, start with 1 then put zeros for below threshold iou's

    tx = torch.zeros(batch_size, num_anchs, grid_size, grid_size)
    ty = torch.zeros(batch_size, num_anchs, grid_size, grid_size)
    tw = torch.zeros(batch_size, num_anchs, grid_size, grid_size)
    th = torch.zeros(batch_size, num_anchs, grid_size, grid_size)
    tconf = torch.ByteTensor(batch_size, num_anchs, grid_size, grid_size).fill_(0)
    ##### tcls = torch.ByteTensor(batch_size, num_anchs, grid_size, grid_size, nC).fill_(0)

    n_real_pos = 0
    n_true_pos = 0
    for b in range(batch_size):
        ### basically iterating iver batches and the number of objects specified in  the target tensor
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue ### as we init the tensor with zeros, if not detections all the cells of the row ill be zeros and hene this will be true
            n_real_pos += 1 ### if we passed through we have a ground truth!

            ### since our grid values between 0 and 1 and hence we can sale it relativeluy into any dimension we want!
            ### below we scale it to grid_size dimension
            gx = target[b, t, 0] * grid_size
            gy = target[b, t, 1] * grid_size
            gw = target[b, t, 2] * grid_size
            gh = target[b, t, 3] * grid_size
            # print(gx); sys.exit()
            # Get shape of gt box
            ### unsqueeze - Returns a new tensor with a dimension of size one inserted at the specified position.
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)

            # Get shape of anchor box
            ### here we basically add x,y as 0,0 to the anchor boxes width and height. we have 3 0,0,w,h insyead od 3 w,h
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou_new(gt_box, anchor_shapes)

            # Get grid box indices
            gi = int(gx)
            gj = int(gy)

            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            ### #dbt why are we ignoring he above??
            
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)

            ##### # One-hot encoding of label
            ##### target_label = int(target[b, t, 0])
            ##### tcls[b, best_n, gj, gi, target_label] = 1

            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou_new(gt_box, pred_box, x1y1x2y2=False)

            ##### pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            ##### if iou > 0.5 and pred_label == target_label and score > 0.5:
            #####     n_true_pos += 1
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.2 and score > 0.7:
                n_true_pos += 1

    ##### return n_real_pos, n_true_pos, mask, conf_mask, tx, ty, tw, th, tconf, tcls
    return n_real_pos, n_true_pos, mask, conf_mask, tx, ty, tw, th, tconf


# def to_categorical(y, num_classes):
#     """ 1-hot encodes a tensor """
#     return torch.from_numpy(np.eye(num_classes, dtype="uint8")[y])



# def bbox_iou_numpy(box1, box2):
#     """Computes IoU between bounding boxes.
#     Parameters
#     ----------
#     box1 : ndarray
#         (N, 4) shaped array with bboxes
#     box2 : ndarray
#         (M, 4) shaped array with bboxes
#     Returns
#     -------
#     : ndarray
#         (N, M) shaped array with IoUs
#     """
#     area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

#     iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
#         np.expand_dims(box1[:, 0], 1), box2[:, 0]
#     )
#     ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
#         np.expand_dims(box1[:, 1], 1), box2[:, 1]
#     )

#     iw = np.maximum(iw, 0)
#     ih = np.maximum(ih, 0)

#     ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih

#     ua = np.maximum(ua, np.finfo(float).eps)

#     intersection = iw * ih

#     return intersection / ua
