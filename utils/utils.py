from collections import OrderedDict, deque
from typing import Dict
from torchvision.transforms.functional import crop
import csv
import os
import math
import torch
import time
import pickle
# from display import Visdom_E
import cv2
import numpy as np
import json
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
import torchvision
import datetime
from torchvision import transforms
import inspect


def readJson(p: str):
    with open(p, 'r') as f:
        config = json.load(f)
    return config


def writeJson(data, p: str):
    with open(p, 'w') as f:
        json.dump(data, f)


def plotBoxes(images: torch.Tensor, boxes: torch.Tensor, OW, OH, color):
    images = ((images - torch.min(images)) / (torch.max(images) - torch.min(images)) * 255).int()
    C, H, W = images.shape
    MAX_N, _ = boxes.shape
    img = images.permute(1, 2, 0).detach().cpu().numpy().copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for key in range(boxes.shape[0]):
        bbox = boxes[key]
        bbox = [int(i) for i in bbox]
        text = str(key)
        f = lambda x: (int(x[0] * OW), int(x[1] * OH))
        cv2.rectangle(img, f(bbox[:2]), f(bbox[2:]), color, 1)
        cv2.putText(img, text, f(bbox[:2]), font, 1, color, 1)
    img = torch.Tensor(img).permute(2, 0, 1)
    return img


def prep_images(images):
    """
    preprocess images
    Args:
        images: pytorch tensor
    """
    images = images.div(255.0)

    images = torch.sub(images, 0.5)
    images = torch.mul(images, 2.0)

    return images


def deBZ(img, mean, std):
    return img * std + mean


def calc_pairwise_distance(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [N,D]
        Y: [M,D]
    Returns:
        dist: [N,M] matrix of euclidean distances
    """
    rx = X.pow(2).sum(dim=1).reshape((-1, 1))
    ry = Y.pow(2).sum(dim=1).reshape((-1, 1))
    dist = rx - 2.0 * X.matmul(Y.t()) + ry.t()
    return torch.sqrt(dist)


def get_min_row_col_matching(grid):
    # cost = np.array([[0.64, 1.63, 3.95, 4.09, 4.48],
    #                  [3.95, 4.48, 0.73, 0.23, 2.26],
    #                  [0.42, 0.70, 4.24, 4.56, 4.35],
    #                  [3.73, 3.84, 1.07, 1.95, 0.85],
    #                  [4.37, 5.02, 1.60, 0.73, 3.48]])
    cost = grid

    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind


def get_max_row_col_matching(grid):
    cost = -grid

    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind


def calc_pairwise_distance_3d(X, Y):
    """
    computes pairwise distance between each element
    Args: 
        X: [B,N,D]
        Y: [B,M,D]
    Returns:
        dist: [B,N,M] matrix of euclidean distances
    """
    B = X.shape[0]

    rx = X.pow(2).sum(dim=2).reshape((B, -1, 1))
    ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1))

    dist = rx - 2.0 * X.matmul(Y.transpose(1, 2)) + ry.transpose(1, 2)

    return torch.sqrt(dist)


def sincos_encoding_2d(positions, d_emb):
    """
    Args:
        positions: [N,2]
    Returns:
        positions high-dimensional representation: [N,d_emb]
    """

    N = positions.shape[0]

    d = d_emb // 2

    idxs = [np.power(1000, 2 * (idx // 2) / d) for idx in range(d)]
    idxs = torch.FloatTensor(idxs).to(device=positions.device)

    idxs = idxs.repeat(N, 2)  # N, d_emb

    pos = torch.cat([positions[:, 0].reshape(-1, 1).repeat(1, d), positions[:, 1].reshape(-1, 1).repeat(1, d)], dim=1)

    embeddings = pos / idxs

    embeddings[:, 0::2] = torch.sin(embeddings[:, 0::2])  # dim 2i
    embeddings[:, 1::2] = torch.cos(embeddings[:, 1::2])  # dim 2i+1

    return embeddings


def print_log(file_path, *args):
    print(*args)
    if file_path is not None:
        with open(file_path, 'a') as f:
            print(*args, file=f)


def show_config(cfg):
    print_log(cfg.log_path, '=====================Config=====================')
    for k, v in cfg.__dict__.items():
        if k == 'generator' or k == 'fv_dict':
            continue
        print_log(cfg.log_path, k, ': ', v)
    caller_function_name = inspect.stack()[1].function
    print_log(cfg.log_path, "called_method", ': ', f"{caller_function_name}")
    print_log(cfg.log_path, '======================End=======================')


def get_xyangle_gt_from_fp(fp_dict, frame_id, pid_list, ratio=1):
    xyangle_gt_list = []
    for id in pid_list:
        fp_str = f"{frame_id}_{id}"
        (x, y, r) = fp_dict[fp_str]
        x = float(x)
        y = float(y)
        r = float(r)
        r -= 90
        if r > 180:
            r = r - 360
        r /= 57.3
        xyangle_gt_list.append([x * ratio, y * ratio, r])

    return xyangle_gt_list


def get_eu_distance_mtraix(features1, features2):
    """
    :param features1:
    :param features2:
    :return:
        [num2, num1]
        sigmoid -> [0, 1]
    """

    if isinstance(features1, list) or isinstance(features2, list):
        features1 = torch.stack(features1, dim=0)
        features2 = torch.stack(features2, dim=0)
    dist_matrix = torch.norm(features1.unsqueeze(0) - features2.unsqueeze(1), p=2, dim=2)
    dist_matrix = torch.sigmoid(dist_matrix)
    # (n2_index, n1_index) relation_pair, -> 0 similarity highest
    return 1 - dist_matrix


def get_cos_distance_matrix(features1, features2):
    if isinstance(features1, list) or isinstance(features2, list):
        features1 = torch.stack(features1, dim=0)
        features2 = torch.stack(features2, dim=0)

    features1 /= features1.norm(p=2, dim=1, keepdim=True)
    features2 /= features2.norm(p=2, dim=1, keepdim=True)
    similarity = 0.5 + (features1.unsqueeze(0) * features2.unsqueeze(1)).sum(dim=2) * 0.5  # similarity \in [0,1]
    return similarity


def clac_rad_distance(rad_angle1, rad_angle2):
    rad_distance = torch.abs(rad_angle1 - rad_angle2)
    if rad_distance > math.pi:
        rad_distance = 2 * math.pi - rad_distance
    return rad_distance


def clac_rad_distance_with_no_tensor(rad_angle1, rad_angle2):
    rad_distance = math.fabs(rad_angle1 - rad_angle2)
    if rad_distance > math.pi:
        rad_distance = 2 * math.pi - rad_distance
    return rad_distance


def crop_img_by_boxes(img, boxes):
    box = boxes[0]
    if len(box) > 4:
        boxes = [box[:4] for box in boxes]

    croped_imgs = []
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((256, 128), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),

        normalizer
    ])
    for index, box in enumerate(boxes):
        # area = (box[2] - box[0]) * (box[3] - box[1])
        # if area < 0.3:

        new_img = crop(img, int(box[1]), int(box[0]), int(box[3] - box[1]), int(box[2] - box[0]))

        new_img = new_img / 255.0

        new_img = transform(new_img)
        croped_imgs.append(new_img)

    return torch.stack(croped_imgs, dim=0)


def crop_img_by_boxes_without_norm(img, boxes):
    box = boxes[0]
    if len(box) > 4:
        boxes = [box[:4] for box in boxes]

    croped_imgs = []

    for index, box in enumerate(boxes):
        # area = (box[2] - box[0]) * (box[3] - box[1])
        # if area < 0.3:

        new_img = crop(img, int(box[1]), int(box[0]), int(box[3] - box[1]), int(box[2] - box[0]))

        croped_imgs.append(new_img)

    return croped_imgs


def get_box_area(box):
    data = box
    area = (data[2] - data[0]) * (data[3] - data[1])
    return area


def make_iou_matrix(pred_bbox, gt_bbox):
    matrix = torch.zeros((len(pred_bbox), len(gt_bbox)))
    for i in range(len(pred_bbox)):
        for j in range(len(gt_bbox)):
            matrix[i][j] = calc_iou(pred_bbox[i], gt_bbox[j])
    return matrix


def get_matchid_from_fv_id_start_from_1(bbox_list, fv_dict, fv_str):
    if len(bbox_list[0]) > 4:
        bbox_list = [box[:4] for box in bbox_list]

    bbox_gt = [elem[1] for elem in fv_dict[fv_str]]
    matrix = make_iou_matrix(bbox_list, bbox_gt)
    # create n * m relation
    # get_pair_relation
    row_indices, col_indices = get_max_row_col_matching(matrix)
    ids = [fv_dict[fv_str][col][0] for col in col_indices]
    return row_indices, ids


def get_top_pair_from_matrix(matrix, top_num=3):
    # 1. recovering matrix1 to original matrix, (without +90)
    # 2. getting (i, j score) pairs, 3. sort it
    # 4. select high pair

    # 1
    matrix1 = matrix
    # 2
    i_j_pair_list = []
    i_j_pair_selected_list = []
    reserved_i_set = set()
    reserved_j_set = set()
    for i in range(len(matrix1)):
        for j in range(len(matrix1[i])):
            i_j_pair_list.append((i, j, matrix1[i][j].item()))

    # 3
    i_j_pair_list.sort(key=lambda pair: pair[2], reverse=True)
    # 4
    selected_pair_num = min(len(matrix1), len(matrix1[0]), top_num)
    while len(i_j_pair_selected_list) < selected_pair_num and len(i_j_pair_list) >= (
            selected_pair_num - len(i_j_pair_selected_list)):
        tmp = i_j_pair_list.pop(0)
        index1, index2, score = tmp
        if index1 in reserved_i_set:
            continue
        if index2 in reserved_j_set:
            continue
        i_j_pair_selected_list.append([index1, index2, score])
        reserved_i_set.add(index1)
        reserved_j_set.add(index2)
    return i_j_pair_selected_list


def rotate_x_y_by_point(x_y_tensor, angle_tensor, delta_x=0, delta_y=0, delta_theta=0):
    """
     rotate
     angele is rad
     (x,y) is the point to be rotated
     (pointx, pointy) is the rotated center #anti-clockwise
     (pointx, pointy) = (0, 0)
    """
    x_y_tensor_new = torch.tensor(x_y_tensor)
    angle_tensor_new = torch.tensor(angle_tensor)

    pointx = delta_x
    pointy = delta_y
    angle = delta_theta

    # n = x_y_tensor.shape[0]
    n = len(x_y_tensor_new)
    for i in range(n):
        x = x_y_tensor_new[i][0]
        y = x_y_tensor_new[i][1]
        nrx = (x - 0) * torch.cos(angle) - (y - 0) * torch.sin(angle) + pointx
        nry = (x - 0) * torch.sin(angle) + (y - 0) * torch.cos(angle) + pointy

        nrd = angle_tensor_new[i] - delta_theta
        if nrd < -math.pi:
            nrd = 2 * math.pi + nrd
        if nrd > math.pi:
            nrd = nrd - math.pi * 2

        x_y_tensor_new[i][0] = nrx
        x_y_tensor_new[i][1] = nry
        angle_tensor_new[i] = nrd

    return x_y_tensor_new, angle_tensor_new


def rotate_x_y_by_point_repair(x_y_tensor, angle_tensor, delta_x=0, delta_y=0, delta_theta=0):
    """
     rotate
     angele is rad
     (x,y) is the point to be rotated
     (pointx, pointy) is the rotated center #anti-clockwise
     (pointx, pointy) = (0, 0)
    """
    x_y_tensor_new = torch.tensor(x_y_tensor)
    angle_tensor_new = torch.tensor(angle_tensor)

    pointx = delta_x
    pointy = delta_y
    angle = delta_theta

    # n = x_y_tensor.shape[0]
    n = len(x_y_tensor_new)
    for i in range(n):
        x = x_y_tensor_new[i][0]
        y = x_y_tensor_new[i][1]
        nrx = (x - pointx) * torch.cos(angle) - (y - pointy) * torch.sin(angle) + pointx
        nry = (x - pointx) * torch.sin(angle) + (y - pointy) * torch.cos(angle) + pointy

        nrd = angle_tensor_new[i] - delta_theta
        if nrd < -math.pi:
            nrd = 2 * math.pi + nrd
        if nrd > math.pi:
            nrd = nrd - math.pi * 2

        x_y_tensor_new[i][0] = nrx
        x_y_tensor_new[i][1] = nry
        angle_tensor_new[i] = nrd

    return x_y_tensor_new, angle_tensor_new


def rotate_x_y_by_point_grad_fn(x_y_tensor, angle_tensor, delta_x=0, delta_y=0, delta_theta=0):
    """
     rotate
     angele is degree instead of rad
     (x,y) is the point to be rotated
     (pointx, pointy) is the rotated center #anti-clockwise
     (pointx, pointy) = (0, 0)
    """
    x_y_tensor_new = x_y_tensor
    angle_tensor_new = angle_tensor

    pointx = delta_x
    pointy = delta_y
    angle = delta_theta

    # n = x_y_tensor.shape[0]
    n = len(x_y_tensor_new)
    for i in range(n):
        x = x_y_tensor_new[i][0]
        y = x_y_tensor_new[i][1]
        nrx = (x - 0) * torch.cos(angle) - (y - 0) * torch.sin(angle) + pointx
        nry = (x - 0) * torch.sin(angle) + (y - 0) * torch.cos(angle) + pointy

        nrd = angle_tensor_new[i] - delta_theta
        if nrd < -math.pi:
            nrd = 2 * math.pi + nrd
        if nrd > math.pi:
            nrd = nrd - math.pi * 2

        x_y_tensor_new[i][0] = nrx
        x_y_tensor_new[i][1] = nry
        angle_tensor_new[i] = nrd

    return x_y_tensor_new, angle_tensor_new


def get_theta_delx_dely_from_pairs(angle1, angle2, xz_list1, xz_list2, pair_list):
    # angle1 = [-math.pi]
    # xz_list1 = [[-1.0, 1.0]]
    # angle2 = [-math.pi/2]
    # xz_list2 = [[1.0, 1.0]]
    # pair_list = [[0, 0, 0.9700127840042114]]

    theta_list = []
    delta_x_list = []
    delta_y_list = []
    for pair in pair_list:
        index_of_view2, index_of_view1, score = pair
        x2, z2 = xz_list2[index_of_view2]
        x1, z1 = xz_list1[index_of_view1]
        alpha1 = (-angle1[index_of_view1]) % (2 * math.pi) - 0.5 * math.pi
        alpha2 = (-angle2[index_of_view2]) % (2 * math.pi) - 0.5 * math.pi
        theta = alpha1 - alpha2
        delta_x = x1 - x2 * torch.cos(theta) + z2 * torch.sin(theta)
        delta_y = z1 - x2 * torch.sin(theta) - z2 * torch.cos(theta)
        theta_list.append(theta)
        delta_x_list.append(delta_x)
        delta_y_list.append(delta_y)

    # theta = torch.mean(torch.tensor(theta_list))
    # delta_x = torch.mean(torch.tensor(delta_x_list))
    # delta_y = torch.mean(torch.tensor(delta_y_list))

    theta_list = torch.tensor(theta_list)
    delta_x_list = torch.tensor(delta_x_list)
    delta_y_list = torch.tensor(delta_y_list)

    return theta_list, delta_x_list, delta_y_list


def get_theta_delx_dely_from_pairs_with_fn(angle1, angle2, xz_list1, xz_list2, pair_list):
    # angle1 = [-math.pi]
    # xz_list1 = [[-1.0, 1.0]]
    # angle2 = [-math.pi/2]
    # xz_list2 = [[1.0, 1.0]]
    # pair_list = [[0, 0, 0.9700127840042114]]

    theta_list = []
    delta_x_list = []
    delta_y_list = []
    for pair in pair_list:
        index_of_view2, index_of_view1, score = pair
        x2, z2 = xz_list2[index_of_view2]
        x1, z1 = xz_list1[index_of_view1]
        alpha1 = (-angle1[index_of_view1]) % (2 * math.pi) - 0.5 * math.pi
        alpha2 = (-angle2[index_of_view2]) % (2 * math.pi) - 0.5 * math.pi
        theta = alpha1 - alpha2
        delta_x = x1 - x2 * torch.cos(theta) + z2 * torch.sin(theta)
        delta_y = z1 - x2 * torch.sin(theta) - z2 * torch.cos(theta)
        theta_list.append(theta)
        delta_x_list.append(delta_x)
        delta_y_list.append(delta_y)

    return theta_list, delta_x_list, delta_y_list


def show_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'XY Loss: %.5f, R Loss: %.5f Using %.1f seconds' % (
        info['xy_loss'], info['r_loss'], info['time']))
    if "prob" in info.keys():
        print_log(log_path, f"xy_target{info['prob'][0]}, r_target{info['prob'][1]}")


def show_epoch_info_stn(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'Loss: %.10f, Sum Loss: %.10f Mul Loss: %.10f  Using %.1f seconds' % (
        info['loss'], info['sum_loss'], info['mul_loss'], info['time']))
    # if "prob" in info.keys():
    #     print_log(log_path, f"xy_target{info['prob'][0]}, r_target{info['prob'][1]}")


def show_monoreid_epoch_info(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'Loss: %.5f,Re-id Loss: %.4f  mono-xy %.4f mono-r: %.4f Using %.1f seconds at %s' %
              (info['loss'], info['re_id_loss'], info['mono_xy_loss'], info['mono_r_loss'], info['time'],
               datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def show_monoreid_epoch_test_info(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'Loss: %.5f,Re-id Loss: %.4f  mono-xy %.4f mono-r: %.4f Using %.1f seconds' %
              (info['loss'], info['re_id_loss'], info['mono_xy_loss'], info['mono_r_loss'], info['time']))
    print_log(log_path, 'Target Prob: xy:%s \nr:%s' % (info['prob'][0], info['prob'][1]))
    print_log(log_path, 'Matrix IOU %s' % (info['iou']))


def show_monoreid_epoch_test_info_contain_gt(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====> %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'Loss: %.5f,Re-id Loss: %.4f  Using %.1f seconds at %s' %
              (info['loss'], info['re_id_loss'], info['time'], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print_log(log_path, 'Cam.Pos.Avg %.4f Cam.Ori.Avg: %.4f' % (info['mono_xy_loss'], info['mono_r_loss'] * 57.3))
    if 'cam_prob' in info.keys():
        print_log(log_path, 'Cam.Pos.@: %s  Cam.Ori.@: %s' % (info['cam_prob'][0], info['cam_prob'][1]))
    if 'person_xy_mean_loss' in info.keys():
        print_log(log_path, 'Sub.Pos.Avg: %s   Sub.Ori.Avg: %s' % (
            info['person_xy_mean_loss'], info['person_r_mean_loss'] * 57.3))
    if 'prob' in info.keys():
        print_log(log_path, 'Sub.Pos.@: %s  Sub.Ori.@: %s' % (info['prob'][0], info['prob'][1]))
        # print_log(log_path, 'Person Prob GT: xy:%s  r:%s'%(info['prob_gt'][0], info['prob_gt'][1]))
    # if 'person_xy_mean_gt_loss' in info.keys():
    #     print_log(log_path, 'Person GT Mean Loss: XY:%s   R:%s'%(info['person_xy_mean_gt_loss'], info['person_r_mean_gt_loss']))
    if 'iou' in info.keys():
        print_log(log_path, 'Matrix F1 %s' % (info['iou']))
        # print_log(log_path, 'Pair Select IOU %s'%(info['pair_point_iou']))


def csv_write_dict(content_dict, write_path):
    # table head
    assert write_path.endswith('.csv')
    new_content_dict = {}
    # pre processing
    for key in content_dict.keys():
        val = content_dict[key]
        if type(val) == float:
            new_content_dict[key] = round(val, 4)
        elif type(val) == tuple:
            xy_order_dict = val[0]
            for inner_key, inner_val in xy_order_dict.items():
                new_key = f"{key}{inner_key}"
                new_content_dict[new_key] = inner_val

            r_order_dict = val[1]
            for inner_key, inner_val in r_order_dict.items():
                new_key = f"{key}{inner_key}"
                new_content_dict[new_key] = inner_val
        else:
            new_content_dict[key] = val

    content_dict = new_content_dict

    table_head = list(content_dict.keys())
    with open(write_path, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=table_head)
        # if the table is empty, adding the table header
        if not os.path.getsize(write_path):
            writer.writeheader()  # writing table header
        writer.writerows([content_dict])


def show_monoreid_epoch_test_info_best_loss(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====>[best loss] %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'Loss: %.5f,Re-id Loss: %.4f  Using %.1f seconds at %s' %
              (info['loss'], info['re_id_loss'], info['time'], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print_log(log_path, 'Cam.Pos.Avg %.4f Cam.Ori.Avg: %.4f' % (info['mono_xy_loss'], info['mono_r_loss'] * 57.3))
    if 'cam_prob' in info.keys():
        print_log(log_path, 'Cam.Pos.@: %s  Cam.Ori.@: %s' % (info['cam_prob'][0], info['cam_prob'][1]))
    if 'person_xy_mean_loss' in info.keys():
        print_log(log_path, 'Sub.Pos.Avg: %s   Sub.Ori.Avg: %s' % (
            info['person_xy_mean_loss'], info['person_r_mean_loss'] * 57.3))
    if 'prob' in info.keys():
        print_log(log_path, 'Sub.Pos.@: %s  Sub.Ori.@: %s' % (info['prob'][0], info['prob'][1]))
        # print_log(log_path, 'Person Prob GT: xy:%s  r:%s'%(info['prob_gt'][0], info['prob_gt'][1]))
    # if 'person_xy_mean_gt_loss' in info.keys():
    #     print_log(log_path, 'Person GT Mean Loss: XY:%s   R:%s'%(info['person_xy_mean_gt_loss'], info['person_r_mean_gt_loss']))
    if 'iou' in info.keys():
        print_log(log_path, 'Matrix F1 %s' % (info['iou']))
    if 'precision' in info.keys():
        print_log(log_path, 'Matrix precision %s, recall %s' % (info['precision'], info['recall']))

    # writing csv
    csv_path = os.path.join(os.path.dirname(log_path), 'result.csv')

    # mapping the keys of info to make it more readable
    new_info = dict()
    new_info['epoch'] = info['epoch']
    new_info['time'] = info['time']
    new_info['total_loss'] = info['loss']
    new_info['re_id_loss'] = info['re_id_loss']
    new_info['Cam.Pos.Avg'] = info['mono_xy_loss']
    new_info['Cam.Ori.Avg'] = info['mono_r_loss'] * 57.3
    new_info['Cam.Pos/Ori.@'] = info['cam_prob']
    new_info['Sub.Pos.Avg'] = info['person_xy_mean_loss']
    new_info['Sub.Ori.Avg'] = info['person_r_mean_loss'] * 57.3
    new_info['Sub.Pos/Ori.@'] = info['prob']
    new_info['F1'] = info['iou']
    new_info['Precision'] = info['precision']
    new_info['recall'] = info['recall']

    table_head = None
    content_lines = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                if 'epoch' in row:
                    table_head = row
                else:
                    content_lines.append(row)

        epoch_index = table_head.index('epoch')
        continue_tag = True
        for content in content_lines:
            if new_info['epoch'] == int(content[epoch_index]):
                continue_tag = False
                break
        if continue_tag:
            csv_write_dict(new_info, csv_path)
    else:
        csv_write_dict(new_info, csv_path)


def get_matchid_from_fv_id_start_from_fv_list(bbox_list, fv_elem, threshold=0.5):
    # fv_elem [[id1, [x1, y1, x2, y2]], [id2, [x1, y1, x2, y2]], []]

    if len(bbox_list[0]) > 4:
        bbox_list = [box[:4] for box in bbox_list]

    bbox_gt = [elem[1] for elem in fv_elem]
    matrix = make_iou_matrix(bbox_list, bbox_gt)
    # create n * m relation
    # get_pair_relation
    row_indices, col_indices = get_max_row_col_matching(matrix)
    final_index = []
    for i in range(len(row_indices)):
        if matrix[row_indices[i]][col_indices[i]] >= threshold:
            final_index.append(i)
    row_indices = [row_indices[i] for i in final_index]
    col_indices = [col_indices[i] for i in final_index]

    ids = [fv_elem[col][0] for col in col_indices]
    return row_indices, ids


def calc_cos_similarity_from_two_matrix(matrix1, matrix2):
    vector1 = matrix1.view(-1)
    vector2 = matrix2.view(-1)

    cos = torch.dot(vector1, vector2) / (torch.norm(vector1) * torch.norm(vector2))

    return cos


def show_aggregation_info_loss(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====>[best loss] %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'Loss: %.5f,Re-id Loss: %.4f  Using %.1f seconds at %s' %
              (info['loss'], info['re_id_loss'], info['time'], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print_log(log_path, 'Cam.Pos.Avg %.4f Cam.Ori.Avg: %.4f' % (info['mono_xy_loss'], info['mono_r_loss'] * 57.3))

    if 'cam_prob' in info.keys():
        print_log(log_path, 'Cam.Pos.@: %s  Cam.Ori.@: %s' % (info['cam_prob'][0], info['cam_prob'][1]))
    if 'total_mean_xy' in info.keys():
        print_log(log_path, 'Sub.Pos.Avg: %s   Sub.Ori.Avg: %s' % (info['total_mean_xy'], info['total_mean_r'] * 57.3))
    if 'total_prob' in info.keys():
        print_log(log_path, 'Sub.Pos.@: %s  Sub.Ori.@: %s' % (info['total_prob'][0], info['total_prob'][1]))
    if 'total_f1' in info.keys():
        print_log(log_path, 'Matrix F1: %s ' % (info['total_f1']))

    # writing csv
    csv_path = os.path.join(os.path.dirname(log_path), 'result.csv')

    # mapping the keys of info to make it more readable
    new_info = dict()
    new_info['epoch'] = info['epoch']
    new_info['time'] = info['time']
    new_info['total_loss'] = info['loss']
    new_info['re_id_loss'] = info['re_id_loss']
    new_info['Cam.Pos.Avg'] = info['mono_xy_loss']
    new_info['Cam.Ori.Avg'] = info['mono_r_loss'] * 57.3
    new_info['Cam.Pos/Ori.@'] = info['cam_prob']
    new_info['Sub.Pos.Avg'] = info['total_mean_xy']
    new_info['Sub.Ori.Avg'] = info['total_mean_r'] * 57.3
    new_info['Sub.Pos/Ori.@'] = info['total_prob']
    new_info['F1'] = info['total_f1']

    table_head = None
    content_lines = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                if 'epoch' in row:
                    table_head = row
                else:
                    content_lines.append(row)

        epoch_index = table_head.index('epoch')
        continue_tag = True
        for content in content_lines:
            if new_info['epoch'] == int(content[epoch_index]):
                continue_tag = False
                break
        if continue_tag:
            csv_write_dict(new_info, csv_path)
    else:
        csv_write_dict(new_info, csv_path)


def show_monoreid_epoch_test_info_best_f1_loss(phase, log_path, info):
    print_log(log_path, '')
    print_log(log_path, '====>[best f1 loss] %s at epoch #%d' % (phase, info['epoch']))
    print_log(log_path, 'Loss: %.5f,Re-id Loss: %.4f  Using %.1f seconds at %s' %
              (info['loss'], info['re_id_loss'], info['time'], datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    print_log(log_path, 'Cam.Pos.Avg %.4f Cam.Ori.Avg: %.4f' % (info['mono_xy_loss'], info['mono_r_loss'] * 57.3))
    if 'cam_prob' in info.keys():
        print_log(log_path, 'Cam.Pos.@: %s  Cam.Ori.@: %s' % (info['cam_prob'][0], info['cam_prob'][1]))
    if 'person_xy_mean_loss' in info.keys():
        print_log(log_path, 'Sub.Pos.Avg: %s   Sub.Ori.Avg: %s' % (
            info['person_xy_mean_loss'], info['person_r_mean_loss'] * 57.3))
    if 'prob' in info.keys():
        print_log(log_path, 'Sub.Pos.@: %s  Sub.Ori.@: %s' % (info['prob'][0], info['prob'][1]))
        # print_log(log_path, 'Person Prob GT: xy:%s  r:%s'%(info['prob_gt'][0], info['prob_gt'][1]))
    # if 'person_xy_mean_gt_loss' in info.keys():
    #     print_log(log_path, 'Person GT Mean Loss: XY:%s   R:%s'%(info['person_xy_mean_gt_loss'], info['person_r_mean_gt_loss']))
    if 'iou' in info.keys():
        print_log(log_path, 'Matrix F1 %s' % (info['iou']))
    if 'precision' in info.keys():
        print_log(log_path, 'Matrix precision %s, recall %s' % (info['precision'], info['recall']))

    # writing csv
    csv_path = os.path.join(os.path.dirname(log_path), 'result.csv')

    # mapping the keys of info to make it more readable
    new_info = dict()
    new_info['epoch'] = info['epoch']
    new_info['time'] = info['time']
    new_info['total_loss'] = info['loss']
    new_info['re_id_loss'] = info['re_id_loss']
    new_info['Cam.Pos.Avg'] = info['mono_xy_loss']
    new_info['Cam.Ori.Avg'] = info['mono_r_loss'] * 57.3
    new_info['Cam.Pos/Ori.@'] = info['cam_prob']
    new_info['Sub.Pos.Avg'] = info['person_xy_mean_loss']
    new_info['Sub.Ori.Avg'] = info['person_r_mean_loss'] * 57.3
    new_info['Sub.Pos/Ori.@'] = info['prob']
    new_info['F1'] = info['iou']
    new_info['Precision'] = info['precision']
    new_info['recall'] = info['recall']

    table_head = None
    content_lines = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f, skipinitialspace=True):
                if 'epoch' in row:
                    table_head = row
                else:
                    content_lines.append(row)

        epoch_index = table_head.index('epoch')
        continue_tag = True
        for content in content_lines:
            if new_info['epoch'] == int(content[epoch_index]):
                continue_tag = False
                break
        if continue_tag:
            csv_write_dict(new_info, csv_path)
    else:
        csv_write_dict(new_info, csv_path)


def log_final_exp_result(log_path, data_path, exp_result):
    no_display_cfg = ['num_workers', 'use_gpu', 'use_multi_gpu', 'device_list',
                      'batch_size_test', 'test_interval_epoch', 'train_random_seed',
                      'result_path', 'log_path', 'device']

    with open(log_path, 'a') as f:
        print('', file=f)
        print('', file=f)
        print('', file=f)
        print('=====================Config=====================', file=f)

        for k, v in exp_result['cfg'].__dict__.items():
            if k not in no_display_cfg:
                print(k, ': ', v, file=f)

        print('=====================Result======================', file=f)

        print('Best result:', file=f)
        print(exp_result['best_result'], file=f)

        print('Cost total %.4f hours.' % (exp_result['total_time']), file=f)

        print('======================End=======================', file=f)

    data_dict = pickle.load(open(data_path, 'rb'))
    data_dict[exp_result['cfg'].exp_name] = exp_result
    pickle.dump(data_dict, open(data_path, 'wb'))


def calc_f1_loss(y_true: torch.Tensor, y_pred: torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1


def calc_f1_loss_by_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


def calc_f1_loss_by_matrix_with_pres_recall(y_true, y_pred):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1, precision, recall


def calc_confusion_matrix_from_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    # epsilon = 1e-7
    #
    # precision = tp / (tp + fp + epsilon)
    # recall = tp / (tp + fn + epsilon)
    #
    # f1 = 2* (precision * recall) / (precision + recall + epsilon)
    return tp, tn, fp, fn


def get_f1_loss_from_confusion_matrix(tp, tn, fp, fn):
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1


class AverageMeter(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.xy_val = 0
        self.r_val = 0
        self.xy_avg = 0
        self.r_avg = 0
        self.xy_sum = 0
        self.r_sum = 0
        self.count = 0

    def update(self, xy_val, r_val):
        self.xy_val = xy_val
        self.r_val = r_val
        self.xy_sum += xy_val
        self.r_sum += r_val
        self.count += 1
        self.xy_avg = self.xy_sum / self.count
        self.r_avg = self.r_sum / self.count


def calc_iou(box1, box2, standard_coordinates=True):
    '''
    :param box1: [Xmin, Ymin, Xmax, Ymax] or [Xcenter, Ycenter, W, H]
    :param box2:
    :param standard_coordinates: True [Xmin, Ymin, Xmax, Ymax] Fasle [Xcenter, Ycenter, W, H]
    :return:
    '''

    if standard_coordinates is True:
        Xmin1, Ymin1, Xmax1, Ymax1 = box1
        Xmin2, Ymin2, Xmax2, Ymax2 = box2
    else:
        Xcenter1, Ycenter1, W1, H1 = box1
        Xcenter2, Ycenter2, W2, H2 = box2
        Xmin1, Ymin1 = int(Xcenter1 - W1 / 2), int(Ycenter1 - H1 / 2)
        Xmax1, Ymax1 = int(Xcenter1 + W1 / 2), int(Ycenter1 + H1 / 2)
        Xmin2, Ymin2 = int(Xcenter2 - W2 / 2), int(Ycenter2 - H2 / 2)
        Xmax2, Ymax2 = int(Xcenter2 + W2 / 2), int(Ycenter2 + H2 / 2)

    inter_Xmin = max(Xmin1, Xmin2)
    inter_Ymin = max(Ymin1, Ymin2)
    inter_Xmax = min(Xmax1, Xmax2)
    inter_Ymax = min(Ymax1, Ymax2)

    W = max(0, inter_Xmax - inter_Xmin)
    H = max(0, inter_Ymax - inter_Ymin)

    inter_area = W * H

    merge_area = (Xmax1 - Xmin1) * (Ymax1 - Ymin1) + (Xmax2 - Xmin2) * (Ymax2 - Ymin2)

    IOU = inter_area / (merge_area - inter_area + 1e-6)

    return IOU


# used by stn
class AverageMeter_STN(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.mul_loss_sum = 0
        self.sum_loss_sum = 0

        self.loss_sum = 0
        self.loss_avg = 0
        self.count = 0

    def update(self, mul_loss, sum_loss, bath_size):
        self.mul_loss_sum += mul_loss
        self.sum_loss_sum += sum_loss
        self.loss_sum += mul_loss + sum_loss
        self.count += 1 * bath_size

        self.mul_avg = self.mul_loss_sum / self.count
        self.sum_avg = self.sum_loss_sum / self.count
        self.loss_avg = self.loss_sum / self.count


class AverageMeter_MonoReid(object):
    """
    Computes the average value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.reid_loss_sum = 0
        self.mono_xy_loss_sum = 0
        self.mono_r_loss_sum = 0

        self.reid_loss_avg = 0
        self.mono_xy_loss_avg = 0
        self.mono_r_loss_avg = 0
        self.count = 0

    def update(self, reid_loss, mono_xy_loss, mono_r_loss):
        self.reid_loss_sum += reid_loss
        self.mono_xy_loss_sum += mono_xy_loss
        self.mono_r_loss_sum += mono_r_loss
        self.count += 1

        self.reid_loss_avg = self.reid_loss_sum / self.count
        self.mono_xy_loss_avg = self.mono_xy_loss_sum / self.count
        self.mono_r_loss_avg = self.mono_r_loss_sum / self.count

        self.total_loss = self.reid_loss_avg + self.mono_xy_loss_avg + self.mono_r_loss_avg


class HitProbabilityMeter_Monoreid(object):
    def __init__(self):
        self.x_range = 3
        # self.x_range = 12
        self.r_range = 3
        # self.r_range = 8
        self.reset()
        self.intersection_sum = 0
        self.union_sum = 0
        self.f1_list = list()
        self.precision_list = list()
        self.recall_list = list()
        self.xy_delta_list = []
        self.r_delta_list = []
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

    def reset(self):
        self.xyr_total_num = 0
        # distance from 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ..., 20
        self.xy_target_list = [0 for i in range(self.x_range)]
        # self.xy_target_list = [0.5, 1, 1.5]
        # r from 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56 ,..., 86
        # self.r_target_list = [0 for i in range(self.r_range)]
        self.r_target_list = [0 for i in range(self.r_range)]

    # compare from 0.1 to 5, step is 0.1
    def update(self, xyr_pred, xyr_gt, intersection_num, union_num):
        for i in range(len(xyr_gt)):
            distance = math.sqrt((xyr_pred[i][0] - xyr_gt[i][0]) ** 2 + (xyr_pred[i][1] - xyr_gt[i][1]) ** 2)
            angle = clac_rad_distance_with_no_tensor(xyr_pred[i][2], xyr_gt[i][2])
            self.xy_delta_list.append(distance)
            self.r_delta_list.append(angle)
            self.xyr_total_num += 1
            # calculating distance target
            for j in range(self.x_range):
                threshold = j * 0.5 + 0.5
                if distance < threshold:
                    self.xy_target_list[j] += 1
            for j in range(self.r_range):
                threshold = 5 + j * 5
                if angle < threshold / 57.3:
                    self.r_target_list[j] += 1

        self.xy_target_prob_list = list(map(lambda x: x / self.xyr_total_num, self.xy_target_list))
        self.r_target_prob_list = list(map(lambda x: x / self.xyr_total_num, self.r_target_list))

        self.intersection_sum += intersection_num
        self.union_sum += union_num

    def add_f1_score(self, f1_score):
        self.f1_list.append(f1_score)

    def add_precision_score(self, pre):
        self.precision_list.append(pre)

    def add_recall_score(self, recall):
        self.recall_list.append(recall)

    def update_confision_matrix(self, tp, tn, fp, fn):
        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def get_f1_from_confusion(self):
        return round(get_f1_loss_from_confusion_matrix(self.tp, self.tn, self.fp, self.fn), 4)

    def get_xy_r_prob_dict(self):
        xy_target_prob_dict = OrderedDict()
        r_target_prob_dict = OrderedDict()
        for j in range(self.x_range):
            xy_target_prob_dict[j * 0.5 + 0.5] = round(self.xy_target_prob_list[j], 4)
        for j in range(self.r_range):
            r_target_prob_dict[5 + j * 5] = round(self.r_target_prob_list[j], 4)

        return (xy_target_prob_dict, r_target_prob_dict)

    def set_xy_r_prob_dict(self, xy_target_prob_list, r_target_prob_list):
        self.xy_target_prob_list = xy_target_prob_list
        self.r_target_prob_list = r_target_prob_list

    def get_matrix_iou(self):
        return (self.intersection_sum / self.union_sum).item()

    def get_f1_score(self):
        f1_score_sum = 0
        size = len(self.f1_list)
        for f1_score in self.f1_list:
            f1_score_sum += f1_score

        return round(f1_score_sum / size, 4)

    def get_pre_score(self):
        pre_score_sum = 0
        size = len(self.precision_list)
        for f1_score in self.precision_list:
            pre_score_sum += f1_score

        return round(pre_score_sum / size, 4)

    def get_recall_score(self):
        f1_score_sum = 0
        size = len(self.recall_list)
        for f1_score in self.recall_list:
            f1_score_sum += f1_score

        return round(f1_score_sum / size, 4)

    def get_xy_mean_error(self):
        sum = 0
        size = len(self.xy_delta_list)
        for delta in self.xy_delta_list:
            sum += delta

        return round(sum / size, 4)

    def get_r_mean_error(self):
        sum = 0
        size = len(self.r_delta_list)
        for delta in self.r_delta_list:
            sum += delta

        return round(sum / size, 4)


class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0


class UnionFind(object):

    def __init__(self, n):
        self.uf = [-1 for i in range(n + 1)]
        self.sets_count = n

    def find(self, p):
        while self.uf[p] >= 0:
            p = self.uf[p]
        return p

    def union(self, p, q):
        proot = self.find(p)
        qroot = self.find(q)
        if proot == qroot:
            return
        elif self.uf[proot] > self.uf[qroot]:
            self.uf[qroot] += self.uf[proot]
            self.uf[proot] = qroot
        else:
            self.uf[proot] += self.uf[qroot]
            self.uf[qroot] = proot
        self.sets_count -= 1

    def is_connected(self, p, q):
        return self.find(p) == self.find(q)


class HitProbabilityMeter(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.xy_cal_num = self.cfg.xy_cal_num
        self.r_cal_num = self.cfg.r_cal_num
        self.xy_scale = self.cfg.xy_scale
        self.r_scale = self.cfg.r_scale

        # 1, 5, 10, 15
        self.x_y_hit_time_list = [0] * self.xy_cal_num
        # 5, 15, 25, 35, 45 degree
        self.r_hit_time_list = [0] * self.r_cal_num
        self.total_time = 0
        self.x_y_prob = [0] * self.xy_cal_num
        self.r_prob = [0] * self.r_cal_num

        self.x_pred_cache = []
        self.y_pred_cache = []
        self.r_pred_cache = []
        self.r_gt_cache = []

    def update(self, xyr_pred, xyr_gt, batch_size):
        self.total_time += batch_size

        x_pred, y_pred = xyr_pred[:, 0] * (self.cfg.x_max - self.cfg.x_min) + self.cfg.x_min, xyr_pred[:, 1] * (
                self.cfg.y_max - self.cfg.y_min) + self.cfg.y_min
        x_gt, y_gt = xyr_gt[:, 0] * (self.cfg.x_max - self.cfg.x_min) + self.cfg.x_min, xyr_gt[:, 1] * (
                self.cfg.y_max - self.cfg.y_min) + self.cfg.y_min

        # x_gt, y_gt = xyr_gt[:,0] * 80, xyr_gt[:,1] * 80
        r_pred = xyr_pred[:, 2] * 360
        r_gt = xyr_gt[:, 2] * 360

        x = x_pred.cpu().numpy().tolist()
        y = y_pred.cpu().numpy().tolist()
        r = r_pred.cpu().numpy().tolist()
        r_g = r_gt.cpu().numpy().tolist()
        self.x_pred_cache.extend(x)
        self.y_pred_cache.extend(y)
        self.r_pred_cache.extend(r)
        self.r_gt_cache.extend(r_g)

        # x y hit
        for i in range(self.xy_cal_num):
            radius = self.cfg.xy_test_start + i * self.xy_scale
            mask = torch.sqrt((x_gt - x_pred) ** 2 + (y_gt - y_pred) ** 2) <= radius
            self.x_y_hit_time_list[i] += mask.sum()
        # r hit
        for i in range(self.r_cal_num):
            degree_delta = self.cfg.r_test_start + i * self.r_scale
            r_loss = torch.abs(r_pred - r_gt)
            mask = r_loss > 180
            r_loss[mask] = 360 - r_loss[mask]

            mask = r_loss <= degree_delta
            self.r_hit_time_list[i] += mask.sum()

        # res = (torch.tensor(self.x_y_hit_time_list) / self.total_time, torch.tensor(self.r_hit_time_list) / self.total_time)

    def get_prob(self):
        res = (
            torch.tensor(self.x_y_hit_time_list) / self.total_time,
            torch.tensor(self.r_hit_time_list) / self.total_time)
        return res

    def get_xyr_cache(self):
        return (self.x_pred_cache, self.y_pred_cache, self.r_pred_cache, self.r_gt_cache)


def analyse_camera_distribution(path):
    x_list = list()
    y_list = list()
    r_list = list()

    with open(path, 'r') as f:
        for line in f:
            tmp = line.split()
            x_list.append(int(float(tmp[1])))
            y_list.append(int(float(tmp[2])))
            r_list.append(int(float(tmp[3])))

    print(f"x_range:[{min(x_list)},{max(x_list)}]")
    print(f"y_range:[{min(y_list)},{max(y_list)}]")
    print(f"r_range:[{min(r_list)},{max(r_list)}]")

    fig, ax = plt.subplots(2, 2)
    ax[0][0].bar([i for i in range(-50, 51)], [x_list.count(i) for i in range(-50, 51)], width=0.5)
    ax[0][1].bar([i for i in range(-10, 91)], [y_list.count(i) for i in range(-10, 91)], width=0.5)
    ax[1][0].bar([i for i in range(0, 361)], [r_list.count(i) for i in range(0, 361)], width=0.5)
    plt.show()  # 显示图像


class Timer(object):
    """
    class to do timekeeping
    """

    def __init__(self):
        self.last_time = time.time()

    def timeit(self):
        old_time = self.last_time
        self.last_time = time.time()
        return self.last_time - old_time


if __name__ == '__main__':
    pass
