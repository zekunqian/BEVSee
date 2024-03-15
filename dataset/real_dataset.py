import torch
from torch.utils import data
import math
import torchvision as tv
import torchvision.transforms as transforms
import os
import json
import numpy as np

from utils.utils import get_matchid_from_fv_id_start_from_1, get_matchid_from_fv_id_start_from_fv_list
from utils.utils import crop_img_by_boxes, crop_img_by_boxes_without_norm


class RealDataset(data.Dataset):

    def __init__(self, cfg, input_dir_list, label_list):
        super(RealDataset, self).__init__()
        self.cfg = cfg
        self.img_pair_path = []
        self.label_list = label_list
        for dir_pair in input_dir_list:
            v0_list = sorted(os.listdir(dir_pair[0]))
            v1_list = sorted(os.listdir(dir_pair[1]))

            v0_list = [dir_pair[0] + '/' + v0 for v0 in v0_list]
            v1_list = [dir_pair[1] + '/' + v1 for v1 in v1_list]

            self.img_pair_path.extend(list(zip(v0_list, v1_list)))

        if cfg.dataset_num != -1:
            self.img_pair_path = self.img_pair_path[:cfg.dataset_num]
            self.label_list = self.label_list[:cfg.dataset_num]

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.img_pair_path)

    def __getitem__(self, index):

        # reading images
        if self.cfg.channel == 1:
            img1 = tv.io.read_image(self.img_pair_path[index][0]).float()
            img2 = tv.io.read_image(self.img_pair_path[index][1]).float()
        else:
            img1 = tv.io.read_image(self.img_pair_path[index][0], tv.io.image.ImageReadMode.RGB).float()
            img2 = tv.io.read_image(self.img_pair_path[index][1], tv.io.image.ImageReadMode.RGB).float()

        frame_vid_pair = []

        path1 = self.img_pair_path[index][0]
        hor, img_path = path1.split('/')[-2], path1.split('/')[-1]
        hor1, img_path1 = hor, img_path
        view_id = int(hor[4])
        # if self.cfg.model_name == 'monoreid':
        frame_id = int(img_path.split('.')[0])

        frame_vid_pair.append(frame_id)
        frame_vid_pair.append(view_id)

        path2 = self.img_pair_path[index][1]
        hor, img_path = path2.split('/')[-2], path2.split('/')[-1]
        hor2, img_path2 = hor, img_path
        view_id = int(hor[4])
        frame_id = int(img_path.split('.')[0])
        frame_vid_pair.append(frame_id)
        frame_vid_pair.append(view_id)

        frame_vid_pair = torch.tensor(frame_vid_pair)

        label = torch.tensor([])

        json_path1 = os.path.join(self.cfg.json_path, hor1,
                                  self.cfg.monoloco_prefix + img_path1 + self.cfg.monoloco_suffix)
        json_path2 = os.path.join(self.cfg.json_path, hor2,
                                  self.cfg.monoloco_prefix + img_path2 + self.cfg.monoloco_suffix)

        with open(json_path1, 'r') as f:
            json_data1 = json.load(f)
        with open(json_path2, 'r') as f:
            json_data2 = json.load(f)

        # filter z-max change z_max to distance max
        z_list1 = [math.sqrt(xyz[0] ** 2 + xyz[2] ** 2) for xyz in json_data1['xyz_pred']]
        while max(z_list1) >= self.cfg.z_max:
            index_of_max = z_list1.index(max(z_list1))
            del json_data1['uv_kps'][index_of_max]
            del json_data1['boxes'][index_of_max]
            del json_data1['xyz_pred'][index_of_max]
            del json_data1['angles'][index_of_max]
            del z_list1[index_of_max]
        z_list2 = [math.sqrt(xyz[0] ** 2 + xyz[2] ** 2) for xyz in json_data2['xyz_pred']]
        while max(z_list2) >= self.cfg.z_max:
            index_of_max = z_list2.index(max(z_list2))
            del json_data2['uv_kps'][index_of_max]
            del json_data2['boxes'][index_of_max]
            del json_data2['xyz_pred'][index_of_max]
            del json_data2['angles'][index_of_max]
            del z_list2[index_of_max]

        sk1, box1 = json_data1['uv_kps'], json_data1['boxes']
        sk2, box2 = json_data2['uv_kps'], json_data2['boxes']

        img1_cropped = crop_img_by_boxes(img1, box1)
        img2_cropped = crop_img_by_boxes(img2, box2)

        sk1 = torch.tensor(sk1)
        sk2 = torch.tensor(sk2)

        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)

        return img1, img2, img1_cropped, img2_cropped, sk1, box1, sk2, box2, label, frame_vid_pair, torch.tensor(
            []), torch.tensor([]), json_data1, json_data2


class RealDataset_CvHMTB(data.Dataset):

    def __init__(self, cfg, input_dir_list, label_list):
        super(RealDataset_CvHMTB, self).__init__()
        self.cfg = cfg
        self.img_pair_path = []
        self.label_list = label_list
        for dir_pair in input_dir_list:
            v0_list = sorted(os.listdir(dir_pair[0]))
            v1_list = sorted(os.listdir(dir_pair[1]))

            v0_list = [dir_pair[0] + '/' + v0 for v0 in v0_list]
            v1_list = [dir_pair[1] + '/' + v1 for v1 in v1_list]

            self.img_pair_path.extend(list(zip(v0_list, v1_list)))
        hor_seq_set = set()
        top_seq_set = set()
        for dir_pair in input_dir_list:
            split_dir1 = dir_pair[0].split('/')
            split_dir2 = dir_pair[1].split('/')
            seq1_name = '_'.join([split_dir1[-2], split_dir1[-1]])
            seq2_name = '_'.join([split_dir2[-2], split_dir2[-1]])
            hor_seq_set.add(seq1_name)
            hor_seq_set.add(seq2_name)
            top_seq_set.add(split_dir1[-2])
            top_seq_set.add(split_dir2[-2])

        self.annotation_dir_path = cfg.annotation_txt_dir_path

        # seq_name(V2_G1_h1):
        #   { 'id' : bbox }
        # seq:
        #   frameid:
        #       [[id, x, y, w, h], [id, x, y, w, h]]
        self.seq_frame_id_dict = {}
        self.min_frame_id = int(self.img_pair_path[0][0].split('/')[-1].split('.')[0].split('_')[-1])

        for seq in hor_seq_set:
            annotation_file_path = os.path.join(self.annotation_dir_path, f'{seq}.txt')

            inner_dict = {}
            with open(annotation_file_path, 'r') as f:
                for line in f:
                    line_split = line.split(',')
                    frame_id = int(line_split[0])
                    person_id = int(line_split[1])
                    x = int(float(line_split[2]))
                    y = int(float(line_split[3]))
                    w = int(float(line_split[4]))
                    h = int(float(line_split[5]))
                    if frame_id not in inner_dict.keys():
                        inner_dict[frame_id] = [[person_id, [x, y, x + w, y + h]]]
                    else:
                        inner_dict[frame_id].append([person_id, [x, y, x + w, y + h]])
            self.seq_frame_id_dict[seq] = inner_dict

        for seq_pre in top_seq_set:
            annotation_file_path = os.path.join(self.annotation_dir_path, f'{seq_pre}_t.txt')
            seq = f'{seq_pre}_t'
            inner_dict = {}
            with open(annotation_file_path, 'r') as f:
                for line in f:
                    line_split = line.split(',')
                    frame_id = int(line_split[0])
                    person_id = int(line_split[1])
                    x = int(line_split[2])
                    y = int(line_split[3])
                    w = int(line_split[4])
                    h = int(line_split[5])
                    if frame_id not in inner_dict.keys():
                        inner_dict[frame_id] = [[person_id, [x, y, x + w, y + h]]]
                    else:
                        inner_dict[frame_id].append([person_id, [x, y, x + w, y + h]])
            self.seq_frame_id_dict[seq] = inner_dict

        if cfg.dataset_num != -1:
            self.img_pair_path = self.img_pair_path[:cfg.dataset_num]
            self.label_list = self.label_list[:cfg.dataset_num]
            # self.counter_list = self.counter_list[:cfg.dataset_num]

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.img_pair_path)

    def __getitem__(self, index):

        try:
            # reading images
            if self.cfg.channel == 1:
                img1 = tv.io.read_image(self.img_pair_path[index][0]).float()
                img2 = tv.io.read_image(self.img_pair_path[index][1]).float()
            else:
                img1 = tv.io.read_image(self.img_pair_path[index][0], tv.io.image.ImageReadMode.RGB).float()
                img2 = tv.io.read_image(self.img_pair_path[index][1], tv.io.image.ImageReadMode.RGB).float()

            hor1_path_split = self.img_pair_path[index][0].split('/')
            seq_name_hor1 = f"{hor1_path_split[-3]}_{hor1_path_split[-2]}"

            hor2_path_split = self.img_pair_path[index][1].split('/')
            seq_name_hor2 = f"{hor2_path_split[-3]}_{hor2_path_split[-2]}"
            seq_top = f'{hor2_path_split[-3]}_t'

            top_view_dir_path = '/'.join(self.img_pair_path[index][0].split('/')[:-2]) + '/t'
            top_view_img_path = os.path.join(top_view_dir_path,
                                             f"{seq_top}_{self.img_pair_path[index][0].split('/')[-1].split('_')[-1]}")
            img_top = tv.io.read_image(top_view_img_path, tv.io.image.ImageReadMode.RGB).float()

            # getting json infomation

            frame_vid_pair = []

            path1 = self.img_pair_path[index][0]
            hor, img_path = path1.split('/')[-2], path1.split('/')[-1]
            hor1, img_path1 = hor, img_path
            hor = hor[1:]
            view_id = int(hor)
            # if self.cfg.model_name == 'monoreid':
            frame_id = int(img_path.split('.')[0].split('_')[-1])

            frame_vid_pair.append(frame_id)
            frame_vid_pair.append(view_id)

            # if frame_id == 1339:
            #     print('debug')

            path2 = self.img_pair_path[index][1]
            hor, img_path = path2.split('/')[-2], path2.split('/')[-1]
            hor2, img_path2 = hor, img_path
            hor = hor[1:]
            view_id = int(hor)
            frame_id = int(img_path.split('.')[0].split('_')[-1])
            frame_vid_pair.append(frame_id)
            frame_vid_pair.append(view_id)

            frame_vid_pair = torch.tensor(frame_vid_pair)

            frame_id = frame_id - self.min_frame_id + 1
            if frame_id not in self.seq_frame_id_dict[seq_name_hor1]:
                return torch.tensor(-1), torch.tensor(-2)
            hor1_id_box = self.seq_frame_id_dict[seq_name_hor1][frame_id]
            hor1_id_list = [elem[0] for elem in hor1_id_box]
            hor1_box_gt_list = [elem[1] for elem in hor1_id_box]
            hor2_id_box = self.seq_frame_id_dict[seq_name_hor2][frame_id]
            hor2_id_list = [elem[0] for elem in hor2_id_box]
            hor2_box_gt_list = [elem[1] for elem in hor2_id_box]
            top_id_box = self.seq_frame_id_dict[seq_top][frame_id]
            top_id_list = [elem[0] for elem in top_id_box]
            top_box_gt_list = [elem[1] for elem in top_id_box]

            # TODO Getting bbox from json
            # box1 format: [[],[],[]]
            label = torch.tensor([])
            # label[2] = label[2]/360
            # label[0], label[1] = (label[0] - self.cfg.x_min)/(self.cfg.x_max - self.cfg.x_min), (label[1] - self.cfg.y_min )/(self.cfg.y_max - self.cfg.y_min)

            json_path1 = os.path.join(self.cfg.json_path, hor1,
                                      self.cfg.monoloco_prefix + img_path1 + self.cfg.monoloco_suffix)
            json_path2 = os.path.join(self.cfg.json_path, hor2,
                                      self.cfg.monoloco_prefix + img_path2 + self.cfg.monoloco_suffix)

            with open(json_path1, 'r') as f:
                json_data1 = json.load(f)
            with open(json_path2, 'r') as f:
                json_data2 = json.load(f)

            # filter z-max change z_max to distance max
            z_list1 = [math.sqrt(xyz[0] ** 2 + xyz[2] ** 2) for xyz in json_data1['xyz_pred']]
            while max(z_list1) >= self.cfg.z_max:
                index_of_max = z_list1.index(max(z_list1))
                del json_data1['uv_kps'][index_of_max]
                del json_data1['boxes'][index_of_max]
                del json_data1['xyz_pred'][index_of_max]
                del json_data1['angles'][index_of_max]
                del z_list1[index_of_max]
            z_list2 = [math.sqrt(xyz[0] ** 2 + xyz[2] ** 2) for xyz in json_data2['xyz_pred']]
            while max(z_list2) >= self.cfg.z_max:
                index_of_max = z_list2.index(max(z_list2))
                del json_data2['uv_kps'][index_of_max]
                del json_data2['boxes'][index_of_max]
                del json_data2['xyz_pred'][index_of_max]
                del json_data2['angles'][index_of_max]
                del z_list2[index_of_max]

            sk1, box1 = json_data1['uv_kps'], json_data1['boxes']
            sk2, box2 = json_data2['uv_kps'], json_data2['boxes']

            indices = nms(np.asarray(box1), 0.8)
            sk1 = [sk1[index] for index in indices]
            box1 = [box1[index] for index in indices]

            # boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in box2]
            # scores = [b[4] for b in box2]
            # indices = preprocessing.non_max_suppression(np.asarray(boxes), 0.8, np.asarray(scores))
            indices = nms(np.asarray(box2), 0.8)
            sk2 = [sk2[index] for index in indices]
            box2 = [box2[index] for index in indices]

            row_indices1, col_ids_list1 = get_matchid_from_fv_id_start_from_fv_list(box1, hor1_id_box)
            row_indices2, col_ids_list2 = get_matchid_from_fv_id_start_from_fv_list(box2, hor2_id_box)

            view1_mask = [True for i in range(len(row_indices1))]
            view2_mask = [True for i in range(len(row_indices2))]
            for i in range(len(row_indices1)):
                if col_ids_list1[i] in top_id_list:
                    view1_mask[i] = True
                else:
                    view1_mask[i] = False
            for i in range(len(row_indices2)):
                if col_ids_list2[i] in top_id_list:
                    view2_mask[i] = True
                else:
                    view2_mask[i] = False

            row_indices1 = [row_indices1[i] for i in range(len(row_indices1)) if view1_mask[i] is True]
            col_ids_list1 = [col_ids_list1[i] for i in range(len(col_ids_list1)) if view1_mask[i] is True]
            row_indices2 = [row_indices2[i] for i in range(len(row_indices2)) if view2_mask[i] is True]
            col_ids_list2 = [col_ids_list2[i] for i in range(len(col_ids_list2)) if view2_mask[i] is True]

            sk1 = [sk1[index] for index in row_indices1]
            box1 = [box1[index] for index in row_indices1]

            sk2 = [sk2[index] for index in row_indices2]
            box2 = [box2[index] for index in row_indices2]

            hor1_top_bbox = [top_box_gt_list[top_id_list.index(id)] for id in col_ids_list1]
            hor2_top_bbox = [top_box_gt_list[top_id_list.index(id)] for id in col_ids_list2]

            # hor detection id
            col_ids_list1 = torch.tensor(col_ids_list1)
            col_ids_list2 = torch.tensor(col_ids_list2)

            img1_cropped = crop_img_by_boxes(img1, box1)
            img2_cropped = crop_img_by_boxes(img2, box2)

            sk1 = torch.tensor(sk1)
            sk2 = torch.tensor(sk2)

            box1 = torch.tensor(box1)
            box2 = torch.tensor(box2)
        except:
            return torch.tensor(-1), torch.tensor(-1)

        return img1, img2, img1_cropped, img2_cropped, sk1, box1, sk2, box2, label, frame_vid_pair, \
            col_ids_list1, col_ids_list2, json_data1, json_data2, \
            torch.tensor(hor1_box_gt_list), torch.tensor(hor1_id_list), torch.tensor(hor2_box_gt_list), torch.tensor(
            hor2_id_list), \
            torch.tensor(hor1_top_bbox), torch.tensor(hor2_top_bbox), torch.tensor(top_box_gt_list), torch.tensor(
            top_id_list), img_top


def nms(boxes, iou_thres):
    """ 非极大值抑制 """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    keep = []

    # 按置信度进行排序
    index = np.argsort(scores)[::-1]

    while (index.size):
        # 置信度最高的框
        i = index[0]
        keep.append(index[0])

        if (index.size == 1):  # 如果只剩一个框，直接返回
            break

        # 计算交集左下角与右上角坐标
        inter_x1 = np.maximum(x1[i], x1[index[1:]])
        inter_y1 = np.maximum(y1[i], y1[index[1:]])
        inter_x2 = np.minimum(x2[i], x2[index[1:]])
        inter_y2 = np.minimum(y2[i], y2[index[1:]])
        # 计算交集的面积
        inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)
        # 计算当前框与其余框的iou
        iou = inter_area / (areas[index[1:]] + areas[i] - inter_area)
        ids = np.where(iou < iou_thres)[0]
        index = index[ids + 1]

    return keep
