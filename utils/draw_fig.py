import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
import matplotlib
import cv2
import random
import json

from torchvision.utils import make_grid
import torchvision
import numpy as np
from torch.nn import functional as F
import os
from PIL import Image
from functools import cmp_to_key
import sys

import seaborn as sns

matplotlib.use('agg')


def draw_line_fig(loss_list, path, is_train=True, is_timestample=True, extra=''):
    x = np.linspace(-10, 10, len(loss_list))
    if len(loss_list) > 0 and type(loss_list[0]) == torch.Tensor:
        loss_list = [elem.item() for elem in loss_list]

    y = np.asarray(loss_list)
    plt.figure()
    plt.plot(x, y)

    if not os.path.exists(f"{path}/figs"):
        os.makedirs(f"{path}/figs")

    if is_train:
        plt.savefig(
            f"{path}/figs/train_{extra}_loss{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) if is_timestample else ''}.png")
    else:
        plt.savefig(
            f"{path}/figs/test_{extra}_loss{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) if is_timestample else ''}.png")
    # plt.show()
    plt.cla()
    plt.close('all')


def draw_more_line_fig(loss_lists, path, cfg, is_xy_prob=True):
    # loss_lists [[a1, a2, a3],[b1, b2, b3],[c1, c2, c3]] -> [[a1 b1 c1], [a2 b2 c2] ...]
    data_list = [elem for elem in list(zip(*loss_lists))]

    for index, elem in enumerate(data_list):
        if is_xy_prob:
            radius = cfg.xy_test_start + index * cfg.xy_scale
        else:
            radius = cfg.r_test_start + index * cfg.r_scale

        plt.plot([i for i in range(len(elem))], elem, label=str(radius))

    plt.legend()  # 让图例生效
    plt.ylim((-0.3, 1.1))

    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    # plt.show()
    if not os.path.exists(f"{path}/figs"):
        print(f"{path}/figs")
        os.makedirs(f"{path}/figs")

    plt.savefig(f"{path}/figs/{'xy_prob' if is_xy_prob else 'r_prob'}.png")
    plt.cla()
    plt.close('all')


def draw_distribution_plot(data, save_path):
    # data like [1,2,3,4]
    x1 = np.asarray(data)
    sns.distplot(x1, bins=20, kde=True, rug=True, rug_kws={'color': 'y'})
    plt.title("reid_val")
    plt.savefig(save_path)


def draw_more_line_fig_key_from_dict(loss_dict, path, is_xy_prob=True):
    loss_lists = [list(inner_dict.values()) for inner_dict in loss_dict]
    # loss_lists [[a1, a2, a3],[b1, b2, b3],[c1, c2, c3]] -> [[a1 b1 c1], [a2 b2 c2] ...]
    data_list = [elem for elem in list(zip(*loss_lists))]
    keys = list(loss_dict[0].keys())

    for index, elem in enumerate(data_list):
        plt.plot([i for i in range(len(elem))], elem,
                 label=str(round(keys[index], 1)) + (" m" if is_xy_prob else " deg"))

    plt.legend()  # 让图例生效
    plt.ylim((-0.3, 1.1))

    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    # plt.show()
    if not os.path.exists(f"{path}/figs"):
        print(f"{path}/figs")
        os.makedirs(f"{path}/figs")

    plt.savefig(f"{path}/figs/{'xy_prob' if is_xy_prob else 'r_prob'}.png")
    plt.cla()
    plt.close('all')


def get_test_loss_from_log(path='./log/epoch60_res/log.txt'):
    test_loss_list = []
    next_line = False

    with open(path, 'r') as f:
        for line in f:
            if 'Test at epoch' in line:
                next_line = True
                continue
            if next_line:
                test_loss_list.append(float(line[6:13]))
                next_line = False

    return test_loss_list


def get_train_loss_from_log(path='./log/epoch60_res/log.txt'):
    train_loss_list = []
    next_line = False

    with open(path, 'r') as f:
        for line in f:
            if 'Train at epoch' in line:
                next_line = True
                continue
            if next_line:
                train_loss_list.append(float(line[6:13]))
                next_line = False

    return train_loss_list


def draw_scatter_distribution(x, y, path, epoch, s=5, r_axis=True):
    # data = np.asarray(data)

    plt.scatter(x, y, marker='o', s=s, cmap=plt.cm.Spectral)

    if r_axis:
        plt.axis([0, 360, 0, 360])
    # plt.show()
    if not os.path.exists(f"{path}/figs"):
        os.makedirs(f"{path}/figs")

    plt.savefig(f"{path}/figs/{'xy' if not r_axis else 'r'}_distribution_{epoch}.png")

    plt.cla()
    plt.close('all')


def save_cover_img(img1, img2, save_path):
    # word_img_path = '/home/clark/dataset2/virtual_dataset/data9999/hor1_video/out_4170.png'
    # bg_img_path = '/home/clark/dataset2/virtual_dataset/data9999/hor1_video/out_6672.png'

    page_img = img1
    bg_img = img2

    _, syn_binary_inv = cv2.threshold(page_img, 200, 1, cv2.THRESH_BINARY_INV)
    _, syn_binary = cv2.threshold(page_img, 200, 1, cv2.THRESH_BINARY)
    bg_weight = round(random.sample(np.arange(0.1, 0.5, 0.05).tolist(), 1)[0], 2)
    page_weight = 1 - bg_weight
    syn_page = (cv2.addWeighted(syn_binary_inv * bg_img, bg_weight, syn_binary_inv * page_img, page_weight,
                                0)) + syn_binary * bg_img

    if not os.path.exists(f"{save_path}/figs"):
        os.makedirs(f"{save_path}/figs")
    cv2.imwrite(f"{save_path}/figs/cover.png", syn_page)


def add_start_point(imgs):
    # imgs = torch.zeros((16,1,631,630))
    # 8: 3, 635, 5058
    # 16: 3, 1268, 5058

    b, c, h, w = imgs.shape
    # draw_direction
    starting_rec = torch.full((26, 26), 255)
    dir_rec = torch.zeros((26, 26))
    dir_rec[0:25, 13 - 7: 13 + 7] = 255.0
    start_point = torch.cat([dir_rec, starting_rec], dim=0)

    start_points = torch.stack([start_point] * b, dim=0)
    start_points = torch.unsqueeze(start_points, dim=1)

    # print(starting_rec.shape)
    imgs[..., h - 26 * 2:h, w // 2 - 13: w // 2 + 13] = start_points

    return imgs


def add_start_point_3d(imgs):
    # imgs = torch.zeros((16,3,631,630))
    # 8: 3, 635, 5058
    # 16: 3, 1268, 5058

    b, c, h, w = imgs.shape
    # draw_direction
    # 26, 26
    starting_rec = torch.full((3, 12, 12), 255)
    # dir_rec = torch.zeros((3, 26, 26))
    dir_rec = torch.zeros((3, 12, 12))
    dir_rec[0:3, 0:12, 6 - 3: 6 + 3] = 255.0
    start_point = torch.cat([dir_rec, starting_rec], dim=1)

    start_points = torch.stack([start_point] * b, dim=0)

    # print(starting_rec.shape)
    imgs[..., h - 12 * 2:h, w // 2 - 6: w // 2 + 6] = start_points

    return imgs


def cover_imgs_colorful(fv_pair, grid, fv_dict, generator, json_path, gt, padding_zero_num=4):
    # fv_pair [12, 4]

    img1s = []
    img2s = []
    for index, fv_pair_one_batch in enumerate(fv_pair):
        x, y, r = gt[index]
        x = x.item()
        y = y.item()
        r = r.item()

        # get img1
        frame_id = fv_pair_one_batch[0].item()
        view_id = fv_pair_one_batch[1].item()
        json_path_detail = json_path + f"/hor{view_id}_video/out_{(padding_zero_num - len(str(frame_id))) * '0' + str(frame_id)}.png.monoloco.json"
        generator.get_heatmap2(x_bias=[0], y_bias=[0], r_bias=[0],
                               json_path_single_list=[json_path_detail], fv_dict=fv_dict, frame_id=frame_id,
                               view_id=view_id, x_gt=x, y_gt=y, r_gt=r)
        generator.write_font("%.2f_%.2f_%.2f" % (x, y, r), (0, 0))
        # 3, 631, 630
        img1 = generator.board
        img1s.append(img1)

        # get img2
        frame_id = fv_pair_one_batch[2].item()
        view_id = fv_pair_one_batch[3].item()
        json_path_detail = json_path + f"/hor{view_id}_video/out_{(padding_zero_num - len(str(frame_id))) * '0' + str(frame_id)}.png.monoloco.json"
        generator.get_heatmap2(x_bias=[0], y_bias=[0], r_bias=[0],
                               json_path_single_list=[json_path_detail], fv_dict=fv_dict, frame_id=frame_id,
                               view_id=view_id, add_border=True)
        img2 = generator.board
        img2s.append(img2)

    img1s = torch.stack(img1s, dim=0)
    img2s = torch.stack(img2s, dim=0)

    output = F.grid_sample(img2s.float(), grid)
    img1s = add_start_point_3d(img1s)
    #
    #
    img2_camera = add_start_point_3d((torch.zeros_like(output)))
    img2_camera = F.grid_sample(img2_camera.float(), grid)
    #
    img2s = output + img2_camera
    # # img2 = cover_two_img(output, img2_camera)
    # # res_imgs = cover_two_img(img2, img1)
    res_imgs = img1s + img2s

    return res_imgs


def cover_imgs(img1, img2, grid):
    # if isinstance(model, torch.nn.DataParallel)	:
    # 	print('yes')
    # 	model = model.module

    # grid = model.grid
    output = F.grid_sample(img2, grid)
    img1 = add_start_point(img1)

    img2_camera = add_start_point((torch.zeros_like(output)))
    img2_camera = F.grid_sample(img2_camera, grid)

    img2 = output + img2_camera
    # img2 = cover_two_img(output, img2_camera)
    # res_imgs = cover_two_img(img2, img1)
    res_imgs = img1 + img2

    return res_imgs


def save_png(imgs, epoch, path='../', if_time=False, extra=''):
    imgs = make_grid(imgs, padding=4, pad_value=255.0)
    imgs = torch.tensor(imgs, dtype=torch.uint8)
    if not os.path.exists(f"{path}/figs"):
        os.makedirs(f"{path}/figs")

    torchvision.io.write_png(imgs.cpu(), f"{path}/figs/test_{epoch}_{extra}_{if_time * time.time()}.png")


def draw_bounding_box(original_img, boxes):
    boxes = torch.tensor(boxes)
    if boxes.shape[1] != 4:
        boxes = boxes[..., : 4]
    else:
        pass
    img = torchvision.utils.draw_bounding_boxes(original_img, boxes)
    return img


def draw_bounding_box_with_color_or_labels(original_img, boxes, labels=None, colors=None, width=4, font_size=50,
                                           font='Ubuntu-B.ttf'):
    boxes = torch.tensor(boxes)
    if boxes.shape[1] != 4:
        boxes = boxes[..., : 4]
    else:
        pass
    img = torchvision.utils.draw_bounding_boxes(original_img, boxes, labels=labels, colors=colors, width=width,
                                                font=font, font_size=font_size)
    return img


def combine_different_view_to_grid_picture(base_dir='virtual/original_img_backup',
                                           view_num=5,
                                           dst_dir='virtual/original_img_backup/combine'):
    view_path_list = [f"{base_dir}/hor{i + 1}_video/" for i in range(view_num)]
    view_path_list.append(f"{base_dir}/top_video/")
    view_diff_list = []
    for i in range(view_num + 1):
        files = os.listdir(view_path_list[i])
        files.sort()
        view_list_tmp = [view_path_list[i] + file for file in files]
        view_diff_list.append(view_list_tmp)

    path_pair = list(zip(*view_diff_list))

    for pair in path_pair:
        img_list = []
        for path in pair:
            img = torchvision.io.read_image(path)
            img_list.append(img)
        img_grid = torchvision.utils.make_grid(img_list, view_num)
        torchvision.io.write_png(img_grid, dst_dir + f"/{path.split('/')[-1]}")
        print(f"Finishing {path.split('/')[-1]}")


def combine_different_view_to_grid_picture_with_bbox(fv_sk_box,
                                                     base_dir='virtual/original_img_backup',
                                                     view_num=5,
                                                     dst_dir='virtual/original_img_backup/combine2',
                                                     ):
    view_path_list = [f"{base_dir}/hor{i + 1}_video/" for i in range(view_num)]
    view_path_list.append(f"{base_dir}/top_video/")
    view_diff_list = []
    for i in range(view_num + 1):
        files = os.listdir(view_path_list[i])
        files.sort()
        view_list_tmp = [view_path_list[i] + file for file in files]
        view_diff_list.append(view_list_tmp)

    path_pair = list(zip(*view_diff_list))

    for pair in path_pair:
        img_list = []
        for view_id, path in enumerate(pair):

            img = torchvision.io.read_image(path)

            if view_id < len(pair) - 1:
                view_id += 1
                frame_id = int(path.split('/')[-1].split('.')[0])
                fv_str = f"{frame_id}_{view_id}"
                annotation_box = fv_sk_box[fv_str][1]
                img = draw_bounding_box(img, annotation_box)

            img_list.append(img)
        img_grid = torchvision.utils.make_grid(img_list, view_num)
        torchvision.io.write_png(img_grid, dst_dir + f"/{path.split('/')[-1]}")
        print(f"Combing img finishing {path.split('/')[-1]}")


if __name__ == '__main__':
    pass
