import torchvision
from torchvision import transforms
import os
import torch
import colorsys
import random
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

import json
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

if __name__ != '__main__':
    from utils.utils import *
else:
    from utils import *

color_list = ['deepskyblue', 'deeppink', 'purple', 'navy', 'darkcyan', 'darkorange', 'darkred']


def get_dic_out_list(paths):
    dic_out_list = []
    for path in paths:
        with open(path, 'r') as f:
            dic_out = json.load(f)
            dic_out_list.append(dic_out)
    return dic_out_list


def get_xzangle(paths=[r'res\out_002282.png.monoloco.json'] * 2):
    dic_out_list = get_dic_out_list(paths)
    xz_angle_preds = []

    for i in range(len(paths)):
        tmp = dic_out_list[i]
        xyz_pred = tmp['xyz_pred']
        xz_pred = [[elem[0], elem[2]] for elem in xyz_pred]

        angles = [np.rad2deg(angle) for angle in tmp['angles']]

        xz_angle_pred = [[*xz_pred[index], angles[index]] for index in range(len(xz_pred))]
        xz_angle_preds.append(xz_angle_pred)
    return xz_angle_preds


def get_bbox(paths=[r'res\out_002282.png.monoloco.json'] * 2):
    dic_out_list = get_dic_out_list(paths)
    box_preds = []

    for i in range(len(paths)):
        tmp = dic_out_list[i]
        box_pred = tmp['boxes']
        box_pred = [[box[0], box[1], box[2], box[3]] for box in box_pred]
        box_preds.append(box_pred)

    return box_preds


def get_3d_figure():
    fig = plt.figure()
    ax = Axes3D(fig)
    return ax


def get_cos(point1, point2):
    """
    :param point1: the point 1
    :param point2: the point 2
    :return: get the positive cos of (point1, point2)
    """
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    tmp = (np.sqrt(x1 ** 2 + y1 ** 2) * np.sqrt(x2 ** 2 + y2 ** 2))
    if tmp == 0:
        return 0
    cos = (x1 * x2 + y1 * y2) / (np.sqrt(x1 ** 2 + y1 ** 2) * np.sqrt(x2 ** 2 + y2 ** 2))
    return cos


def cos2deg(cos):
    return np.rad2deg(np.arccos(cos))


def cos2rad(cos):
    return np.arccos(cos)


def get_angle(point1, point2):
    cos = get_cos(point1, point2)
    angle = cos2deg(cos)
    return angle


def get_distance(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def rotate_x_y_by_0_0(x, y, angele, pointx=0, pointy=0):
    """
     rotate
     angele is degree insted of rad
     (x,y) is the point to be rotated
     (pointx, pointy) is the rotated center #anti-clockwise
     (pointx, pointy) = (0, 0)
    """
    angle = np.deg2rad(angele)
    nrx = (x - pointx) * np.cos(angle) - (y - pointy) * np.sin(angle) + pointx
    nry = (x - pointx) * np.sin(angle) + (y - pointy) * np.cos(angle) + pointy
    return (nrx, nry)


def index2relative_point(point, board, ratio=1):
    """
        index of 2-d array to a Two dimensional coordinate system, the (0,0) is the center of the 2-d array
        the default interval is 1, but we can change it by the parameter ratio( the default of it is 0 )

    """

    if len(board.shape) > 2:
        c, height, width = board.shape
    else:
        height, width = board.shape
    # square means ok
    center_point = (height // 2, width // 2)

    delta_x = (point[0] - center_point[0]) * ratio
    delta_y = (point[1] - center_point[1]) * ratio

    return ((delta_x), -(delta_y))


def relative_point2index(point, board, ratio=0, mode='center'):
    if len(board.shape) > 2:
        c, height, width = board.shape
    else:
        height, width = board.shape

    # height, width = board.shape
    center_point = (height // 2, width // 2)

    x_part = point[0] if ratio == 0 else int(point[0] // ratio)
    y_part = point[1] if ratio == 0 else int(point[1] // ratio)

    delta_x = center_point[0] + x_part
    delta_y = center_point[1] - y_part
    return (delta_x, delta_y)


def get_gaussian_color(angle, angle_range=45):
    gauss_bar = get_gaussian_bar()
    ratio = angle / angle_range
    index = int(ratio * (len(gauss_bar) - 1))

    return gauss_bar[index]


def get_gaussian_bar(path='gaussian4040.pth.npy'):
    """
        the single line array with gaussian distribution
    """
    data = np.load(path)
    x, y, z = data
    guassian_bar = z[20]
    return guassian_bar[20:]


def get_gaussian_rectangle(path='gaussian4040.pth.npy'):
    # data = np.load('gaussian4040.pth.npy')
    data = np.load(path)
    return data


def draw_gaussian_rectangle(length, mean=0, standard_deviation=1):
    """
        size = (length*2+1, length*2+1)
    """
    x = np.linspace(-3, 3, 2 * length + 1);
    y = np.linspace(-3, 3, 2 * length + 1);
    x, y = np.meshgrid(x, y);
    z = np.exp(-((y - mean) ** 2 + (x - mean) ** 2) / (2 * (standard_deviation ** 2)))
    z = z / (np.sqrt(2 * np.pi) * standard_deviation);
    return (x, y, z);


def draw_guassian_sector(length, angle, delta_angle=45):
    '''
        anti-clockwise
        initial direction is <-
        overall: draw the right part of the sector and then draw the mirror to the left
        right:(deprecated)

        :param length: the length of the arrow
        :param angle: anti degree of the rotation : clockwise start position is ↓
        :param delta_angle: the angle range

        overall: search the rectangle area of the board to fill every point
        setp 1:  initial the center vector by using (0, length) ane rotate function
        step 2:  traverse the rectangle to judege
                    1. distance
                    2. angle
        setp 3:
        setp 4:
        setp 5:

    '''
    board = np.zeros((length * 2 + 1, length * 2 + 1))
    (rows, cols) = board.shape
    vector_start = (0, length)
    vector_center = rotate_x_y_by_0_0(*vector_start, angle)
    for i in range(rows):
        for j in range(cols):
            point_tmp = index2relative_point((i, j), board)
            if point_tmp == (0, 0):
                board[i][j] = get_gaussian_color(0)
                continue
            dis = get_distance(point_tmp, (0, 0))
            ang = get_angle(vector_center, point_tmp)
            # print(f"({i},{j}), {point_tmp}, distance:{dis}, angle: {ang}")
            epsilon = 0.1
            if dis <= length + epsilon and ang <= delta_angle + epsilon:
                board[i][j] = get_gaussian_color(get_angle(vector_center, point_tmp), delta_angle)
    return board


def get_1_1_grid(x_range, y_range):
    x = np.linspace(-1, 1, x_range);
    y = np.linspace(-1, 1, y_range);
    x, y = np.meshgrid(x, y);
    return (x, y);


def assemble_rectangle_and_board(length, angle, delta_angle=90):
    # assemble the center gaus with the direct gaus

    board = draw_guassian_sector(length, angle, delta_angle)
    data = draw_gaussian_rectangle(length)
    rectangle = data[2]

    assembly = np.zeros(board.shape)
    col, row = assembly.shape
    for i in range(col):
        for j in range(row):
            if rectangle[i][j] > board[i][j]:
                assembly[i][j] = rectangle[i][j]
            else:
                assembly[i][j] = board[i][j]
    X, Y = get_1_1_grid(2 * length + 1, 2 * length + 1)
    return (X, Y, assembly)


def get_person_num(xzangle_pred, index):
    return len(xzangle_pred[index])


def get_grid_and_whole_board(length, width=40, height=40):
    # length is the scale of every standard '1'
    # TODO weather BUG here?
    x = np.linspace(-width, width, (2 * length + 1) * width * 2)
    y = np.linspace(-height, height, (2 * length + 1) * height * 2)
    X, Y = np.meshgrid(x, y)
    board = np.zeros((len(x), len(y)))
    # board.fill(-0.001)
    board.fill(0)
    return X, Y, board


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def get_n_colors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def create_figure(X, Y, Z, x_lable='x', y_lable='y'):
    ax = get_3d_figure()
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def show_figure():
    plt.show()


def fill_board(board, sector_cache, start_point, length, start_angle, figure_stride, if_filter=False):
    # clockwise
    # start_point = (0, 0)
    angle = (270 - (start_angle + 90) + 90) % 360
    # angle = xzangle[2]
    angle = int(angle) if angle > 0 else int(360 + angle)
    start_index = relative_point2index(start_point, board, figure_stride)
    start_index = (int(start_index[1]), int(start_index[0]))

    # _, _, assembly = assemble_rectangle_and_board(length,angle)
    # print(assembly.shape == sector_cache[length][angle].shape)
    assembly = sector_cache[length][angle if angle != 360 else 0]

    # fix the border of the sector( avoid the rectangle shape of zeros)
    if if_filter:
        epsilon = 50
        mask = assembly > epsilon

        # show_board(assembly[mask])
        board[start_index[0] - length: start_index[0] + length + 1,
        start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask]
    else:
        roi = board[start_index[0] - length: start_index[0] + length + 1,
              start_index[1] - length:start_index[1] + length + 1]
        if torch.max(roi) > 100:
            # do not add the heatmap
            pass
        else:
            epsilon = 50
            mask = assembly > epsilon
            board[start_index[0] - length: start_index[0] + length + 1,
            start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask]

    return board


# in the new task: the code is not necessary now
def process_xz_angle(xz_angle_list, x_bias, z_bias, angle_bias, add_start_point=False):
    xz_angle_list = torch.as_tensor(xz_angle_list)
    # processor = torch.as_tensor([x_bias, z_bias, angle_bia])

    # print(xz_angle_list.shape)
    for i in range(len(xz_angle_list)):
        xz_angle_list[i][0], xz_angle_list[i][1] = rotate_x_y_by_0_0(xz_angle_list[i][0], xz_angle_list[i][1],
                                                                     - angle_bias)
        xz_angle_list[i][2] += angle_bias
        xz_angle_list[i][0] += x_bias
        xz_angle_list[i][1] += z_bias

    return xz_angle_list


def make_sector_cache(sector_cache, lrange, rrange, slices, index):
    for length in range(lrange, rrange):
        slice_size = 360 // slices
        for angle in range(index * slice_size, index * slice_size + slice_size):
            print(
                f"generating length:{length}, angle:{angle}, angle-size:{index * slice_size} - {index * slice_size + slice_size}")
            _, _, assembly = assemble_rectangle_and_board(length, angle)
            assembly = torch.as_tensor(assembly) * 255 * 2
            sector_cache[length][angle] = assembly


def show_board(board, tag=False):
    """
        tag: weather cut the bottom half of the picture
    """
    board = torch.as_tensor(board)
    height, width = board.shape
    if tag:
        board = board.int()[0:height // 2 + 1, :]
    # board = board.int()
    img = transforms.ToPILImage()(board)
    img.show()


# def get_match(board, xz_anle_filled, xz_angle_list_later_to_add):
#     """
#     :param board_new: the final board
#     :param xz_anle_filled: some point data has been put on the board
#     :param xz_angle_list_later_to_add: some point data need to be put on the board
#     """
#     res_list = []
#     min = 99999999
#     for i in range(len(xz_angle_list_later_to_add)):
#         for j in range(len(xz_anle_filled)):
#             x_bias = xz_anle_filled[j][0] - xz_angle_list_later_to_add[i][0]
#             z_bias = xz_anle_filled[j][1] - xz_angle_list_later_to_add[i][1]
#             for angle_degree in range(360):
#                 #rotation
#                 xzangle_list = process_xz_angle(xz_angle_list_later_to_add, x_bias, z_bias, angle_degree)
#                 board_new = deepcopy(board)
#
#                 try:
#                     for _, xzangle in enumerate(xzangle_list):
#                         start_point = (xzangle[0], xzangle[1])
#                         angle = (270 - (xzangle[2] + 90) + 90) % 360
#                         # angle = xzangle[2]
#                         angle = int(angle) if angle > 0 else int(360 + angle)
#                         start_index = relative_point2index(start_point, board_new, figure_stride)
#                         assembly = sector_cache[length][angle if angle != 360 else 0]
#                         board_new = fill_board(board_new, assembly, start_index, length)
#                 except IndexError:
#                     print(
#                         f"the x_bias is {x_bias}, the y_bias is {z_bias}, the angle is {angle_degree}, out of border")
#                     # show_board(board)
#                     continue
#                 sum = get_sum_res(board_new)
#                 if sum < min:
#                     min = sum
#                     print( f"epoch: {i},{j}/{len(xz_angle_list_later_to_add)},{len(xz_anle_filled)} the x_bias is {x_bias}, the y_bias is {z_bias}, the angle is {angle_degree}, the sum is {sum}")
#                     res_list.append([x_bias, z_bias, angle_degree, sum])
#                 else:
#                     pass
#             torch.save(res_list,'models/res_match.pth')


def get_sum_res(board):
    return torch.sum(torch.as_tensor(board)).item()


def make_iou_matrix(pred_bbox, gt_bbox):
    matrix = torch.zeros((len(pred_bbox), len(gt_bbox)))
    for i in range(len(pred_bbox)):
        for j in range(len(gt_bbox)):
            matrix[i][j] = calc_iou(pred_bbox[i], gt_bbox[j])
    return matrix


class Generator(object):
    def __init__(self, sector_length=5, angle_delta=0, cache_path=None):
        self.length = sector_length
        self.angle_delta = angle_delta  # clockwise
        if cache_path == None:
            self.sector_cache = torch.load('../models/sector_cache_torch_45_thread.pth')
        else:
            self.sector_cache = torch.load(cache_path)

        self.counters = list()
        self.counter = 0
        # self.rgb_table = [[255, 218, 185],[0, 245, 255],[46, 139, 87],[238, 238, 0],
        #                   [205, 92, 92],[255, 48, 48],[160, 32, 240],[139, 10, 80],
        #                   [0,0,255], [144,238,144], [255, 0, 0], [0, 255, 0]]

        # total number is 25
        # self.rgb_table = [[191, 36, 42], [255, 70, 31], [132, 90, 50], [23, 133, 170],
        #                   [22, 169, 81], [255, 242, 223], [0, 52, 115], [255, 181, 30],
        #                   [254, 71, 119], [254, 241, 67], [189, 221, 34], [163, 226, 197],
        #                   [62, 237, 232], [77, 34, 26], [186, 202, 199], [204, 164, 227],
        #                   [87, 0, 79], [205, 92, 92], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        #                   [0, 191, 255], [0, 100, 0], [255, 0, 255], [65, 85, 92], [119, 32, 55]]
        self.rgb_table = [[191, 36, 42], [255, 70, 31], [255, 181, 30], [23, 133, 170],
                          [22, 169, 81], [255, 242, 223], [0, 52, 115], [255, 0, 255],
                          [254, 71, 119], [0, 100, 0], [189, 221, 34], [163, 226, 197],
                          [62, 237, 232], [0, 191, 255], [186, 202, 199], [204, 164, 227],
                          [87, 0, 79], [205, 92, 92], [0, 0, 255], [255, 0, 0], [0, 255, 0],
                          [77, 34, 26], [254, 241, 67], [132, 90, 50], [65, 85, 92], [119, 32, 55]]

        # self.rgb_table = get_n_colors(20)
        # print(self.rgb_table)

    def board123(self):

        board = torch.unsqueeze(self.board, 0)
        board = torch.cat([board] * 3, dim=0)
        self.board = board

    def get_heatmap(self, x_bias=[0], y_bias=[0], r_bias=[0], json_path_single_list=['../data/json/test.json'],
                    crop=True):
        X, Y, board = get_grid_and_whole_board(self.length, 30, 30)
        board = torch.as_tensor(board)
        figure_stride = X[0][1] - X[0][0]

        # [path1, path2, ..., pathn] -> [[[xza1], [xza2], [xza3]](path1),[[xza1],[xza2],[xza3]](path2)]
        xzangles = get_xzangle(json_path_single_list)
        # print(xzangles)
        for index, xzangle_list in enumerate(xzangles):
            # clockwise rotation
            xzangle_list = process_xz_angle(xzangle_list, x_bias[index], y_bias[index], r_bias[index])
            for i, xzangle in enumerate(xzangle_list):
                board = self.fill_board(board, self.sector_cache, start_point=[xzangle_list[i][0], xzangle_list[i][1]], \
                                        length=self.length, start_angle=xzangle_list[i][2], figure_stride=figure_stride)

        height, width = board.shape
        if crop:
            self.board = board.int()[0:height // 2 + 1, width // 4: int(width * (3 / 4))]
        else:
            self.board = board.int()

    def get_heatmap_from_xzangle_id(self, xzangle_list, ids, if_cropped=True, adding_border=False):
        '''
        :param xzangle_list: shape of [n, 3(x, z, angle(rad))]
        :param id_list: [n]
        :return: None
        '''

        X, Y, board = get_grid_and_whole_board(self.length, 30, 30)
        board = torch.as_tensor(board)
        figure_stride = X[0][1] - X[0][0]
        self.board = board
        self.board123()

        # traverse every person
        for i, xzangle in enumerate(xzangle_list):
            # TODO change id here, just using i to test whether it work.
            # TODO xzangle -> [x, z , angle, +id]

            try:
                self.board = self.fill_board_by_id(self.board, self.sector_cache, start_point=[xzangle[0], xzangle[1]], \
                                                   length=self.length, start_angle=xzangle[2] * 57.3,
                                                   figure_stride=figure_stride, id=ids[i], add_border=adding_border)
            except:
                print('hello world')

        # draw camera
        # if x_gt != -1 and y_gt != -1 and r_gt != -1:
        #     self.board = self.fill_board_cam(self.board, self.sector_cache, start_point=[x_gt, y_gt], length=self.length,
        #                                      start_angle=r_gt, figure_stride=figure_stride)

        self.board = self.board.int()
        if if_cropped:
            c, height, width = self.board.shape

            # cropped
            self.board = self.board.int()[0:c, 0:height // 2 + 1, width // 4: int(width * (3 / 4))]

    def get_heatmap_from_xzangle_id_big_size(self, xzangle_list, ids, if_cropped=True, height=100, width=100):
        '''
        :param xzangle_list: shape of [n, 3(x, z, angle(rad))]
        :param id_list: [n]
        :return: None
        '''

        X, Y, board = get_grid_and_whole_board(self.length, height, width)
        board = torch.as_tensor(board)
        figure_stride = X[0][1] - X[0][0]
        self.board = board
        self.board123()

        # traverse every person
        for i, xzangle in enumerate(xzangle_list):
            # TODO change id here, just using i to test whether it work.
            # TODO xzangle -> [x, z , angle, +id]

            self.board = self.fill_board_by_id(self.board, self.sector_cache, start_point=[xzangle[0], xzangle[1]], \
                                               length=self.length, start_angle=xzangle[2] * 57.3,
                                               figure_stride=figure_stride, id=ids[i])

        # draw camera
        # if x_gt != -1 and y_gt != -1 and r_gt != -1:
        #     self.board = self.fill_board_cam(self.board, self.sector_cache, start_point=[x_gt, y_gt], length=self.length,
        #                                      start_angle=r_gt, figure_stride=figure_stride)

        self.board = self.board.int()
        if if_cropped:
            c, height, width = self.board.shape

            # cropped
            self.board = self.board.int()[0:c, 0:height // 2 + 1, width // 4: int(width * (3 / 4))]

    def get_heatmap2(self, x_bias=[0], y_bias=[0], r_bias=[0], json_path_single_list=['../data/json/test.json'],
                     crop=True, color_bias=0, color_ratio=1, if_energy=False, fv_dict=None, frame_id=0, view_id=0,
                     add_border=False,
                     x_gt=-1, y_gt=-1, r_gt=-1):
        X, Y, board = get_grid_and_whole_board(self.length, 30, 30)
        board = torch.as_tensor(board)
        figure_stride = X[0][1] - X[0][0]
        self.board = board
        self.board123()

        # [path1, path2, ..., pathn] -> [[[xza1], [xza2], [xza3]](path1),[[xza1],[xza2],[xza3]](path2)]
        xzangles = get_xzangle(json_path_single_list)
        bboxes = get_bbox(json_path_single_list)
        # print(xzangles)
        # traverse every picture
        for index, xzangle_list in enumerate(xzangles):
            # clockwise rotation
            xzangle_list = process_xz_angle(xzangle_list, x_bias[index], y_bias[index], r_bias[index])
            bbox_list = bboxes[index]
            # bbox_list:
            # boxes[[392.4, 168.0165, 456.72, 293.93350000000004, 0.814], [787.786, 152.16, 853.474, 285.26000000000005, 0.731], [473.198, 178.0675, 518.3419999999999, 255.7825, 0.709], [922.1410000000001, 151.38049999999998, 984.409, 314.3895, 0.626], [586.502, 170.9105, 625.4780000000001, 283.31950000000006, 0.602], [ 134.40200000000002, 170.735, 190.29799999999997, 316.705, 0.389], [347.103, 201.433, 379.467, 261.647, 0.317], [274.812, 204.9255, 338.26800000000003, 274.54449999999997, 0.232]]
            # bbox_gt the same as above
            bbox_gt = [elem[1] for elem in fv_dict[f"{frame_id}_{view_id}"]]
            matrix = make_iou_matrix(bbox_list, bbox_gt)
            # create n * m relation
            # get_pair_relation
            row_indices, col_indices = get_max_row_col_matching(matrix)
            ids = [fv_dict[f"{frame_id}_{view_id}"][col][0] - 1 for col in col_indices]

            # traverse every person
            for i, xzangle in enumerate(xzangle_list[row_indices]):
                self.board = self.fill_board_by_id(self.board, self.sector_cache, start_point=[xzangle[0], xzangle[1]], \
                                                   length=self.length, start_angle=xzangle[2],
                                                   figure_stride=figure_stride, id=ids[i], \
                                                   color_ratio=color_ratio, color_bias=color_bias, if_energy=if_energy,
                                                   add_border=add_border)

            if x_gt != -1 and y_gt != -1 and r_gt != -1:
                self.board = self.fill_board_cam(self.board, self.sector_cache, start_point=[x_gt, y_gt],
                                                 length=self.length,
                                                 start_angle=r_gt, figure_stride=figure_stride)

        c, height, width = self.board.shape
        if crop:
            self.board = self.board.int()[0:c, 0:height // 2 + 1, width // 4: int(width * (3 / 4))]
        else:
            self.board = self.board.int()

    def write_font(self, content, position, color=(255, 0, 0), fontsize=40):
        background = transforms.ToPILImage()(torch.tensor(self.board, dtype=torch.uint8))
        draw = ImageDraw.Draw(background)
        fontsize = fontsize
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), fontsize)
        draw.text(position, str(content), fill=color, font=font)
        self.board = transforms.PILToTensor()(background)

    def fill_board_by_id(self, board, sector_cache, start_point, length, start_angle, figure_stride, if_filter=True,
                         id=-1, if_energy=False, color_ratio=1, color_bias=0, add_border=False):

        def put_down():
            epsilon = 50
            mask = assembly > epsilon

            for i in range(3):
                # here may result problem from out of board
                if id < 0:
                    board[i, start_index[0] - length: start_index[0] + length + 1,
                    start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask] * self.rgb_table[id][
                        i] * color_ratio

                    if add_border:
                        width = 1
                        # |
                        board[i, start_index[0] - length: start_index[0] + length + 1,
                        start_index[1] - length:start_index[1] - length + width] = 255.0

                        # -
                        board[i, start_index[0] - length: start_index[0] - length + width,
                        start_index[1] - length:start_index[1] + length + 1] = 255.0

                        #  |
                        board[i, start_index[0] - length: start_index[0] + length + 1,
                        start_index[1] + length + 1 - width:start_index[1] + length + 1] = 255.0

                        # _
                        board[i, start_index[0] + length + 1 - width: start_index[0] + length + 1,
                        start_index[1] - length:start_index[1] + length + 1] = 255.0


                else:
                    if if_energy:
                        board[i, start_index[0] - length: start_index[0] + length + 1,
                        start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask] * \
                                                                                     self.rgb_table[id + color_bias][
                                                                                         i] * color_ratio





                    else:
                        board[i, start_index[0] - length: start_index[0] + length + 1,
                        start_index[1] - length:start_index[1] + length + 1][mask] = self.rgb_table[id + color_bias][
                                                                                         i] * color_ratio
                        if add_border:
                            width = 1
                            # |
                            board[i, start_index[0] - length: start_index[0] + length + 1,
                            start_index[1] - length:start_index[1] - length + width] = 255.0

                            # -
                            board[i, start_index[0] - length: start_index[0] - length + width,
                            start_index[1] - length:start_index[1] + length + 1] = 255.0

                            #  |
                            board[i, start_index[0] - length: start_index[0] + length + 1,
                            start_index[1] + length + 1 - width:start_index[1] + length + 1] = 255.0

                            # _
                            board[i, start_index[0] + length + 1 - width: start_index[0] + length + 1,
                            start_index[1] - length:start_index[1] + length + 1] = 255.0

        # clockwise
        # start_point = (0, 0)
        angle = (270 - (start_angle + 90) + 90) % 360
        # angle = xzangle[2]
        angle = int(angle) if angle > 0 else int(360 + angle)
        start_index = relative_point2index(start_point, board, figure_stride)
        start_index = (int(start_index[1]), int(start_index[0]))

        # _, _, assembly = assemble_rectangle_and_board(length,angle)
        # print(assembly.shape == sector_cache[length][angle].shape)
        assembly = sector_cache[length][angle if angle != 360 else 0]

        # fix the border of the sector( avoid the rectangle shape of zeros)
        if not if_filter:
            # epsilon = 50
            # mask = assembly > epsilon
            #
            # # show_board(assembly[mask])
            # board[start_index[0] - length: start_index[0] + length + 1,
            # start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask]
            put_down()
            self.counter += 1
        else:
            roi = board[0:3, start_index[0] - length: start_index[0] + length + 1,
                  start_index[1] - length:start_index[1] + length + 1]
            val_epsilon = 10
            num_epsilon = 30
            roi = (roi > val_epsilon).view(3, -1)
            if torch.max(torch.sum(roi, dim=1)) > num_epsilon:
                # do not add the heatmap
                pass
            else:
                # epsilon = 50
                # mask = assembly > epsilon
                # board[start_index[0] - length: start_index[0] + length + 1,
                # start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask]
                put_down()
                self.counter += 1

        return board

    def fill_board_cam(self, board, sector_cache, start_point, length, start_angle, figure_stride):

        x, y = start_point
        if x > 0:
            x = 10
        else:
            x = -10

        if y > 50:
            y = 20
        else:
            y = 3

        if x == 10 and y == 20:
            start_angle = 270 + 90 - 45
        elif x == 10 and y == 3:
            start_angle = 270 + 90 - 45 + 90
        elif x == -10 and y == 3:
            start_angle = 270 + 90 - 45 + 90 + 90
        else:
            start_angle = 270 + 90 - 45 + 90 + 180

        start_point = (x, y)

        def put_down():
            epsilon = 50
            mask = assembly > epsilon

            for i in range(3):
                # energy
                board[i, start_index[0] - length: start_index[0] + length + 1,
                start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask] * self.rgb_table[-1][i]

                # add border
                # |
                width = 2
                board[i, start_index[0] - length: start_index[0] + length + 1,
                start_index[1] - length:start_index[1] - length + width] = 255.0

                # -
                board[i, start_index[0] - length: start_index[0] - length + width,
                start_index[1] - length:start_index[1] + length + 1] = 255.0

                #  |
                board[i, start_index[0] - length: start_index[0] + length + 1,
                start_index[1] + length + 1 - width:start_index[1] + length + 1] = 255.0

                # _
                board[i, start_index[0] + length + 1 - width: start_index[0] + length + 1,
                start_index[1] - length:start_index[1] + length + 1] = 255.0

        # clockwise
        # start_point = (0, 0)
        angle = (270 - (start_angle + 90) + 90) % 360
        angle = (angle + 180) % 360
        # angle = xzangle[2]
        angle = int(angle) if angle > 0 else int(360 + angle)
        start_index = relative_point2index(start_point, board, figure_stride)
        start_index = (int(start_index[1]), int(start_index[0]))

        # _, _, assembly = assemble_rectangle_and_board(length,angle)
        # print(assembly.shape == sector_cache[length][angle].shape)
        assembly = sector_cache[length][angle if angle != 360 else 0]

        put_down()

        return board

    def fill_board(self, board, sector_cache, start_point, length, start_angle, figure_stride, if_filter=True):

        def put_down():
            epsilon = 50
            mask = assembly > epsilon

            # show_board(assembly[mask])
            board[start_index[0] - length: start_index[0] + length + 1,
            start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask]

        # clockwise
        # start_point = (0, 0)
        angle = (270 - (start_angle + 90) + 90) % 360
        # angle = xzangle[2]
        angle = int(angle) if angle > 0 else int(360 + angle)
        start_index = relative_point2index(start_point, board, figure_stride)
        start_index = (int(start_index[1]), int(start_index[0]))

        # _, _, assembly = assemble_rectangle_and_board(length,angle)
        # print(assembly.shape == sector_cache[length][angle].shape)
        assembly = sector_cache[length][angle if angle != 360 else 0]

        # fix the border of the sector( avoid the rectangle shape of zeros)
        if not if_filter:
            # epsilon = 50
            # mask = assembly > epsilon
            #
            # # show_board(assembly[mask])
            # board[start_index[0] - length: start_index[0] + length + 1,
            # start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask]
            put_down()
            self.counter += 1
        else:
            roi = board[start_index[0] - length: start_index[0] + length + 1,
                  start_index[1] - length:start_index[1] + length + 1]
            if torch.max(roi) > 200:
                # do not add the heatmap
                pass
            else:
                # epsilon = 50
                # mask = assembly > epsilon
                # board[start_index[0] - length: start_index[0] + length + 1,
                # start_index[1] - length:start_index[1] + length + 1][mask] = assembly[mask]
                put_down()
                self.counter += 1

        return board

    def save_img(self, path):
        self.board = self.board.float()
        if torch.max(self.board) > 1:
            self.board /= 255.0

        # if single_channel:
        #     img = transforms.ToPILImage()(self.board)
        # else:
        img = transforms.ToPILImage()(self.board.float())

        # important to save the original image
        # if single_channel:
        #     img = img.convert("L")
        #     pass

        img.save(path)

    def show(self):
        show_board(self.board)

    def save_counter(self, path):
        with open(path, 'w') as f:
            for elem in self.counters:
                f.write(f"{elem}\n")


def load_files(dir='img_work2'):
    files = filter(lambda file: file.endswith('.json'), os.listdir(dir))
    return list(files)


def read_img(path):
    img = torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)
    return img


def write_font_to_img(img, content, position, color=(255, 0, 0)):
    '''

    :param img: tensor img
    :param content: string content
    :param position: (x, y) positive position of the start_point
    :return: img
    '''
    background = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(background)
    draw.text(position, str(content), fill=color)
    img = transforms.PILToTensor()(background)
    return img


def draw_color_card(output_path='../figs/colors_grid.png'):
    generator = Generator(10)
    rgb_table = generator.rgb_table
    imgs = []
    for i in range(len(rgb_table)):
        id = i + 1
        rgb = rgb_table[i]
        r, g, b = rgb
        img = torch.zeros((3, 20, 20), dtype=torch.uint8)
        img[0, ...] = r
        img[1, ...] = g
        img[2, ...] = b

        # background = transform(img)
        background = torchvision.transforms.ToPILImage()(img)

        draw = ImageDraw.Draw(background)
        draw.text((0, 0), str(id), fill=(255, 0, 0))
        img = transforms.PILToTensor()(background)

        # imgs.append(torch.tensor(img, dtype=torch.uint8))
        imgs.append(img)

    output_img = torchvision.utils.make_grid(imgs, padding=10, pad_value=255)
    torchvision.io.write_png(output_img, output_path)


if __name__ == '__main__':
    generator = Generator(10)

    xy_list = [[-1.4092, 4.4262],
               [3.0835, 6.6117],
               [1.2053, 4.3532],
               [0.9025, 4.9521],
               [-0.5227, 4.3464],
               [1.5533, 4.7135]]
    for i in range(len(xy_list)):
        xy_list[i][0] *= 5
        xy_list[i][1] *= 5
        xy_list[i].append(0)
    xzangle_tensor = torch.tensor(xy_list)

    generator.get_heatmap_from_xzangle_id(xzangle_tensor, [3, 4, 7, 10, 11, 15])
    generator.save_img('../figs/mono_23_209.png')

    '''
        json file example:
         
        gt [False, False, False, False, False, False, False, False]
        boxes [[392.4, 168.0165, 456.72, 293.93350000000004, 0.814], [787.786, 152.16, 853.474, 285.26000000000005, 0.731], [473.198, 178.0675, 518.3419999999999, 255.7825, 0.709], [922.1410000000001, 151.38049999999998, 984.409, 314.3895, 0.626], [586.502, 170.9105, 625.4780000000001, 283.31950000000006, 0.602], [134.40200000000002, 170.735, 190.29799999999997, 316.705, 0.389], [347.103, 201.433, 379.467, 261.647, 0.317], [274.812, 204.9255, 338.26800000000003, 274.54449999999997, 0.232]]
        confs [0.5551682191349899, 0.4945583906336592, 0.46869670977966377, 0.432027940032474, 0.42863526505436134, 0.23917186364884482, 0.1990817264615386, 0.13476613280808147]
        dds_pred [10.01045036315918, 8.939640998840332, 15.731131553649902, 8.569985389709473, 10.588171005249023, 11.58285140991211, 14.184322357177734, 12.546560287475586]
        stds_ale [0.513714075088501, 0.46247467398643494, 0.8328798413276672, 0.4346209168434143, 0.5204722881317139, 0.6593607068061829, 0.7905048131942749, 0.7559619545936584]
        stds_epi [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        xyz_pred [[-2.375635862350464, 0.6379697322845459, 9.70352840423584], [2.613187789916992, 0.42017892003059387, 8.538846015930176], [-2.266498327255249, 0.7361559271812439, 15.54958438873291], [3.775320529937744, 0.5230872631072998, 7.67580509185791], [0.0759270042181015, 0.6476588249206543, 10.568072319030762], [-6.017462253570557, 0.8473520874977112, 9.860760688781738], [-4.430905818939209, 0.917340099811554, 13.443235397338867], [-4.728568077087402, 0.9146665930747986, 11.585344314575195]]
        uv_kps [[[423.81, 426.0, 421.48, 430.15, 418.32, 437.9, 413.07, 444.04, 407.95, 447.0, 402.05, 433.76, 419.28, 428.34, 424.05, 423.09, 434.52], [177.7, 175.42, 175.52, 177.02, 177.27, 191.64, 192.15, 210.07, 209.51, 228.2, 225.32, 227.02, 227.24, 252.53, 253.12, 282.06, 273.16], [0.96, 0.9, 0.84, 0.75, 0.62, 0.74, 0.81, 0.75, 0.83, 0.75, 0.75, 0.79, 0.81, 0.81, 0.75, 0.78, 0.7]], [[823.73, 826.77, 821.35, 830.5, 816.27, 834.21, 807.34, 842.79, 800.15, 832.64, 797.66, 830.95, 813.68, 832.34, 821.01, 832.36, 821.06], [164.11, 161.61, 160.87, 162.83, 160.82, 176.27, 177.03, 192.93, 200.69, 175.04, 221.73, 219.32, 219.1, 247.24, 239.64, 272.9, 268.32], [0.86, 0.83, 0.91, 0.53, 0.81, 0.75, 0.78, 0.72, 0.74, 0.62, 0.66, 0.74, 0.79, 0.58, 0.67, 0.21, 0.39]], [[496.68, 498.51, 495.04, 501.47, 492.96, 505.72, 490.29, 511.39, 484.27, 511.54, 479.83, 503.37, 493.93, 505.32, 493.27, 508.27, 489.69], [184.62, 182.73, 182.86, 183.46, 183.81, 190.94, 192.12, 200.32, 201.45, 207.93, 209.24, 211.68, 211.77, 228.61, 229.25, 244.29, 248.31], [0.79, 0.79, 0.85, 0.64, 0.66, 0.67, 0.64, 0.62, 0.69, 0.53, 0.51, 0.68, 0.66, 0.66, 0.7, 0.65, 0.71]], [[938.73, 938.92, 955.84, 941.16, 958.53, 937.85, 968.97, 933.65, 973.75, 933.71, 967.97, 945.16, 966.2, 941.04, 967.48, 934.06, 968.02], [165.4, 162.32, 160.74, 161.95, 161.51, 177.67, 179.85, 202.32, 204.88, 223.76, 222.46, 228.76, 229.06, 261.66, 262.89, 291.67, 300.19], [0.01, 0.01, 0.01, 0.66, 0.46, 0.88, 0.79, 0.76, 0.52, 0.7, 0.23, 0.78, 0.75, 0.74, 0.72, 0.74, 0.74]], [[612.23, 612.18, 611.05, 601.76, 607.57, 595.73, 609.8, 592.96, 610.93, 593.7, 613.91, 598.02, 608.91, 600.46, 614.84, 601.91, 618.14], [179.15, 177.13, 177.15, 178.13, 177.97, 189.37, 189.08, 206.27, 206.8, 218.73, 222.03, 218.33, 217.85, 244.74, 243.97, 273.94, 272.32], [0.36, 0.01, 0.38, 0.25, 0.53, 0.63, 0.63, 0.51, 0.67, 0.36, 0.69, 0.57, 0.64, 0.63, 0.78, 0.66, 0.87]], [[155.83, 157.26, 154.86, 162.99, 169.02, 167.91, 174.67, 143.98, 154.84, 160.15, 163.08, 166.31, 172.49, 155.89, 170.86, 157.84, 180.08], [196.02, 193.09, 194.23, 191.79, 188.54, 202.7, 200.44, 194.94, 189.96, 182.11, 181.83, 245.05, 243.77, 272.06, 274.62, 304.65, 294.93], [0.01, 0.01, 0.01, 0.01, 0.01, 0.64, 0.4, 0.56, 0.01, 0.01, 0.01, 0.68, 0.48, 0.78, 0.37, 0.7, 0.01]], [[368.19, 369.12, 366.68, 359.03, 361.82, 356.02, 356.99, 368.12, 366.22, 372.49, 373.71, 353.54, 354.71, 370.34, 372.65, 368.99, 373.47], [207.68, 205.77, 205.93, 205.47, 206.64, 213.21, 214.93, 218.62, 227.07, 214.88, 221.79, 236.12, 237.58, 238.11, 240.1, 255.59, 255.47], [0.45, 0.01, 0.53, 0.01, 0.62, 0.47, 0.7, 0.01, 0.45, 0.01, 0.01, 0.01, 0.39, 0.01, 0.01, 0.01, 0.01]], [[305.88, 306.64, 313.8, 310.69, 318.13, 310.88, 321.87, 307.39, 327.77, 297.17, 330.26, 309.15, 319.34, 284.14, 300.7, 284.29, 306.92], [211.25, 209.86, 209.16, 210.42, 210.35, 221.24, 220.11, 235.06, 233.27, 234.66, 241.93, 246.56, 245.11, 243.23, 243.74, 267.29, 264.56], [0.01, 0.01, 0.01, 0.35, 0.01, 0.55, 0.44, 0.01, 0.01, 0.01, 0.01, 0.47, 0.37, 0.24, 0.01, 0.01, 0.01]]]
        uv_centers [[425, 229], [820, 217], [496, 216], [954, 230], [606, 226], [162, 243], [364, 231], [307, 238]]
        uv_shoulders [[425, 192], [821, 177], [498, 192], [953, 179], [603, 189], [171, 202], [357, 214], [316, 221]]
        uv_heads [[424, 177], [824, 162], [497, 183], [947, 162], [609, 178], [160, 193], [365, 206], [311, 210]]
        angles [2.0668063163757324, 1.185426115989685, 1.7406575679779053, -1.8351571559906006, -0.8056965470314026, -3.0001931190490723, -0.1805298626422882, -2.4659781455993652]
        angles_egocentric [1.830427646636963, 1.476772665977478, 1.5657367706298828, -1.3650436401367188, -0.7854276895523071, 2.7431282997131348, -0.488336980342865, -2.833751916885376]
        aux []
    
    '''
