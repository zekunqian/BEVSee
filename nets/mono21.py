import os
import torch
import logging
import PIL
import openpifpaf
import openpifpaf.datasets as datasets
from openpifpaf import decoder, network, visualizer, show, logger
import json
import glob
import copy
import torch.nn as nn
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict, OrderedDict
import time
import torchvision

LOG = logging.getLogger(__name__)


class mono(nn.Module):
    def __init__(self, device=torch.device('cuda:0'),
                 model='', without_load_model=False, im_size=None):

        super(mono, self).__init__()

        args = Args()
        args.model = model

        self.focal = args.focal
        self.Sx = args.Sx
        self.Sy = args.Sy
        self.device = device

        args, dic_models = factory_from_args(args)

        # self.kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration
        if im_size is not None:
            args.im_size = im_size

        self.kk = [
            [args.im_size[0] * args.focal / args.Sx, 0., args.im_size[0] / 2],
            [0., args.im_size[1] * args.focal / args.Sy, args.im_size[1] / 2],
            [0., 0., 1.]]

        self.kk = torch.tensor(self.kk, device=device)

        # Load Models
        self.net = Loco(
            model=dic_models[args.mode],
            device=device,
            n_dropout=args.n_dropout,
            p_dropout=args.dropout,
            without_load_model=without_load_model)

        self.net.train()

        for m in self.net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
            if isinstance(m, torch.nn.Dropout):
                m.eval()

    def update_kk(self, im_size):

        self.kk = [
            [im_size[0] * self.focal / self.Sx, 0., im_size[0] / 2],
            [0., im_size[1] * self.focal / self.Sy, im_size[1] / 2],
            [0., 0., 1.]]

        self.kk = torch.tensor(self.kk, device=self.device)

    def forward(self, keypoints, boxes):
        """
        :param keypoints:  shape of keypoints is [[[x1, x2, conf]*17](person1), [[x1, x2, conf]*17](person2),
        [[x1, x2, conf]*17](person3), ..., [[x1, x2, conf]*17](personx)]
        :param boxes: shape of boxes is [[x_min, y_min, x_max, y_max](person1), [x_min, y_min, x_max, y_max](person2), ... ,
        [x_min, y_min, x_max, y_max](personx)]
        :return: dict_out: dict_keys(['gt', 'boxes', 'confs', 'dds_pred', 'stds_ale', 'stds_epi', 'xyz_pred',
                'uv_kps', 'uv_centers', 'uv_shoulders', 'uv_heads', # calculate from keypoints 'angles', 'angles_egocentric', 'aux'(no need)])
                to generate the virtual top view, the (x, y, angels) corresponds to (xyz_pred[index][0], xyz_pred[index][1], angles[index]

        """
        # return dic_out
        dic_out = self.net(keypoints, self.kk, boxes)
        dic_out['xz_pred'] = [[xyz[0], xyz[2]] for xyz in dic_out['xyz_pred']]
        return dic_out

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict_mono'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict_mono'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict_mono'][key]
        self.load_state_dict(process_dict)
        print('Load loconet all parameter from: ', filepath)


class Args():
    def __init__(self):
        self.ablation_caf_no_rescore = False
        self.ablation_cifseeds_nms = False
        self.ablation_cifseeds_no_rescore = False
        self.ablation_independent_kp = False
        self.activities = []
        self.basenet = None
        self.caf_seeds = False
        self.caf_th = 0.2
        self.camera = 0
        self.cf3_dropout = 0.0
        self.cf3_inplace_ops = True
        self.checkpoint = ''
        self.cif_th = 0.1
        self.command = 'predict'
        self.connection_method = 'blend'
        self.cross_talk = 0.0
        self.debug = False
        self.debug_indices = []
        self.decoder = None
        self.decoder_workers = None
        self.dense_connections = 0.0
        self.disable_cuda = False
        self.download_progress = True
        self.dpi = 150
        self.dropout = 0.2
        self.fast_rescaling = True
        self.focal = 5.7
        self.font_size = 0
        self.force_complete_caf_th = 0.001
        self.force_complete_pose = False
        self.glob = None
        self.greedy = False
        self.head_consolidation = 'filter_and_extend'
        self.hide_distance = False
        self.image_dpi_factor = 2.0
        self.image_height = None
        self.image_min_dpi = 50.0
        self.image_width = None
        self.images = ['', '']
        self.instance_threshold = None
        self.keypoint_threshold = 0.15
        self.keypoint_threshold_rel = 0.5
        self.line_width = None
        self.log_stats = False
        self.long_edge = None
        self.mobilenetv2_pretrained = True
        self.mobilenetv3_pretrained = True
        self.mode = 'mono'
        self.model = ''
        self.monocolor_connections = False
        self.n_dropout = 0
        self.net = None
        self.nms_before_force_complete = False
        self.no_save = False
        self.output_directory = ''
        self.output_types = ['multi']
        self.path_gt = None
        self.profile_decoder = None
        self.quiet = False
        self.radii = (0.3, 0.5, 1)
        self.resnet_block5_dilation = 1
        self.resnet_input_conv2_stride = 0
        self.resnet_input_conv_stride = 2
        self.resnet_pool0_stride = 0
        self.resnet_pretrained = True
        self.resnet_remove_last_block = False
        self.reverse_match = True
        self.save_all = None
        self.seed_threshold = 0.5
        self.show = False
        self.show_all = False
        self.show_box = False
        self.show_decoding_order = False
        self.show_file_extension = 'jpeg'
        self.show_frontier_order = False
        self.show_joint_confidences = False
        self.show_joint_scales = False
        self.show_only_decoded_connections = False
        self.shufflenetv2_pretrained = True
        self.shufflenetv2k_conv5_as_stage = False
        self.shufflenetv2k_group_norm = False
        self.shufflenetv2k_input_conv2_outchannels = None
        self.shufflenetv2k_input_conv2_stride = 0
        self.shufflenetv2k_instance_norm = False
        self.shufflenetv2k_kernel = 5
        self.shufflenetv2k_leaky_relu = False
        self.shufflenetv2k_stage4_dilation = 1
        self.skeleton_solid_threshold = 0.5
        self.squeezenet_pretrained = True
        self.text_color = 'white'
        self.textbox_alpha = 0.5
        self.threshold_dist = 2.5
        self.threshold_prob = 0.25
        self.video_dpi = 100
        self.video_fps = 10
        self.webcam = False
        self.white_overlay = False
        self.z_max = 100
        self.im_size = [1024, 768]
        self.Sx = 7.2  # nuScenes sensor size (mm)
        self.Sy = 5.4  # nuScenes sensor size (mm)


class Loco(torch.nn.Module):
    """
        Modify from an existing network
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    LINEAR_SIZE_MONO = 256
    N_SAMPLES = 100

    def __init__(self, model, device=None, n_dropout=0, p_dropout=0.2, linear_size=1024, without_load_model=False):

        super(Loco, self).__init__()

        # Select networks
        input_size = 34
        output_size = 9

        if not device:
            self.device = torch.device('cpu')
        else:
            self.device = device
        self.n_dropout = n_dropout
        self.epistemic = bool(self.n_dropout > 0)

        if without_load_model:
            self.model = LocoModel(p_dropout=p_dropout, input_size=input_size, output_size=output_size,
                                   linear_size=linear_size, device=self.device)
        else:
            # if the path is provided load the model parameters
            print('loading LocoNet model from %s' % model)
            if isinstance(model, str):
                model_path = model
                self.model = LocoModel(p_dropout=p_dropout, input_size=input_size, output_size=output_size,
                                       linear_size=linear_size, device=self.device)
                # print('loading mono model from ' + model_path)
                state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
                # combine state_dict with opimizer and epoch
                if len(state_dict.keys()) < 10:
                    # extract mono state dict
                    new_state_dict = OrderedDict()
                    for key, val in state_dict['state_dict_mono'].items():
                        new_state_dict[key[10:]] = val
                    self.model.load_state_dict(new_state_dict)
                else:
                    # map_location: gpu -> cpu
                    self.model.load_state_dict(state_dict)
            else:
                self.model = model
        self.model.to(self.device)

    def forward(self, keypoints, kk, boxes):
        """
        It includes preprocessing and postprocessing of data
        """

        # with torch.no_grad():
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints).to(self.device)
            kk = torch.tensor(kk).to(self.device)

        # openpif paf [17 * 3(x, y, score)]
        # inputs: [31, 34] -> keypoint [17 * 2(x, y)] # outpus: 31*9 = 31*8 + 31*9(1024 hidden layer)
        if len(keypoints.shape) < 3:
            keypoints = keypoints.unsqueeze(dim=0)
        inputs = preprocess_monoloco(keypoints, kk)

        outputs = self.model(inputs)
        dic_out = extract_outputs(outputs)

        if self.n_dropout > 0 and self.net != 'monstereo':
            varss = self.epistemic_uncertainty(inputs)
            dic_out['epi'] = varss
        else:
            dic_out['epi'] = [0.] * outputs.shape[0]
            # Add in the dictionary

        # self.post_process(dic_out, )
        dic_out = self.post_process(dic_out, boxes, keypoints, kk)
        return dic_out

    def epistemic_uncertainty(self, inputs):
        """
        Apply dropout at test time to obtain combined aleatoric + epistemic uncertainty
        """

        self.model.dropout.training = True  # Manually reactivate dropout in eval
        total_outputs = torch.empty((0, inputs.size()[0])).to(self.device)

        for _ in range(self.n_dropout):
            outputs = self.model(inputs)

            # Extract localization output
            if self.net == 'monoloco':
                db = outputs[:, 0:2]
            else:
                db = outputs[:, 2:4]

            # Unnormalize b and concatenate
            bi = unnormalize_bi(db)
            outputs = torch.cat((db[:, 0:1], bi), dim=1)

            samples = laplace_sampling(outputs, self.N_SAMPLES)
            total_outputs = torch.cat((total_outputs, samples), 0)
        varss = total_outputs.std(0)
        self.model.dropout.training = False
        return varss

    @staticmethod
    def post_process(dic_in, boxes, keypoints, kk, dic_gt=None, iou_min=0.3, reorder=True, verbose=False):
        """Post process monoloco to output final dictionary with all information for visualizations"""
        if len(boxes.shape) == 1:
            boxes = boxes.unsqueeze(dim=0)
        dic_out = defaultdict(list)
        if dic_in is None:
            return dic_out

        if dic_gt:
            boxes_gt = dic_gt['boxes']
            dds_gt = [el[3] for el in dic_gt['ys']]
            matches = get_iou_matches(boxes, boxes_gt, iou_min=iou_min)
            dic_out['gt'] = [True]
            if verbose:
                print("found {} matches with ground-truth".format(len(matches)))

            # Keep track of instances non-matched
            idxs_matches = [el[0] for el in matches]
            not_matches = [idx for idx, _ in enumerate(boxes) if idx not in idxs_matches]

        else:
            matches = []
            not_matches = list(range(len(boxes)))
            if verbose:
                print("NO ground-truth associated")

        if reorder and matches:
            matches = reorder_matches(matches, boxes, mode='left_right')

        all_idxs = [idx for idx, _ in matches] + not_matches
        dic_out['gt'] = [True] * len(matches) + [False] * len(not_matches)

        uv_shoulders = get_keypoints(keypoints, mode='shoulder')
        uv_heads = get_keypoints(keypoints, mode='head')
        uv_centers = get_keypoints(keypoints, mode='center')
        xy_centers = pixel_to_camera(uv_centers, kk, 1)

        # Add all the predicted annotations, starting with the ones that match a ground-truth
        for idx in all_idxs:
            kps = keypoints[idx]
            box = boxes[idx]
            # dd_pred = float(dic_in['d'][idx])
            # bi = float(dic_in['bi'][idx])
            # var_y = float(dic_in['epi'][idx])

            dd_pred = dic_in['d'][idx]
            bi = dic_in['bi'][idx]
            var_y = dic_in['epi'][idx]

            uu_s, vv_s = uv_shoulders.tolist()[idx][0:2]
            uu_c, vv_c = uv_centers.tolist()[idx][0:2]
            uu_h, vv_h = uv_heads.tolist()[idx][0:2]
            uv_shoulder = [round(uu_s), round(vv_s)]
            uv_center = [round(uu_c), round(vv_c)]
            uv_head = [round(uu_h), round(vv_h)]
            xyz_pred = xyz_from_distance(dd_pred, torch.tensor(xy_centers[idx], device=dd_pred.device))[0]
            distance = math.sqrt(float(xyz_pred[0]) ** 2 + float(xyz_pred[1]) ** 2 + float(xyz_pred[2]) ** 2)
            conf = 0.035 * (box[-1]) / (bi / distance)

            dic_out['boxes'].append(box)
            dic_out['confs'].append(conf)
            dic_out['dds_pred'].append(dd_pred)
            dic_out['stds_ale'].append(bi)
            dic_out['stds_epi'].append(var_y)

            # dic_out['xyz_pred'].append(xyz_pred.squeeze().tolist())
            dic_out['xyz_pred'].append(xyz_pred.squeeze())
            dic_out['uv_kps'].append(kps)
            dic_out['uv_centers'].append(uv_center)
            dic_out['uv_shoulders'].append(uv_shoulder)
            dic_out['uv_heads'].append(uv_head)

            # For MonStereo / MonoLoco++
            try:
                # dic_out['angles'].append(float(dic_in['yaw'][0][idx]))  # Predicted angle
                # dic_out['angles_egocentric'].append(float(dic_in['yaw'][1][idx]))  # Egocentric angle
                dic_out['angles'].append((dic_in['yaw'][0][idx]))  # Predicted angle
                dic_out['angles_egocentric'].append((dic_in['yaw'][1][idx]))  # Egocentric angle
            except KeyError:
                continue

            # Only for MonStereo
            # try:
            #     dic_out['aux'].append(float(dic_in['aux'][idx]))
            # except KeyError:
            #     continue

        for idx, idx_gt in matches:
            dd_real = dds_gt[idx_gt]
            xyz_real = xyz_from_distance(dd_real, xy_centers[idx])
            dic_out['dds_real'].append(dd_real)
            dic_out['boxes_gt'].append(boxes_gt[idx_gt])
            dic_out['xyz_real'].append(xyz_real.squeeze().tolist())
        return dic_out


def predict_new_clear_before():
    args = Args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    cnt = 0
    assert args.mode in ('keypoints', 'mono', 'stereo')
    # args:; dic_models -> {'keypoints': '', 'mono':''}
    # keypoints is the ['model_state_dict', 'epoch', 'meta'] = checkpoint
    # mono is the [mono.state_dict()](fc)
    args, dic_models = factory_from_args(args)

    # Load Models
    net = Loco(
        model=dic_models[args.mode],
        device=args.device,
        n_dropout=args.n_dropout,
        p_dropout=args.dropout,

    )

    # predict here not used to train
    # net.eval()
    # net.model.train()
    # net.eval()
    # net.model.train()

    # for openpifpaf predicitons
    predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    # data
    data = datasets.ImageList(args.images, preprocess=predictor.preprocess)

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

    # for batch_i, (_, _, meta_batch) in enumerate(data_loader):
    for i in range(10):
        batchdata = (args.images[0], args.images[1])
        # unbatch (only for MonStereo) # preds is the [openpifpaf.annotation.Annotation object] * 8
        # preds, _, meta = list(predictor.dataset(datasets.ImageList(args.images[batch_i:batch_i+1],preprocess=predictor.preprocess)))[0]
        # preds2, _, meta2 = list(predictor.dataset(datasets.ImageList(args.images[batch_i+1:batch_i+1 + 1],preprocess=predictor.preprocess)))[0]
        preds, _, meta = list(predictor.dataset(datasets.ImageList([batchdata[0]], preprocess=predictor.preprocess)))[0]
        preds2, _, meta2 = list(predictor.dataset(datasets.ImageList([batchdata[1]], preprocess=predictor.preprocess)))[
            0]

        # preds is the result of openpifpaf : keypoints, bbx, score, category_id
        # Load image and collect pifpaf results
        # pifpaf_outs = {
        #     # 'pred': preds, # all preds object list
        #     'left': [ann.json_data() for ann in preds] #  {'keypoints':[17 * 3], 'bbox':[4], 'score' : 0.814, 'category_id' : 1}
        # }
        #
        # pifpaf_outs2 = {
        #     # 'pred': preds2, # all preds object list
        #     'left': [ann.json_data() for ann in preds2] #  {'keypoints':[17 * 3], 'bbox':[4], 'score' : 0.814, 'category_id' : 1}
        # }

        # 3D Predictions
        # im_size = (image.shape[3], image.shape[2])
        im_size = (meta['width_height'][0], meta['width_height'][1])
        im_size2 = (meta2['width_height'][0], meta2['width_height'][1])
        # im_size = (cpu_image.size[0], cpu_image.size[1])  # Original
        # kk is the calibration matrix used to estimate the intrinsic and extrinsic parameters
        kk = [[718.3351, 0., 600.3891], [0., 718.3351, 181.5122], [0., 0., 1.]]  # Kitti calibration

        # Preprocess pifpaf outputs and run monoloco
        # boxes -> (x_min, y_min, x_max, y_max)
        # keypoints to three list() [split them to different list(x1, x2), (y1, y2), (z1, z2)))]
        # to open pifpaf the info get from the model is
        # 1. boxes and keypoints
        # 2. how to connect keypoints
        # enlarge the box 10% and cut off the outside part
        boxes, keypoints = preprocess_pifpaf(
            pifpaf_outs['left'], im_size, enlarge_boxes=False)
        boxes2, keypoints2 = preprocess_pifpaf(
            pifpaf_outs2['left'], im_size2, enlarge_boxes=False
        )
        # 1.extract the first two info, discard the confidence
        # 2.apply calibration matrix to keypoints
        # dic_out = net.forward(keypoints, kk)

        net.train()
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        # dic_out = net(keypoints, kk, boxes)
        dic_out2 = net(keypoints2, kk, boxes2)
        # print(dic_out['angles'])
        print(dic_out2['angles'])

        break
        cnt += 1


from munch import DefaultMunch


def get_args(**kws):
    kws = DefaultMunch.fromDict(kws)
    return kws


def predict_new_clear():
    args = Args()

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    cnt = 0
    assert args.mode in ('keypoints', 'mono', 'stereo')
    # args:; dic_models -> {'keypoints': '', 'mono':''}
    # keypoints is the ['model_state_dict', 'epoch', 'meta'] = checkpoint
    # mono is the [mono.state_dict()](fc)
    args, dic_models = factory_from_args(args)

    # Load Models
    net = Loco(
        model=dic_models[args.mode],
        device=args.device,
        n_dropout=args.n_dropout,
        p_dropout=args.dropout)

    # predict here not used to train
    # net.eval()
    # net.model.train()
    # net.eval()
    # net.model.train()

    # for openpifpaf predicitons
    # predictor = openpifpaf.Predictor(checkpoint=args.checkpoint)

    # data
    # data = datasets.ImageList(args.images, preprocess=predictor.preprocess)

    # data_loader = torch.utils.data.DataLoader(
    #     data, batch_size=args.batch_size, shuffle=False,
    #     pin_memory=False, collate_fn=datasets.collate_images_anns_meta)

    # for batch_i, (_, _, meta_batch) in enumerate(data_loader):
    for i in range(10):

        fv_sk_box = torch.load('/home/clark/dataset2/virtual_dataset/data6666/annotation/fv_sk_box.pth')

        keypoints, boxes = fv_sk_box['13_1']

        kk = [
            [args.im_size[0] * args.focal / args.Sx, 0., args.im_size[0] / 2],
            [0., args.im_size[1] * args.focal / args.Sy, args.im_size[1] / 2],
            [0., 0., 1.]]

        net.train()

        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
            if isinstance(m, torch.nn.Dropout):
                m.eval()
        dic_out = net(keypoints, kk, boxes)
        for key, val in dic_out.items():
            print(key, val)

        break
        cnt += 1


def factory_from_args(args):
    # Data
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")
    if args.path_gt is None:
        args.show_all = True

    # Models
    dic_models = {'keypoints': args.checkpoint, 'mono': args.model}
    args.checkpoint = dic_models['keypoints']

    logger.configure(args, LOG)  # logger first

    # Devices
    args.device = torch.device('cpu')
    args.pin_memory = False
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    # Add visualization defaults
    args.figure_width = 10
    args.dpi_factor = 1.0

    args.z_max = 10
    args.show_all = True
    args.no_save = True
    args.batch_size = 1

    if args.long_edge is None:
        args.long_edge = 144
    # Make default pifpaf argument
    args.force_complete_pose = True
    # LOG.info("Force complete pose is active")

    # Configure
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args, dic_models


class LocoModel(nn.Module):

    # change num_stage to 2
    def __init__(self, input_size, output_size=2, linear_size=512, p_dropout=0.2, num_stage=1, device='cuda'):
        super().__init__()

        self.num_stage = num_stage
        self.stereo_size = input_size
        self.mono_size = int(input_size / 2)
        self.output_size = output_size - 1
        self.linear_size = linear_size
        self.p_dropout = p_dropout
        self.num_stage = num_stage
        self.linear_stages = []
        self.device = device

        # Initialize weights

        # Preprocessing
        self.w1 = nn.Linear(self.stereo_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        # Internal loop
        for _ in range(num_stage):
            self.linear_stages.append(MyLinearSimple(self.linear_size, self.p_dropout))
        self.linear_stages = nn.ModuleList(self.linear_stages)

        # Post processing
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

        # ------------------------Other----------------------------------------------
        # Auxiliary
        self.w_aux = nn.Linear(self.linear_size, 1)

        # Final
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

        # NO-weight operations
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):

        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        # y = self.dropout(y) * (1/(1-self.p_dropout))
        y = self.dropout(y)

        for i in range(self.num_stage):
            y = self.linear_stages[i](y)

        # Auxiliary task
        # y = self.w2(y)
        aux = self.w_aux(y)

        # Final layers
        y = self.w3(y)
        y = self.batch_norm3(y)
        y = self.relu(y)
        # y = self.dropout(y) * (1/(1-self.p_dropout))
        y = self.dropout(y)
        y = self.w_fin(y)

        # Cat with auxiliary task
        y = torch.cat((y, aux), dim=1)
        return y


class MyLinearSimple(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y

        return out


def preprocess_monoloco(keypoints, kk, zero_center=False):
    """ Preprocess batches of inputs
    keypoints = torch tensors of (m, 3, 17)  or list [3,17]
    Outputs =  torch tensors of (m, 34) in meters normalized (z=1) and zero-centered using the center of the box
    """
    if isinstance(keypoints, list):
        keypoints = torch.tensor(keypoints)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    # Projection in normalized image coordinates and zero-center with the center of the bounding box
    uv_center = get_keypoints(keypoints, mode='center')
    xy1_center = pixel_to_camera(uv_center, kk, 10)
    xy1_all = pixel_to_camera(keypoints[:, 0:2, :], kk, 10)
    if zero_center:
        kps_norm = xy1_all - xy1_center.unsqueeze(1)  # (m, 17, 3) - (m, 1, 3)
    else:
        kps_norm = xy1_all
    kps_out = kps_norm[:, :, 0:2].reshape(kps_norm.size()[0], -1)  # no contiguous for view
    # kps_out = torch.cat((kps_out, keypoints[:, 2, :]), dim=1)
    return kps_out


def get_keypoints(keypoints, mode):
    """
    Extract center, shoulder or hip points of a keypoint
    Input --> list or torch/numpy tensor [(m, 3, 17) or (3, 17)]
    Output --> torch.tensor [(m, 2)]
    """
    if isinstance(keypoints, (list, np.ndarray)):
        keypoints = torch.tensor(keypoints)
    if len(keypoints.size()) == 2:  # add batch dim
        keypoints = keypoints.unsqueeze(0)
    assert len(keypoints.size()) == 3 and keypoints.size()[1] == 3, "tensor dimensions not recognized"
    assert mode in ['center', 'bottom', 'head', 'shoulder', 'hip', 'ankle']

    kps_in = keypoints[:, 0:2, :]  # (m, 2, 17)
    if mode == 'center':
        kps_max, _ = kps_in.max(2)  # returns value, indices
        kps_min, _ = kps_in.min(2)
        kps_out = (kps_max - kps_min) / 2 + kps_min  # (m, 2) as keepdims is False

    elif mode == 'bottom':  # bottom center for kitti evaluation
        kps_max, _ = kps_in.max(2)
        kps_min, _ = kps_in.min(2)
        kps_out_x = (kps_max[:, 0:1] - kps_min[:, 0:1]) / 2 + kps_min[:, 0:1]
        kps_out_y = kps_max[:, 1:2]
        kps_out = torch.cat((kps_out_x, kps_out_y), -1)

    elif mode == 'head':
        kps_out = kps_in[:, :, 0:5].mean(2)

    elif mode == 'shoulder':
        kps_out = kps_in[:, :, 5:7].mean(2)

    elif mode == 'hip':
        kps_out = kps_in[:, :, 11:13].mean(2)

    elif mode == 'ankle':
        kps_out = kps_in[:, :, 15:17].mean(2)

    return kps_out  # (m, 2)


def pixel_to_camera(uv_tensor, kk, z_met):
    """
    Convert a tensor in pixel coordinate to absolute camera coordinates
    It accepts lists or torch/numpy tensors of (m, 2) or (m, x, 2)
    where x is the number of keypoints
    """
    if isinstance(uv_tensor, (list, np.ndarray)):
        uv_tensor = torch.tensor(uv_tensor)
    if isinstance(kk, list):
        kk = torch.tensor(kk)
    if uv_tensor.size()[-1] != 2:
        uv_tensor = uv_tensor.permute(0, 2, 1)  # permute to have 2 as last dim to be padded
        assert uv_tensor.size()[-1] == 2, "Tensor size not recognized"
    uv_padded = F.pad(uv_tensor, pad=(0, 1), mode="constant", value=1)  # pad only last-dim below with value 1

    kk_1 = torch.inverse(kk)
    xyz_met_norm = torch.matmul(uv_padded, kk_1.t())  # More general than torch.mm
    xyz_met = xyz_met_norm * z_met

    return xyz_met


def xyz_from_distance(distances, xy_centers):
    """
    From distances and normalized image coordinates (z=1), extract the real world position xyz
    distances --> tensor (m,1) or (m) or float
    xy_centers --> tensor(m,3) or (3)
    """

    if isinstance(distances, float):
        distances = torch.tensor(distances).unsqueeze(0)
    if len(distances.size()) == 1:
        distances = distances.unsqueeze(1)
    if len(xy_centers.size()) == 1:
        xy_centers = xy_centers.unsqueeze(0)

    assert xy_centers.size()[-1] == 3 and distances.size()[-1] == 1, "Size of tensor not recognized"

    return xy_centers * distances / torch.sqrt(1 + xy_centers[:, 0:1].pow(2) + xy_centers[:, 1:2].pow(2))


def get_iou_matches(boxes, boxes_gt, iou_min=0.3):
    """From 2 sets of boxes and a minimum threshold, compute the matching indices for IoU matches"""

    matches = []
    used = []
    if not boxes or not boxes_gt:
        return []
    confs = [box[4] for box in boxes]

    indices = list(np.argsort(confs))
    for idx in indices[::-1]:
        box = boxes[idx]
        ious = []
        for box_gt in boxes_gt:
            iou = calculate_iou(box, box_gt)
            ious.append(iou)
        idx_gt_max = int(np.argmax(ious))
        if (ious[idx_gt_max] >= iou_min) and (idx_gt_max not in used):
            matches.append((int(idx), idx_gt_max))
            used.append(idx_gt_max)
    return matches


def calculate_iou(box1, box2):
    # Calculate the (x1, y1, x2, y2) coordinates of the intersection of box1 and box2. Calculate its Area.
    # box1 = [-3, 8.5, 3, 11.5]
    # box2 = [-3, 9.5, 3, 12.5]
    # box1 = [1086.84, 156.24, 1181.62, 319.12]
    # box2 = [1078.333357, 159.086347, 1193.771014, 322.239107]

    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max((xi2 - xi1), 0) * max((yi2 - yi1), 0)  # Max keeps into account not overlapping box

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


def reorder_matches(matches, boxes, mode='left_rigth'):
    """
    Reorder a list of (idx, idx_gt) matches based on position of the detections in the image
    ordered_boxes = (5, 6, 7, 0, 1, 4, 2, 4)
    matches = [(0, x), (2,x), (4,x), (3,x), (5,x)]
    Output --> [(5, x), (0, x), (3, x), (2, x), (5, x)]
    """

    assert mode == 'left_right'

    # Order the boxes based on the left-right position in the image and
    ordered_boxes = np.argsort([box[0] for box in boxes])  # indices of boxes ordered from left to right
    matches_left = [int(idx) for (idx, _) in matches]

    return [matches[matches_left.index(idx_boxes)] for idx_boxes in ordered_boxes if idx_boxes in matches_left]


def laplace_sampling(outputs, n_samples):
    torch.manual_seed(1)
    mu = outputs[:, 0]
    bi = torch.abs(outputs[:, 1])

    # Analytical
    # uu = np.random.uniform(low=-0.5, high=0.5, size=mu.shape[0])
    # xx = mu - bi * np.sign(uu) * np.log(1 - 2 * np.abs(uu))

    # Sampling
    cuda_check = outputs.is_cuda
    if cuda_check:
        get_device = outputs.get_device()
        device = torch.device(type="cuda", index=get_device)
    else:
        device = torch.device("cpu")

    laplace = torch.distributions.Laplace(mu, bi)
    xx = laplace.sample((n_samples,)).to(device)

    return xx


def unnormalize_bi(loc):
    """
    Unnormalize relative bi of a nunmpy array
    Input --> tensor of (m, 2)
    """
    assert loc.size()[1] == 2, "size of the output tensor should be (m, 2)"
    bi = torch.exp(loc[:, 1:2]) * loc[:, 0:1]

    return bi


def extract_outputs(outputs, tasks=()):
    dic_out = {'x': outputs[:, 0:1],
               'y': outputs[:, 1:2],
               'd': outputs[:, 2:4],
               'h': outputs[:, 4:5],
               'w': outputs[:, 5:6],
               'l': outputs[:, 6:7],
               'ori': outputs[:, 7:9]}

    bi = unnormalize_bi(dic_out['d'])
    dic_out['bi'] = bi

    x = to_cartesian(outputs[:, 0:3], mode='x')
    y = to_cartesian(outputs[:, 0:3], mode='y')
    d = dic_out['d'][:, 0:1]
    z = torch.sqrt(d ** 2 - x ** 2 - y ** 2)
    dic_out['xyzd'] = torch.cat((x, y, z, d), dim=1)
    dic_out.pop('d')
    dic_out.pop('x')
    dic_out.pop('y')
    dic_out['d'] = d

    yaw_pred = torch.atan2(dic_out['ori'][:, 0:1], dic_out['ori'][:, 1:2])
    yaw_orig = back_correct_angles(yaw_pred, dic_out['xyzd'][:, 0:3])
    dic_out['yaw'] = (yaw_pred, yaw_orig)  # alpha, ry

    # if outputs.shape[1] == 10:
    #     dic_out['aux'] = torch.sigmoid(dic_out['aux'])
    return dic_out


def to_cartesian(rtp, mode=None):
    """convert from spherical to cartesian"""

    if isinstance(rtp, torch.Tensor):
        if mode in ('x', 'y'):
            r = rtp[:, 2]
            t = rtp[:, 0]
            p = rtp[:, 1]
        if mode == 'x':
            x = r * torch.sin(p) * torch.cos(t)
            return x.view(-1, 1)

        if mode == 'y':
            y = r * torch.cos(p)
            return y.view(-1, 1)

        xyz = rtp.clone()
        xyz[:, 0] = rtp[:, 0] * torch.sin(rtp[:, 2]) * torch.cos(rtp[:, 1])
        xyz[:, 1] = rtp[:, 0] * torch.cos(rtp[:, 2])
        xyz[:, 2] = rtp[:, 0] * torch.sin(rtp[:, 2]) * torch.sin(rtp[:, 1])
        return xyz

    x = rtp[0] * math.sin(rtp[2]) * math.cos(rtp[1])
    y = rtp[0] * math.cos(rtp[2])
    z = rtp[0] * math.sin(rtp[2]) * math.sin(rtp[1])
    return [x, y, z]


def back_correct_angles(yaws, xyz):
    corrections = torch.atan2(xyz[:, 0], xyz[:, 2])
    yaws = yaws + corrections.view(-1, 1)
    mask_up = yaws > math.pi
    yaws[mask_up] -= 2 * math.pi
    mask_down = yaws < -math.pi
    yaws[mask_down] += 2 * math.pi
    # TODO MAYBE PROBLEM HERE
    # assert torch.all(yaws < math.pi) & torch.all(yaws > - math.pi)
    return yaws


def preprocess_pifpaf(annotations, im_size=None, enlarge_boxes=True, min_conf=0.3, max_area=180000):
    """
    Preprocess pif annotations:
    1. enlarge the box of 10%
    2. Constraint it inside the image (if image_size provided)
    """

    boxes = []
    keypoints = []
    enlarge = 1 if enlarge_boxes else 2  # Avoid enlarge boxes for social distancing

    for dic in annotations:
        kps = prepare_pif_kps(dic['keypoints'])
        box = dic['bbox']
        try:
            conf = dic['score']
            # Enlarge boxes
            delta_h = (box[3]) / (10 * enlarge)
            delta_w = (box[2]) / (5 * enlarge)
            # from width height to corners
            box[2] += box[0]
            box[3] += box[1]

        except KeyError:
            all_confs = np.array(kps[2])
            score_weights = np.ones(17)
            score_weights[:3] = 3.0
            score_weights[5:] = 0.1
            # conf = np.sum(score_weights * np.sort(all_confs)[::-1])
            conf = float(np.mean(all_confs))
            # Add 15% for y and 20% for x
            delta_h = (box[3] - box[1]) / (7 * enlarge)
            delta_w = (box[2] - box[0]) / (3.5 * enlarge)
            assert delta_h > -5 and delta_w > -5, "Bounding box <=0"

        box[0] -= delta_w
        box[1] -= delta_h
        box[2] += delta_w
        box[3] += delta_h

        # Put the box inside the image
        if im_size is not None:
            box[0] = max(0, box[0])
            box[1] = max(0, box[1])
            box[2] = min(box[2], im_size[0])
            box[3] = min(box[3], im_size[1])

        if conf >= min_conf and get_box_area(box) <= max_area:
            box.append(conf)
            boxes.append(box)
            keypoints.append(kps)

    return boxes, keypoints


def get_box_area(box):
    data = box
    area = (data[2] - data[0]) * (data[3] - data[1])
    return area


def prepare_pif_kps(kps_in):
    """Convert from a list of 51 to a list of 3, 17"""

    assert len(kps_in) % 3 == 0, "keypoints expected as a multiple of 3"
    xxs = kps_in[0:][::3]
    yys = kps_in[1:][::3]  # from offset 1 every 3
    ccs = kps_in[2:][::3]

    return [xxs, yys, ccs]
