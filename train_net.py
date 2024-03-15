import torch.optim as optim
import copy
from collections import Counter, OrderedDict
from nets import reidnet, mono21
from torch.utils import data
from config.config import *
from dataset.dataset import *
from utils.utils import *
from utils.draw_fig import *
from itertools import chain
from utils.utils import get_cos_distance_matrix, get_top_pair_from_matrix
import random
import torchvision as tv
from tqdm import tqdm

rgb_table = [[191, 36, 42], [255, 70, 31], [255, 181, 30], [23, 133, 170],
             [22, 169, 81], [255, 242, 223], [0, 52, 115], [255, 0, 255],
             [254, 71, 119], [0, 100, 0], [189, 221, 34], [163, 226, 197],
             [62, 237, 232], [0, 191, 255], [186, 202, 199], [204, 164, 227],
             [87, 0, 79], [205, 92, 92], [0, 0, 255], [255, 0, 0], [0, 255, 0],
             [77, 34, 26], [254, 241, 67], [132, 90, 50], [65, 85, 92], [119, 32, 55]]


def test_more_view_aggregation_net(cfg, training_set=None, validation_set=None):
    """
        training monoreid
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    cfg.fps_dict = torch.load(os.path.join(cfg.label, 'fps.pth'))  # [frame_id(1)] -> [pid(1)] = [x, y, r]
    cfg.fp_dict = torch.load(os.path.join(cfg.label, 'fp.pth'))  # ['frameid_pid'] = [x, y, r]

    candidate_view_dataset = []
    # for i in range(2, 5):
    for i in range(2, cfg.view_num):
        cfg.view_num = i
        if i == 2:
            view1_dataset, view2_dataset = return_more_view_dataset(cfg)
            candidate_view_dataset.append(view1_dataset)
            candidate_view_dataset.append(view2_dataset)
        else:
            _, view2_dataset = return_more_view_dataset(cfg)
            candidate_view_dataset.append(view2_dataset)

    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Setting data devices
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model_mono = mono21.mono(model=cfg.loconet_pretrained_model_path, without_load_model=cfg.without_load_model)
    model_reid = reidnet.reidnet()

    if cfg.iscontinue:
        assert cfg.continue_path != ''
        model_mono.loadmodel(cfg.continue_path)
        model_reid.loadmodel(cfg.continue_path)

    model_mono = model_mono.to(device=device)
    model_reid = model_reid.to(device=device)

    # model_mono.train in the inner of model_mono

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, chain(model_mono.parameters(), model_reid.parameters())),
                           lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    start_epoch = 1
    if cfg.iscontinue:
        state = torch.load(cfg.continue_path)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

    test = _test_monoreid_cam_gt_centroid_soft_choose_more_view

    test_info = test(candidate_view_dataset, model_mono, model_reid, device, 0, cfg)
    show_aggregation_info_loss('Test', cfg.log_path, test_info)


def test_more_view_aggregation_without_fig_net(cfg, training_set=None, validation_set=None):
    """
        training monoreid
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    cfg.fps_dict = torch.load(os.path.join(cfg.label, 'fps.pth'))
    cfg.fp_dict = torch.load(os.path.join(cfg.label, 'fp.pth'))

    # collecting the datasets from all views
    candidate_view_dataset = []
    for i in range(2, cfg.view_num):
        cfg.view_num = i
        if i == 2:
            view1_dataset, view2_dataset = return_more_view_dataset(cfg)
            candidate_view_dataset.append(view1_dataset)
            candidate_view_dataset.append(view2_dataset)
        else:
            _, view2_dataset = return_more_view_dataset(cfg)
            candidate_view_dataset.append(view2_dataset)

    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # setting data devices
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model_mono = mono21.mono(model=cfg.loconet_pretrained_model_path, without_load_model=cfg.without_load_model)
    model_reid = reidnet.reidnet()

    if cfg.iscontinue:
        assert cfg.continue_path != ''
        model_mono.loadmodel(cfg.continue_path)
        model_reid.loadmodel(cfg.continue_path)

    model_mono = model_mono.to(device=device)
    model_reid = model_reid.to(device=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, chain(model_mono.parameters(), model_reid.parameters())),
                           lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    if cfg.iscontinue:
        state = torch.load(cfg.continue_path)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

    test = _test_monoreid_cam_gt_centroid_soft_choose_more_view_without_fig

    test_info = test(candidate_view_dataset, model_mono, model_reid, device, 0, cfg)
    show_aggregation_info_loss('Test', cfg.log_path, test_info)


def train_monoreid_net_with_xy_and_r(cfg, training_set=None, validation_set=None):
    """
        training BEVSee's LonoNet and Resnet-50 (train ReID by self supervised way)
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    cfg.fps_dict = torch.load(os.path.join(cfg.label, 'fps.pth'))
    cfg.fp_dict = torch.load(os.path.join(cfg.label, 'fp.pth'))

    # Reading dataset
    if (training_set is None) or (validation_set is None):
        # train_dataloader,  valid dataloader
        training_set, validation_set = return_dataset(cfg)

    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': cfg.num_workers,
        'pin_memory': cfg.train_pin_memory,
        'drop_last': True,
        'worker_init_fn': seed_worker,
        'generator': g,
    }

    params_test = {
        'batch_size': cfg.test_batch_size,
        'shuffle': False,
        'num_workers': cfg.num_workers,
        'pin_memory': False,
        'worker_init_fn': seed_worker,
        'generator': g,
    }

    training_loader = data.DataLoader(training_set, **params)
    validation_loader = data.DataLoader(validation_set, **params_test)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Set data position
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # Build model and optimizer
    model_mono = mono21.mono(model=cfg.loconet_pretrained_model_path, without_load_model=cfg.without_load_model)
    model_reid = reidnet.reidnet()

    ## choosing model ##

    if cfg.iscontinue:
        assert cfg.continue_path != ''
        model_mono.loadmodel(cfg.continue_path)
        model_reid.loadmodel(cfg.continue_path)

    model_mono = model_mono.to(device=device)
    model_reid = model_reid.to(device=device)
    model_reid.train()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, chain(model_mono.parameters(), model_reid.parameters())),
                           lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    start_epoch = 1
    if cfg.iscontinue:
        state = torch.load(cfg.continue_path)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

    train = _train_monoreid_cam_gt_centroid
    test = _test_monoreid_cam_gt_centroid_soft_choose

    # recording loss curves and best model result
    train_reid_loss = []
    train_mono_xy_loss = []
    train_mono_r_loss = []
    test_reid_loss = []
    test_mono_xy_loss = []
    test_mono_r_loss = []
    train_loss = []
    test_loss = []
    test_xy_prob_dict_loss = []
    test_r_prob_dict_loss = []
    test_matrix_iou = []

    best_loss = 999999
    best_loss_info = ''
    high_f1_loss = 999999
    high_f1_info = ''

    if cfg.test_before_train or cfg.only_test:
        test_info = test(validation_loader, model_mono, model_reid, device, 0, cfg)

        show_monoreid_epoch_test_info_contain_gt('Test', cfg.log_path, test_info)
        torch.cuda.empty_cache()
        if cfg.only_test:
            return

        test_loss.append(test_info['loss'])
        test_reid_loss.append(test_info['re_id_loss'])
        test_mono_xy_loss.append(test_info['mono_xy_loss'])
        test_mono_r_loss.append(test_info['mono_r_loss'])
        test_xy_prob_dict_loss.append(test_info['prob'][0])
        test_r_prob_dict_loss.append(test_info['prob'][1])
        test_matrix_iou.append(test_info['iou'])

    for epoch in range(start_epoch, start_epoch + cfg.max_epoch):

        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])

        train_info = train(training_loader, model_mono, model_reid, device, optimizer, epoch, cfg)

        train_reid_loss.append(train_info['re_id_loss'])
        train_mono_xy_loss.append(train_info['mono_xy_loss'])
        train_mono_r_loss.append(train_info['mono_r_loss'])
        train_loss.append(train_info['loss'])

        show_monoreid_epoch_info('Train', cfg.log_path, train_info)

        if epoch % cfg.test_interval_epoch == 0:
            test_info = test(validation_loader, model_mono, model_reid, device, epoch, cfg)

            if test_info['mono_xy_loss'] + test_info['mono_r_loss'] < best_loss:
                # best_loss = test_info['loss']
                best_loss = test_info['mono_xy_loss'] + test_info['mono_r_loss']
                best_loss_info = test_info
            if test_info['iou'] >= cfg.reid_f1_threshold and test_info['mono_xy_loss'] + test_info[
                'mono_r_loss'] < high_f1_loss:
                high_f1_loss = test_info['mono_xy_loss'] + test_info['mono_r_loss']
                high_f1_info = test_info

            torch.cuda.empty_cache()
            show_monoreid_epoch_test_info_contain_gt('Test', cfg.log_path, test_info)

            # show best loss
            if best_loss_info != '':
                show_monoreid_epoch_test_info_best_loss('Test', cfg.log_path, best_loss_info)
            if high_f1_info != '':
                show_monoreid_epoch_test_info_best_f1_loss('Test', cfg.log_path, high_f1_info)

            test_loss.append(test_info['loss'])
            test_reid_loss.append(test_info['re_id_loss'])
            test_mono_xy_loss.append(test_info['mono_xy_loss'])
            test_mono_r_loss.append(test_info['mono_r_loss'])
            test_xy_prob_dict_loss.append(test_info['prob'][0])
            test_r_prob_dict_loss.append(test_info['prob'][1])
            test_matrix_iou.append(test_info['iou'])

            # Save model
            if epoch % cfg.save_model_interval_epoch == 0:
                state = {
                    'epoch': epoch,
                    'state_dict_mono': model_mono.state_dict(),
                    'state_dict_reid': model_reid.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                filepath = cfg.result_path + '/epoch%d.pth' % (epoch)
                torch.save(state, filepath)
                print('model saved to:', filepath)

        # visualization of loss curves
        if cfg.draw_fig and epoch % cfg.draw_fig_interval_epoch == 0:
            filepath = cfg.result_path

            draw_line_fig(train_loss, filepath, is_train=True, is_timestample=False, extra='total')
            draw_line_fig(train_reid_loss, filepath, is_train=True, is_timestample=False, extra='reid')
            draw_line_fig(train_mono_xy_loss, filepath, is_train=True, is_timestample=False, extra='cam_xy')
            draw_line_fig(train_mono_r_loss, filepath, is_train=True, is_timestample=False, extra='cam_angle')

            draw_line_fig(test_loss, filepath, is_train=False, is_timestample=False, extra='total')
            draw_line_fig(test_reid_loss, filepath, is_train=False, is_timestample=False, extra='reid')
            draw_line_fig(test_mono_xy_loss, filepath, is_train=False, is_timestample=False, extra='cam_xy')
            draw_line_fig(test_mono_r_loss, filepath, is_train=False, is_timestample=False, extra='cam_angle')
            draw_line_fig(test_matrix_iou, filepath, is_train=False, is_timestample=False, extra='matrix_f1')

            draw_more_line_fig_key_from_dict(test_xy_prob_dict_loss, filepath, True)
            draw_more_line_fig_key_from_dict(test_r_prob_dict_loss, filepath, False)


def test_more_view_aggregation_cosine_similarity(cfg, training_set=None, validation_set=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.device_list

    # Show config parameters
    cfg.init_config()
    show_config(cfg)

    # loading more datasets from seq_list
    final_dataset_dict = {}

    start_time = time.time()
    for seq_name in cfg.real_dataset_name_list:
        cfg_inner = Config(seq_name)
        cfg_inner.aggregate_view_num = (cfg_inner.view_num + 1) if cfg_inner.view_num == 2 else cfg_inner.view_num

        candidate_view_dataset = []
        for i in range(2, cfg_inner.aggregate_view_num):
            cfg_inner.view_num = i
            if i == 2:
                view1_dataset, view2_dataset = return_more_view_dataset_real_without_annotation(cfg_inner)
                candidate_view_dataset.append(view1_dataset)
                if view2_dataset is not None:
                    candidate_view_dataset.append(view2_dataset)
            else:
                _, view2_dataset = return_more_view_dataset_real_without_annotation(cfg_inner)
                candidate_view_dataset.append(view2_dataset)
        final_dataset_dict[seq_name] = candidate_view_dataset
    end_time = time.time()
    print(f'Loading seq[{len(cfg.real_dataset_name_list)}] in {round(end_time - start_time, 2)}s')

    g = torch.Generator()
    g.manual_seed(cfg.train_random_seed)

    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    torch.cuda.manual_seed_all(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    # Set data position
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    img_size_dict = {}
    img_size_dict['V2_G1'] = [2688, 1512]
    img_size_dict['V2_G3'] = [2688, 1512]
    img_size_dict['V3_G2'] = [2688, 1512]
    img_size_dict['V3_G3'] = [2704, 1520]
    img_size_dict['V3_G5'] = [2704, 1520]
    img_size_dict['V4_G1'] = [2704, 1520]
    img_size_dict['V4_G2'] = [2704, 1520]

    case_image_path = os.path.join(cfg.root_view, os.listdir(cfg.root_view)[0])
    case_img = tv.io.read_image(case_image_path)
    im_size = [case_img.shape[2], case_img.shape[1]]

    ## choosing model ##
    model_mono = mono21.mono(model=cfg.loconet_pretrained_model_path, without_load_model=cfg.without_load_model,
                             im_size=im_size)
    model_reid = reidnet.reidnet()

    if cfg.iscontinue:
        assert cfg.continue_path != ''
        model_mono.loadmodel(cfg.continue_path)
        model_reid.loadmodel(cfg.continue_path)
        if cfg.is_only_used_pretrained_reid:
            model_reid = reidnet.reidnet()
    if cfg.reload_reid_path != '' and os.path.exists(cfg.reload_reid_path):
        model_reid.loadmodel(cfg.reload_reid_path)
        print(f'Reload reid from {cfg.reload_reid_path}')

    model_mono = model_mono.to(device=device)
    model_reid = model_reid.to(device=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, chain(model_mono.parameters(), model_reid.parameters())),
                           lr=cfg.train_learning_rate, weight_decay=cfg.weight_decay)

    start_epoch = 1
    if cfg.iscontinue:
        state = torch.load(cfg.continue_path)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']

    # train = _train_monoreid_cam_gt_centroid_without_r
    test = _test_monoreid_cam_gt_centroid_soft_choose_more_view_without_annotation_with_pred_tracking_original_way
    single_view_avg_list = []
    pair_avg_list = []
    view_num_to_agg_list_dict = {}

    original_cfg_result_path = cfg.result_path
    cos_similarity_file = os.path.join(original_cfg_result_path, 'similarity.txt')
    final_dataset_dict_items = tqdm(final_dataset_dict.items())
    time_extra = ''.join(str(time.time()).split('.'))
    for seq_name, candidate_view_dataset in final_dataset_dict_items:

        # update kk
        sub_seq_name = seq_name[6:6 + 5]
        img_size = img_size_dict[sub_seq_name]
        model_mono.update_kk(img_size)

        final_dataset_dict_items.set_description(f'Processing {seq_name}')
        new_result_path = os.path.join(original_cfg_result_path, seq_name)
        os.makedirs(new_result_path, exist_ok=True)
        cfg.result_path = new_result_path
        test_info = test(candidate_view_dataset, model_mono, model_reid, device, 0, cfg, time_extra)
        view_avg = test_info['view_avg']
        pair_avg = test_info['pair_avg']
        single_avg = test_info['single_view_avg']
        view_num = test_info['view_num']
        if view_num in view_num_to_agg_list_dict.keys():
            view_num_to_agg_list_dict[view_num].append(view_avg)
        else:
            view_num_to_agg_list_dict[view_num] = [view_avg]
        single_view_avg_list.append(single_avg)
        pair_avg_list.append(pair_avg)

        single_avg_val = sum(single_view_avg_list) / len(single_view_avg_list)
        pair_avg_val = sum(pair_avg_list) / len(pair_avg_list)
        with open(cos_similarity_file, 'a') as f:
            f.write(f'{seq_name}: single cos: {single_avg_val}, pair cos: {pair_avg_val} ')
            for key, val_list in view_num_to_agg_list_dict.items():
                f.write(f' View {key}: {sum(val_list) / len(val_list)} ')
            f.write('\n')
    # print final results
    for key, val_list in view_num_to_agg_list_dict.items():
        print(f' View {key}: {sum(val_list) / len(val_list)} ')


def _train_monoreid_cam_gt_centroid(data_loader, model_mono, model_reid, device, optimizer, epoch, cfg):
    cfg.epoch = epoch

    loss_meter = AverageMeter_MonoReid()
    epoch_timer = Timer()
    for iter, batch_data in enumerate(data_loader):

        if batch_data[4].shape[1] == 1 or batch_data[6].shape[1] == 1:
            continue

        ####        setting train and eval mode     ####
        model_mono.train()
        for m in model_mono.net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
            if isinstance(m, torch.nn.Dropout):
                m.eval()

        model_reid.train()

        ####        initialize optimizer and batch_data     ####
        optimizer.zero_grad()
        # prepare batch data
        batch_data = [try_to(b, device) for b in batch_data]

        #### [VTM]       loconet forward        ####
        output_dict1 = model_mono(batch_data[4].squeeze(), batch_data[5].squeeze())
        output_dict2 = model_mono(batch_data[6].squeeze(), batch_data[7].squeeze())
        original_xz_list2 = output_dict2['xz_pred']
        original_xz_list1 = output_dict1['xz_pred']
        original_angle_list2 = output_dict2['angles']
        original_angle_list1 = output_dict1['angles']

        ####        reid forward        ####
        reid_output1 = model_reid(batch_data[2].squeeze())
        reid_output2 = model_reid(batch_data[3].squeeze())
        match_matrix = get_eu_distance_mtraix(reid_output1, reid_output2)

        ####  [SAM]      rotating  and caculating camera position and loss      ####
        #  getting top3 point pairs
        selected_pair_list = get_top_pair_from_matrix(match_matrix.clone().detach(), 3)
        # getting candidate camera poses
        theta_list, deltax_list, deltay_list = get_theta_delx_dely_from_pairs_with_fn(output_dict1['angles'],
                                                                                      output_dict2['angles'],
                                                                                      output_dict1['xz_pred'],
                                                                                      output_dict2['xz_pred'],
                                                                                      selected_pair_list)

        # centroid selection strategy
        x_centroid = 0
        y_centroid = 0

        for i in range(len(deltax_list)):
            x_centroid += deltax_list[i].item()
            y_centroid += deltay_list[i].item()
        x_centroid /= len(deltay_list)
        y_centroid /= len(deltay_list)

        select_index = 0
        min_distance = 9999
        for i in range(len(theta_list)):
            distance = math.sqrt((deltax_list[i] - x_centroid) ** 2 + (deltay_list[i] - y_centroid) ** 2)
            if distance < min_distance:
                select_index = i
                min_distance = distance

        # creating camera x, y, r gt
        x_gt, y_gt, r_gt = batch_data[8][0][0].item(), batch_data[8][0][1].item(), batch_data[8][0][2].item()
        x_gt *= cfg.gt_ratio
        y_gt *= cfg.gt_ratio
        r_gt -= 90
        if r_gt > 180:
            r_gt = r_gt - 360
        r_gt /= 57.3

        # getting camera losses
        cam_xy_loss = 0
        cam_r_loss = 0
        for i in range(min(2, len(theta_list))):
            theta_output = theta_list[i] * 57.3
            if theta_output < 0:
                theta_output += 360
            theta_output = 360 - theta_output

            theta_output -= 90
            if theta_output > 180:
                theta_output = theta_output - 360
            r_pred = theta_output / 57.3
            x_pred = deltax_list[i]
            y_pred = deltay_list[i]

            cam_xy_loss += torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
            cam_r_loss += clac_rad_distance(r_gt, r_pred)

        #### [Association part of SAM]        calculating reid matrix loss     ####
        # creating pseudo matrix
        new_xzpred2, new_angle2 = rotate_x_y_by_point(output_dict2['xz_pred'], output_dict2['angles'],
                                                      delta_x=deltax_list[select_index],
                                                      delta_y=deltay_list[select_index],
                                                      delta_theta=theta_list[select_index])

        pseudo_matrix = torch.zeros_like(match_matrix)
        pseudo_r_matrix = torch.zeros_like(match_matrix)
        n, m = pseudo_matrix.shape
        for i in range(n):
            for j in range(m):
                pseudo_matrix[i][j] = math.sqrt(((new_xzpred2[i][0].item() - original_xz_list1[j][0].item()) ** 2 + (
                        new_xzpred2[i][1].item() - original_xz_list1[j][1].item()) ** 2))
                pseudo_r_matrix[i][j] = clac_rad_distance(new_angle2[i], original_angle_list1[j]).item()

        pseudo_matrix[pseudo_matrix < 0.0002] += 0.0001
        pseudo_r_matrix[pseudo_r_matrix < 0.0002] += 0.0001

        # converting the pseudo matrix to 0-1 matrix
        n, m = pseudo_matrix.shape
        matrix_dis_pseudo = 1 / pseudo_matrix
        matrix_r_pseudo = 1 / pseudo_r_matrix

        # processing distance matrix
        for i in range(n):
            line_max = matrix_dis_pseudo[i].max()
            line_min = matrix_dis_pseudo[i].min()
            size = line_max - line_min
            for j in range(m):
                matrix_dis_pseudo[i][j] = (matrix_dis_pseudo[i][j] - line_min) / size

        # processing angle matrix
        for i in range(n):
            line_max = matrix_r_pseudo[i].max()
            line_min = matrix_r_pseudo[i].min()
            size = line_max - line_min
            for j in range(m):
                matrix_r_pseudo[i][j] = (matrix_r_pseudo[i][j] - line_min) / size
        matrix_r_pseudo[matrix_r_pseudo < 0.9] = 0

        # commented-out code for gt matrix
        # id1 = batch_data[10]
        # id2 = batch_data[11]
        # matrix_gt = torch.zeros_like(match_matrix)
        # for i in range(id2.squeeze().shape[0]):
        #     for j in range(id1.squeeze().shape[0]):
        #         if id1.squeeze()[j] == id2.squeeze()[i]:
        #             matrix_gt[i][j] = 1.0

        matrix_pseudo = cfg.dis_pseudo_ratio * matrix_dis_pseudo + (1 - cfg.dis_pseudo_ratio) * matrix_r_pseudo;
        re_id_matrix_loss = torch.nn.MSELoss()(match_matrix, matrix_pseudo)

        # getting the total loss
        total_loss = re_id_matrix_loss + cfg.xy_ratio * cam_xy_loss + cfg.r_ratio * cam_r_loss

        # recording the loss data to loss_meter
        mono_xy_record = cam_xy_loss.item()
        mono_r_record = cam_r_loss.item()
        loss_meter.update(re_id_matrix_loss.item(), mono_xy_record, mono_r_record)

        total_loss.backward()
        optimizer.step()

    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        're_id_loss': loss_meter.reid_loss_avg,
        'mono_xy_loss': loss_meter.mono_xy_loss_avg,
        'mono_r_loss': loss_meter.mono_r_loss_avg,
        'loss': loss_meter.total_loss,
    }
    return train_info


def _train_monoreid_cam_gt_centroid(data_loader, model_mono, model_reid, device, optimizer, epoch, cfg):
    # Input augmented data,
    # train vindicator
    # save vindicator and return

    cfg.epoch = epoch

    # here is meter
    loss_meter = AverageMeter_MonoReid()
    epoch_timer = Timer()
    for iter, batch_data in enumerate(data_loader):

        if batch_data[4].shape[1] == 1 or batch_data[6].shape[1] == 1:
            continue

        ####        setting train and eval mode     ####
        model_mono.train()

        for m in model_mono.net.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.eval()
            if isinstance(m, torch.nn.Dropout):
                m.eval()

        model_reid.train()
        ####        setting train and eval mode     ####

        ####        initialize optimizer and batch_data     ####
        optimizer.zero_grad()
        # prepare batch data
        batch_data = [try_to(b, device) for b in batch_data]
        ####        initialize optimizer and batch_data     ####

        ####        mono forward        ####

        output_dict1 = model_mono(batch_data[4].squeeze(), batch_data[5].squeeze())
        # output_dict2 = model_mono(batch_data[6].squeeze(), batch_data[7].squeeze())
        output_dict2 = model_mono(batch_data[6].squeeze(), batch_data[7].squeeze())

        original_xz_list2 = output_dict2['xz_pred']
        original_xz_list1 = output_dict1['xz_pred']
        original_angle_list2 = output_dict2['angles']
        original_angle_list1 = output_dict1['angles']
        ####        mono forward        ####

        ####        reid forward        ####
        reid_output1 = model_reid(batch_data[2].squeeze())
        reid_output2 = model_reid(batch_data[3].squeeze())

        # match_matrix = get_cos_distance_matrix(reid_output1, reid_output2)
        match_matrix = get_eu_distance_mtraix(reid_output1, reid_output2)
        ####        reid forward        ####

        ####        rotation  and caculating camera position loss      ####
        #### using point of index 2
        selected_pair_list = get_top_pair_from_matrix(match_matrix.clone().detach(), 3)

        theta_list, deltax_list, deltay_list = get_theta_delx_dely_from_pairs_with_fn(output_dict1['angles'],
                                                                                      output_dict2['angles'],
                                                                                      output_dict1['xz_pred'],
                                                                                      output_dict2['xz_pred'],
                                                                                      selected_pair_list)

        # creating centroid
        x_centroid = 0
        y_centroid = 0

        for i in range(len(deltax_list)):
            x_centroid += deltax_list[i].item()
            y_centroid += deltay_list[i].item()
        x_centroid /= len(deltay_list)
        y_centroid /= len(deltay_list)

        select_index = 0
        min_distance = 9999
        for i in range(len(theta_list)):
            distance = math.sqrt((deltax_list[i] - x_centroid) ** 2 + (deltay_list[i] - y_centroid) ** 2)
            if distance < min_distance:
                select_index = i
                min_distance = distance

        # creating camera x, y, r gt
        x_gt, y_gt, r_gt = batch_data[8][0][0].item(), batch_data[8][0][1].item(), batch_data[8][0][2].item()
        x_gt *= cfg.gt_ratio
        y_gt *= cfg.gt_ratio
        r_gt -= 90
        if r_gt > 180:
            r_gt = r_gt - 360
        r_gt /= 57.3

        cam_xy_loss = 0
        cam_r_loss = 0
        for i in range(min(2, len(theta_list))):
            theta_output = theta_list[i] * 57.3
            if theta_output < 0:
                theta_output += 360
            theta_output = 360 - theta_output

            theta_output -= 90
            if theta_output > 180:
                theta_output = theta_output - 360
            r_pred = theta_output / 57.3
            x_pred = deltax_list[i]
            y_pred = deltay_list[i]

            cam_xy_loss += torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
            cam_r_loss += clac_rad_distance(r_gt, r_pred)

        ####        rotation        ####

        ####        calculating reid matrix loss     ####
        # calc loss1: matrix distance loss, only using distance to create it
        # pseudo matrix

        # new_xzpred/angle with grad fn
        new_xzpred2, new_angle2 = rotate_x_y_by_point(output_dict2['xz_pred'], output_dict2['angles'],
                                                      delta_x=deltax_list[select_index],
                                                      delta_y=deltay_list[select_index],
                                                      delta_theta=theta_list[select_index])

        pseudo_matrix = torch.zeros_like(match_matrix)
        pseudo_r_matrix = torch.zeros_like(match_matrix)
        n, m = pseudo_matrix.shape
        for i in range(n):
            for j in range(m):
                # here has no grad, because new_xzpred2 has no gard and original_xz_list.item()
                pseudo_matrix[i][j] = math.sqrt(((new_xzpred2[i][0].item() - original_xz_list1[j][0].item()) ** 2 + (
                        new_xzpred2[i][1].item() - original_xz_list1[j][1].item()) ** 2))
                pseudo_r_matrix[i][j] = clac_rad_distance(new_angle2[i], original_angle_list1[j]).item()

        pseudo_matrix[pseudo_matrix < 0.0002] += 0.0001
        pseudo_r_matrix[pseudo_r_matrix < 0.0002] += 0.0001

        # create pesudo matrix to 0-1 matrix
        n, m = pseudo_matrix.shape
        matrix_dis_pseudo = 1 / pseudo_matrix
        matrix_r_pseudo = 1 / pseudo_r_matrix

        # processing distance matrix
        for i in range(n):
            line_max = matrix_dis_pseudo[i].max()
            line_min = matrix_dis_pseudo[i].min()
            size = line_max - line_min
            for j in range(m):
                matrix_dis_pseudo[i][j] = (matrix_dis_pseudo[i][j] - line_min) / size

        # processing r matrix
        for i in range(n):
            line_max = matrix_r_pseudo[i].max()
            line_min = matrix_r_pseudo[i].min()
            size = line_max - line_min
            for j in range(m):
                matrix_r_pseudo[i][j] = (matrix_r_pseudo[i][j] - line_min) / size
        matrix_r_pseudo[matrix_r_pseudo < 0.9] = 0

        # matrix gt
        id1 = batch_data[10]
        id2 = batch_data[11]
        matrix_gt = torch.zeros_like(match_matrix)
        for i in range(id2.squeeze().shape[0]):
            for j in range(id1.squeeze().shape[0]):
                if id1.squeeze()[j] == id2.squeeze()[i]:
                    matrix_gt[i][j] = 1.0
        # matrix gt

        matrix_pseudo = cfg.dis_pseudo_ratio * matrix_dis_pseudo + (1 - cfg.dis_pseudo_ratio) * matrix_r_pseudo;
        re_id_matrix_loss = torch.nn.MSELoss()(match_matrix, matrix_pseudo)
        # epoch, frame_id, matrix_gt.cpu(), match_matrix.cpu(), matrix_pseudo.cpu()
        # cfg.statistic_heatmap_cache.append([epoch, batch_data[9][0][0].item(), matrix_gt.cpu(), match_matrix.cpu(), matrix_pseudo.cpu()])

        ####        calculating reid matrix loss     ####

        total_loss = re_id_matrix_loss + cam_xy_loss + cam_r_loss

        mono_xy_record = cam_xy_loss.item()
        mono_r_record = cam_r_loss.item()
        loss_meter.update(re_id_matrix_loss.item(), mono_xy_record, mono_r_record)

        # Optim
        # with torch.autograd.detect_anomaly():
        total_loss.backward()
        optimizer.step()

    # torch.save(cfg.statistic_heatmap_cache, os.path.join(cfg.result_path, f"statistic_heatmap_cache.pth"))
    train_info = {
        'time': epoch_timer.timeit(),
        'epoch': epoch,
        're_id_loss': loss_meter.reid_loss_avg,
        'mono_xy_loss': loss_meter.mono_xy_loss_avg,
        'mono_r_loss': loss_meter.mono_r_loss_avg,
        'loss': loss_meter.total_loss,
    }
    return train_info


def _test_monoreid_cam_gt_centroid_soft_choose(data_loader, model_mono, model_reid, device, epoch, cfg):
    cfg.epoch = epoch
    if cfg.matrix_threshold > 0.5:
        matrix_threshold = cfg.matrix_threshold / 2
    else:
        matrix_threshold = cfg.matrix_threshold

    epoch_timer = Timer()

    loss_meter = AverageMeter_MonoReid()
    hit_meter = HitProbabilityMeter_Monoreid()
    hit_meter_gt = HitProbabilityMeter_Monoreid()  # using gt to aggregate the pair points, only used to analyse
    camera_meter = HitProbabilityMeter_Monoreid()
    # calculating pair point IOU
    pair_point_intersection = 0
    pair_point_union = 0

    with torch.no_grad():
        for iter, batch_data in enumerate(data_loader):

            model_mono.eval()
            model_reid.eval()

            # prepare batch data
            batch_data = [try_to(b, device) for b in batch_data]

            # forward loconet
            output_dict1 = model_mono(batch_data[4].squeeze(), batch_data[5].squeeze())
            output_dict2 = model_mono(batch_data[6].squeeze(), batch_data[7].squeeze())

            # loconet output
            original_xz_list1 = output_dict1['xz_pred']
            original_xz_list2 = output_dict2['xz_pred']
            original_angle_list1 = output_dict1['angles']
            original_angle_list2 = output_dict2['angles']

            # drawing transparent ground truth coverage image (used to visualization)
            fp_dict = cfg.fp_dict
            frame_id = int(batch_data[9][0][0].item())
            id1_used_here = batch_data[10].squeeze().tolist()
            id2_used_here = batch_data[11].squeeze().tolist()
            id1_set = set(id1_used_here)
            id2_set = set(id2_used_here)
            id_list = list(id1_set.union(id2_set))
            gt_xyangle_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id_list, cfg.gt_ratio)
            cfg.generator.get_heatmap_from_xzangle_id(gt_xyangle_list, id_list, False)
            gt_coverage = (cfg.generator.board * 0.4).int()

            # collecting view 1 results
            n = len(original_xz_list1)
            # xzangle_list of view 1
            xzangle1_list = []
            for i in range(n):
                xzangle1_list.append([original_xz_list1[i][0], original_xz_list1[i][1], original_angle_list1[i]])

            # collecting view 2 results
            n = len(original_xz_list2)
            # xzangle_list of view 2
            xzangle2_list = []
            for i in range(n):
                xzangle2_list.append([original_xz_list2[i][0], original_xz_list2[i][1], original_angle_list2[i]])

            # Adding camera and prepare id_list
            # id 1
            id1 = batch_data[10]
            # id 2
            id2 = batch_data[11]

            if iter < cfg.test_draw_fig_num:
                ######          view 1 and view 2 loconet results without rotation ######
                if_cropped = False

                # Adding camera
                xzangle1_list.append([torch.tensor(0.0, device=device), torch.tensor(0.0, device=device),
                                      torch.tensor(-90 / 57.3, device=device)])
                xzangle2_list.append([torch.tensor(0.0, device=device), torch.tensor(0.0, device=device),
                                      torch.tensor(-90 / 57.3, device=device)])

                xzangle1_list = torch.tensor(xzangle1_list)
                xzangle2_list = torch.tensor(xzangle2_list)

                # new id1 with camera
                new_id1 = id1.squeeze().tolist()
                new_id1.append(-2)
                cfg.generator.get_heatmap_from_xzangle_id(xzangle1_list, new_id1, if_cropped=if_cropped)
                view1_img = cfg.generator.board
                batch_data[9] = batch_data[9].squeeze()
                cfg.generator.board += gt_coverage
                cfg.generator.save_img(os.path.join(cfg.result_path,
                                                    f"figs/epoch{epoch}_frame{batch_data[9][0].item()}_view{batch_data[9][1].item()}.png"))

                # original view2
                # id2_tmp = id2.squeeze().tolist()
                # id2_tmp.append(-1)
                # cfg.generator.get_heatmap_from_xzangle_id(xzangle2_list, id2_tmp, if_cropped=if_cropped)
                # # cfg.generator.board += gt_coverage
                # cfg.generator.save_img(os.path.join(cfg.result_path,
                #                                     f"figs/{batch_data[9][2].item()}_{batch_data[9][3].item()}_2_{epoch}.png"))
                expand = 1.5
                xzangle2_list = [[x * expand, y * expand, anlge] for x, y, anlge in xzangle2_list]
                # new id2 with camera
                id2_tmp = id2.squeeze().tolist()
                id2_tmp.append(-1)
                cfg.generator.get_heatmap_from_xzangle_id(xzangle2_list, id2_tmp, if_cropped=if_cropped)
                # cfg.generator.board += gt_coverage
                cfg.generator.save_img(os.path.join(cfg.result_path,
                                                    f"figs/epoch{epoch}_frame{batch_data[9][2].item()}_view{batch_data[9][3].item()}_cmp.png"))

                xzangle1_list = [[x * expand, y * expand, anlge] for x, y, anlge in xzangle1_list]
                # new id2 with camera
                cfg.generator.get_heatmap_from_xzangle_id(xzangle1_list, new_id1, if_cropped=if_cropped)
                # cfg.generator.board += gt_coverage
                cfg.generator.save_img(os.path.join(cfg.result_path,
                                                    f"figs/epoch{epoch}_frame{batch_data[9][0].item()}_view{batch_data[9][1].item()}_main.png"))

            ######     evaluation re-id resutl      ########
            # resnt-50 outputs
            reid_output1 = model_reid(batch_data[2].squeeze())
            reid_output2 = model_reid(batch_data[3].squeeze())

            # selecting top3 matching pairs
            match_matrix = get_eu_distance_mtraix(reid_output1, reid_output2)
            selected_pair_list = get_top_pair_from_matrix(match_matrix, 3)

            # creating matrix gt
            matrix_gt = torch.zeros_like(match_matrix)
            for i in range(id2.squeeze().shape[0]):
                for j in range(id1.squeeze().shape[0]):
                    if id1.squeeze()[j] == id2.squeeze()[i]:
                        matrix_gt[i][j] = 1.0

            re_id_loss = torch.nn.MSELoss()(match_matrix, matrix_gt)

            # calculating matrix iou
            preidct = match_matrix.clone().detach()
            preidct[preidct > matrix_threshold] = 1.0
            preidct[preidct <= matrix_threshold] = 0.0
            gt = matrix_gt.clone().detach()

            preidct = preidct.int()
            gt = gt.int()

            intersection_num = torch.sum(preidct * gt)
            union_num = torch.sum(torch.bitwise_or(preidct, gt))

            # calculating f1 score
            reid_f1, precision, recall = calc_f1_loss_by_matrix_with_pres_recall(gt, preidct)

            ###### [SAM]        rotation here       ######

            # Using selected pair index of 2 to rotate
            theta_list, deltax_list, deltay_list = get_theta_delx_dely_from_pairs(output_dict1['angles'],
                                                                                  output_dict2['angles'],
                                                                                  output_dict1['xz_pred'],
                                                                                  output_dict2['xz_pred'],
                                                                                  selected_pair_list)

            # centroid selection strategy
            x_centroid = 0
            y_centroid = 0

            for i in range(len(deltax_list)):
                x_centroid += deltax_list[i].item()
                y_centroid += deltay_list[i].item()
            x_centroid /= len(deltay_list)
            y_centroid /= len(deltay_list)

            select_index = 0
            min_distance = 9999
            for i in range(len(theta_list)):
                distance = math.sqrt((deltax_list[i] - x_centroid) ** 2 + (deltay_list[i] - y_centroid) ** 2)
                if distance < min_distance:
                    select_index = i
                    min_distance = distance

            ######          caculating cam xy and r loss       ######

            x_gt, y_gt, r_gt = batch_data[8][0][0].item(), batch_data[8][0][1].item(), batch_data[8][0][2].item()
            x_gt *= cfg.gt_ratio
            y_gt *= cfg.gt_ratio
            r_gt -= 90
            if r_gt > 180:
                r_gt = r_gt - 360
            r_gt /= 57.3

            cam_xy_loss = 0
            cam_r_loss = 0

            # select the min distance camera to calculate the loss
            i = select_index
            theta_output = theta_list[i] * 57.3
            if theta_output < 0:
                theta_output += 360
            theta_output = 360 - theta_output

            theta_output -= 90
            if theta_output > 180:
                theta_output = theta_output - 360
            r_pred = theta_output / 57.3
            x_pred = deltax_list[i]
            y_pred = deltay_list[i]

            cam_xy_loss += torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
            cam_r_loss += clac_rad_distance(r_gt, r_pred)

            loss_meter.update(re_id_loss.item(), cam_xy_loss.item(), cam_r_loss.item())

            ######          calculating xy and r hit probability       ######

            new_xzpred2, new_angle2 = rotate_x_y_by_point(output_dict2['xz_pred'], output_dict2['angles'],
                                                          delta_x=deltax_list[select_index],
                                                          delta_y=deltay_list[select_index],
                                                          delta_theta=theta_list[select_index])

            fp_dict = cfg.fp_dict
            if len(batch_data[9].shape) >= 2:
                tmp_batch = batch_data[9].squeeze()
            else:
                tmp_batch = batch_data[9]
            frame_id = int(tmp_batch[0].item())
            # getting subject gts
            id1_used_here = batch_data[10].squeeze().tolist()
            id2_used_here = batch_data[11].squeeze().tolist()
            gt_xyangle_view1_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id1_used_here, cfg.gt_ratio)
            gt_xyangle_view2_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id2_used_here, cfg.gt_ratio)

            # aggregate the same subject from different views into BEV [[x, y, r], [x, y, r], ...]
            aggregated_points = []
            gt_points = []  # the same shape as aggregated points
            id_added_list = []
            # record person id pair added to aggregated pair
            # [[id_view1, id_view2], [id_view1, id_view2], [], []]
            id_pair_list = []

            # using by gt
            id_pair_list2 = []

            # choosing point pairs by using gt
            aggregated_points2 = []
            gt_points2 = []  # the same shape as aggregated points
            id_added_list2 = []

            # fusing the same subject by gt, here only used to analyse results, the real output of the algorithm is `aggregated_points`
            for i in range(len(id1_used_here)):
                for j in range(len(id2_used_here)):

                    if id1_used_here[i] in id2_used_here and id1_used_here[i] == id2_used_here[j]:
                        r_tmp = (original_angle_list1[i].item() + new_angle2[j].item()) / 2
                        if r_tmp > math.pi:
                            r_tmp = r_tmp - 2 * math.pi
                        if r_tmp < -math.pi:
                            r_tmp = 2 * math.pi + r_tmp

                        aggregated_points2.append([(original_xz_list1[i][0].item() + new_xzpred2[j][0].item()) / 2,
                                                   (original_xz_list1[i][1].item() + new_xzpred2[j][1].item()) / 2,
                                                   r_tmp])
                        gt_points2.append(gt_xyangle_view1_list[i])
                        id_added_list2.append(id1_used_here[i])

                        id_pair_list2.append([id1_used_here[i], id2_used_here[j]])

                    else:
                        # subject only in view 1 but not in view2
                        if id1_used_here[i] not in id_added_list2 and id1_used_here[i] not in id2_used_here:
                            id_added_list2.append(id1_used_here[i])
                            aggregated_points2.append([original_xz_list1[i][0].item(), original_xz_list1[i][1].item(),
                                                       original_angle_list1[i].item()])
                            gt_points2.append(gt_xyangle_view1_list[i])
                        # person only in view2 and not in view1
                        if id2_used_here[j] not in id_added_list2 and id2_used_here[j] not in id1_used_here:
                            id_added_list2.append(id2_used_here[j])
                            aggregated_points2.append([new_xzpred2[j][0].item(), new_xzpred2[j][1].item(),
                                                       new_angle2[j].item()])
                            gt_points2.append(gt_xyangle_view2_list[j])

            id_added_set = set(id_added_list2)
            id_need_to_add = set(id1_used_here).union(set(id2_used_here))
            # make sure that the resutl is reasonable
            assert (id_need_to_add - id_added_set) == set() and len(id_need_to_add) == len(aggregated_points2) == len(
                gt_points2)

            # creating matrix statistic unit
            pseudo_dis_matrix = torch.zeros_like(match_matrix)
            pseudo_r_matrix = torch.zeros_like(match_matrix)
            n, m = pseudo_dis_matrix.shape
            for i in range(n):
                for j in range(m):
                    # here has no grad, because new_xzpred2 has no gard and original_xz_list.item()
                    pseudo_dis_matrix[i][j] = math.sqrt(((new_xzpred2[i][0].item() - original_xz_list1[j][
                        0].item()) ** 2 + (new_xzpred2[i][1].item() - original_xz_list1[j][1].item()) ** 2))
                    pseudo_r_matrix[i][j] = clac_rad_distance(new_angle2[i], original_angle_list1[j]).item()

            reid_val_matrix = match_matrix.clone().detach()
            dis_mask = pseudo_dis_matrix <= cfg.distance_threshold
            reid_val_mask = reid_val_matrix >= matrix_threshold
            mask = torch.logical_and(dis_mask, reid_val_mask)

            n, m = mask.shape

            pair_list = []
            for i in range(n):
                for j in range(m):
                    if mask[i][j] == True:
                        pair_list.append([[j, i], pseudo_dis_matrix[i][j].item(), reid_val_matrix[i][j].item(),
                                          pseudo_r_matrix[i][j].item()])

            pair_list.sort(key=lambda x: (x[1], x[2]))

            # used in view1
            used_list1 = set()
            # used in view2
            used_list2 = set()
            pair_list_filterd = []
            for index, pair in enumerate(pair_list):
                i_tmp, j_tmp = pair[0]
                if i_tmp in used_list1 or j_tmp in used_list2:
                    pass
                else:
                    pair_list_filterd.append(pair)
                    used_list1.add(pair[0][0])
                    used_list2.add(pair[0][1])

            # making aggregation from points pair, here is the output of the algorithm
            view1_used_index = set()
            view2_used_index = set()
            # pair points
            # i : view1
            # j : view2
            try:
                for (i, j), _, _, _ in pair_list_filterd:
                    x_view1 = original_xz_list1[i][0].item()
                    y_view1 = original_xz_list1[i][1].item()
                    r_view1 = original_angle_list1[i].item()

                    x_view2 = new_xzpred2[j][0].item()
                    y_view2 = new_xzpred2[j][1].item()
                    r_view2 = new_angle2[j].item()

                    r_tmp = (r_view1 + r_view2) / 2
                    if r_tmp > math.pi:
                        r_tmp = r_tmp - 2 * math.pi
                    if r_tmp < -math.pi:
                        r_tmp = 2 * math.pi + r_tmp

                    aggregated_points.append([(x_view1 + x_view2) / 2, (y_view1 + y_view2) / 2, r_tmp])
                    gt_points.append(gt_xyangle_view1_list[i])
                    id_added_list.append(id1_used_here[i])

                    view1_used_index.add(i)
                    view2_used_index.add(j)

                    id_pair_list.append([id1_used_here[i], id2_used_here[j]])

                # cauculating single points
                view1_all_index = set([k for k in range(len(original_xz_list1))])
                view2_all_index = set([k for k in range(len(new_xzpred2))])

                view1_left_index = view1_all_index - view1_used_index
                view2_left_index = view2_all_index - view2_used_index
                # adding single points to aggregated point collections
                for index in view1_left_index:
                    aggregated_points.append([original_xz_list1[index][0].item(), original_xz_list1[index][1].item(),
                                              original_angle_list1[index].item()])
                    gt_points.append(gt_xyangle_view1_list[index])
                    id_added_list.append(id1_used_here[index])
                for index in view2_left_index:
                    aggregated_points.append(
                        [new_xzpred2[index][0].item(), new_xzpred2[index][1].item(), new_angle2[index].item()])
                    gt_points.append(gt_xyangle_view2_list[index])
                    id_added_list.append(id2_used_here[index])

            except:
                pass

            pair_point_intersection += sum([1 if pair[0] == pair[1] else 0 for pair in id_pair_list])
            pair_point_union += len(id_pair_list2)
            # iou = len(id_added_set)/ len(id_need_to_add)

            hit_meter.update(aggregated_points, gt_points, intersection_num, union_num)
            hit_meter_gt.update(aggregated_points2, gt_points2, intersection_num, union_num)

            camera_meter.update([[x_pred, y_pred, r_pred]], [[x_gt, y_gt, r_gt]], 0, 0)

            # f1 about
            hit_meter.add_f1_score(reid_f1.item())
            hit_meter.add_precision_score(precision.item())
            hit_meter.add_recall_score(recall.item())
            hit_meter_gt.add_f1_score(reid_f1.item())

            if iter < cfg.test_draw_fig_num:
                ######          visualizing view1 and view2 coverage images      ######
                # adding camera here
                output_dict2['xz_pred'].append([torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)])
                output_dict2['angles'].append(torch.tensor(-90 / 57.3, device=device))

                id_gt_pair = [[id2.squeeze()[i2], id1.squeeze()[i1]] for i2, i1, score in selected_pair_list]
                i = select_index

                new_xzpred2, new_angle2 = rotate_x_y_by_point(output_dict2['xz_pred'], output_dict2['angles'],
                                                              delta_x=deltax_list[i], delta_y=deltay_list[i],
                                                              delta_theta=theta_list[i])
                new_xzangle_list = []
                for j in range(len(new_xzpred2)):
                    new_xzangle_list.append([new_xzpred2[j][0], new_xzpred2[j][1], new_angle2[j]])
                new_xzangle_list = torch.tensor(new_xzangle_list)

                new_id2 = id2.squeeze().tolist()
                new_id2.append(-1)
                cfg.generator.get_heatmap_from_xzangle_id(new_xzangle_list, ids=new_id2, if_cropped=if_cropped)

                batch_data[8] = batch_data[8].squeeze()
                x, y, r = batch_data[8][0].item(), batch_data[8][1].item(), batch_data[8][2].item()

                theta_output = theta_list[i] * 57.3
                if theta_output < 0:
                    theta_output += 360

                theta_output = 360 - theta_output

                cfg.generator.write_font("%.2f_%.2f_%.2f_%.2f_%.2f_%.2f from %s %s %s" % (
                    x * cfg.gt_ratio, y * cfg.gt_ratio, r, deltax_list[i], deltay_list[i], theta_output,
                    batch_data[9].squeeze().tolist(), id_gt_pair[i][0].item() == id_gt_pair[i][1].item(),
                    cfg.generator.rgb_table[id2_used_here[selected_pair_list[i][0]]]
                ), (0, 0), fontsize=24)

                cfg.generator.board += view1_img
                cfg.generator.board += gt_coverage
                cfg.generator.save_img(os.path.join(cfg.result_path,
                                                    f"figs/epoch{epoch}_frame{batch_data[9][2].item()}_view{batch_data[9][3].item()}.png"))
                # cfg.generator.save_img(os.path.join(cfg.result_path,
                #                                     f"figs/frame{batch_data[9][2].item()}_view{batch_data[9][3].item()}_{3 + i}_epoch{epoch}.png"))

        test_info = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            're_id_loss': loss_meter.reid_loss_avg,
            'mono_xy_loss': loss_meter.mono_xy_loss_avg,
            'mono_r_loss': loss_meter.mono_r_loss_avg,
            'loss': loss_meter.total_loss,
            'prob': hit_meter.get_xy_r_prob_dict(),
            'iou': hit_meter.get_f1_score(),
            'prob_gt': hit_meter_gt.get_xy_r_prob_dict(),
            'pair_point_iou': pair_point_intersection / pair_point_union,
            'person_xy_mean_loss': hit_meter.get_xy_mean_error(),
            'person_r_mean_loss': hit_meter.get_r_mean_error(),
            'person_xy_mean_gt_loss': hit_meter_gt.get_xy_mean_error(),
            'person_r_mean_gt_loss': hit_meter_gt.get_r_mean_error(),
            'cam_prob': camera_meter.get_xy_r_prob_dict(),
            'precision': hit_meter.get_pre_score(),
            'recall': hit_meter.get_recall_score()
        }

        return test_info


def _test_monoreid_cam_gt_centroid_soft_choose_more_view_without_annotation_with_pred_tracking_original_way(
        view_candidate_dataset_list, model_mono, model_reid, device, epoch, cfg,
        time_extra=''.join(str(time.time()).split('.'))):
    seq_name = cfg.result_path.split('/')[-1]

    view_pair_num = len(view_candidate_dataset_list)

    cfg.epoch = epoch
    if cfg.matrix_threshold > 0.5:
        matrix_threshold = cfg.matrix_threshold / 2
    else:
        matrix_threshold = cfg.matrix_threshold

    epoch_timer = Timer()

    model_mono.eval()
    model_reid.eval()

    target_dict = {}
    view1_cos_list = []
    view2_cos_list = []
    pair_cos_list = []
    total_cos_list = []

    with torch.no_grad():
        tqdm_line = tqdm(view_candidate_dataset_list[0])
        tqdm_line.set_description(f'Test {seq_name}')
        # for iter in range(len(view_candidate_dataset_list[0])):
        for iter, _ in enumerate(tqdm_line):

            if iter >= len(tqdm_line):
                break

            cam_xy_pred_list = []
            cam_r_pred_list = []

            rotated_points_xy_list = []
            rotated_points_r_list = []
            rotated_bbox_list = []
            rotated_id_list = []

            main_points_xyangle_list = []
            main_bbox_list = []
            main_id_list = []

            top_bbox_gt_all = []
            top_id_list_all = []

            # [view1_reid_output, view2_reid_output, view3_reid_output]
            view_reid_feature_list = []

            person_num_list = []

            try:
                for view_pair_id in range(len(view_candidate_dataset_list)):

                    batch_data = view_candidate_dataset_list[view_pair_id][iter]

                    # problem dataset
                    if len(batch_data[0].shape) == 0:
                        continue

                    json_data1 = batch_data[12]
                    json_data2 = batch_data[13]

                    # prepare batch data
                    batch_data_pre_part = [try_to(b.unsqueeze(dim=0), device) for b in batch_data[:12]]
                    for i in range(12, len(batch_data)):
                        batch_data_pre_part.append(batch_data[i])
                    batch_data = batch_data_pre_part

                    top_bbox_gt_all = batch_data[20].tolist()
                    top_id_list_all = batch_data[21].tolist()

                    output_dict1 = model_mono(batch_data[4].squeeze(), batch_data[5].squeeze())
                    output_dict2 = model_mono(batch_data[6].squeeze(), batch_data[7].squeeze())

                    # loconet output
                    original_xz_list1 = output_dict1['xz_pred']
                    original_xz_list2 = output_dict2['xz_pred']
                    original_angle_list1 = output_dict1['angles']
                    original_angle_list2 = output_dict2['angles']

                    # record ground truth coverage
                    frame_id = int(batch_data[9][0][0].item())
                    view_id1 = int(batch_data[9][0][1].item())
                    view_id2 = int(batch_data[9][0][3].item())

                    ###################### save main and view visualization  ###############################
                    img_top = batch_data[22].cpu()
                    top_box_of_view1 = batch_data[18].squeeze().cpu().tolist()
                    top_box_of_view2 = batch_data[19].squeeze().cpu().tolist()

                    xz_list_tmp = [[x.cpu().item(), z.cpu().item()] for x, z, in original_xz_list1]
                    view1_pred_position_list = xz_list_tmp[::]
                    view1_top_box_position_list = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in
                                                   top_box_of_view1]
                    angle_list_tmp = [elem.item() for elem in original_angle_list1]
                    # id_list = [i + 1 for i in range(len(angle_list_tmp))]
                    id_list_view1 = batch_data[10].squeeze().cpu().tolist()
                    xz_list_tmp.append([0, 0])
                    angle_list_tmp.append(-90 / 57.3)
                    id_list_view1.append(-1)
                    xz_angle_list = [[xz_list_tmp[i][0], xz_list_tmp[i][1], angle_list_tmp[i]] for i in
                                     range(len(angle_list_tmp))]
                    box1_here = batch_data[5].squeeze().cpu().tolist()

                    xz_list_tmp = [[x.cpu().item(), z.cpu().item()] for x, z, in original_xz_list2]
                    view2_pred_position_list = xz_list_tmp[::]
                    view2_top_box_position_list = [[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in
                                                   top_box_of_view2]
                    angle_list_tmp = [elem.item() for elem in original_angle_list2]
                    id_list_view2 = batch_data[11].squeeze().cpu().tolist()
                    xz_list_tmp.append([0, 0])
                    angle_list_tmp.append(-90 / 57.3)
                    id_list_view2.append(-1)
                    xz_angle_list = [[xz_list_tmp[i][0], xz_list_tmp[i][1], angle_list_tmp[i]] for i in
                                     range(len(angle_list_tmp))]
                    box2_here = batch_data[7].squeeze().cpu().tolist()

                    ###################### save main and view visualization  ###############################

                    ##################### creating self distance matrix of each single view ##############################

                    # view 1
                    matrix_view1_1 = torch.tensor(view1_pred_position_list).reshape((-1, 1, 2))
                    matrix_view1_2 = torch.tensor(view1_pred_position_list).reshape((1, -1, 2))
                    self_distance_matrix_of_view1 = torch.sqrt(
                        torch.sum((matrix_view1_1 - matrix_view1_2) ** 2, axis=-1) + 1e-6)
                    self_distance_matrix_of_view1_reg = self_distance_matrix_of_view1 / (
                            torch.max(self_distance_matrix_of_view1).item() + 1e-6)

                    matrix_view1_1_top = torch.tensor(view1_top_box_position_list).reshape((-1, 1, 2))
                    matrix_view1_2_top = torch.tensor(view1_top_box_position_list).reshape((1, -1, 2))
                    self_distance_matrix_of_view1_top = torch.sqrt(
                        torch.sum((matrix_view1_1_top - matrix_view1_2_top) ** 2, axis=-1) + 1e-6)
                    self_distance_matrix_of_view1_top_reg = self_distance_matrix_of_view1_top / (
                            torch.max(self_distance_matrix_of_view1_top).item() + 1e-6)
                    cos_similarity_view1 = calc_cos_similarity_from_two_matrix(self_distance_matrix_of_view1_reg,
                                                                               self_distance_matrix_of_view1_top_reg)
                    view1_cos_list.append(cos_similarity_view1.item())

                    # view 2
                    matrix_view2_1 = torch.tensor(view2_pred_position_list).reshape((-1, 1, 2))
                    matrix_view2_2 = torch.tensor(view2_pred_position_list).reshape((1, -1, 2))
                    self_distance_matrix_of_view2 = torch.sqrt(
                        torch.sum((matrix_view2_1 - matrix_view2_2) ** 2, axis=-1) + 1e-6)
                    self_distance_matrix_of_view2_reg = self_distance_matrix_of_view2 / (
                            torch.max(self_distance_matrix_of_view2).item() + 1e-6)

                    matrix_view2_1_top = torch.tensor(view2_top_box_position_list).reshape((-1, 1, 2))
                    matrix_view2_2_top = torch.tensor(view2_top_box_position_list).reshape((1, -1, 2))
                    self_distance_matrix_of_view2_top = torch.sqrt(
                        torch.sum((matrix_view2_1_top - matrix_view2_2_top) ** 2, axis=-1) + 1e-6)
                    self_distance_matrix_of_view2_top_reg = self_distance_matrix_of_view2_top / (
                            torch.max(self_distance_matrix_of_view2_top).item() + 1e-6)
                    cos_similarity_view2 = calc_cos_similarity_from_two_matrix(self_distance_matrix_of_view2_reg,
                                                                               self_distance_matrix_of_view2_top_reg)
                    view2_cos_list.append(cos_similarity_view2.item())

                    ##################### creating self distance matrix of each single view ##############################

                    bbox1 = [[view_id1, bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]] for bbox_score in
                             batch_data[5].squeeze().cpu().tolist()]
                    bbox2 = [[view_id2, bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]] for bbox_score in
                             batch_data[7].squeeze().cpu().tolist()]

                    # record ground truth coverage

                    # create xzangle_list() from xz_list() and angle_list()
                    # view 1
                    n = len(original_xz_list1)
                    # xzangle_list of view 1
                    xzangle1_list = []
                    for i in range(n):
                        xzangle1_list.append(
                            [original_xz_list1[i][0], original_xz_list1[i][1], original_angle_list1[i]])
                    # xzangle1_list = torch.tensor(xzangle1_list)

                    # view 2
                    n = len(original_xz_list2)
                    # xzangle_list of view 2
                    xzangle2_list = []
                    for i in range(n):
                        xzangle2_list.append(
                            [original_xz_list2[i][0], original_xz_list2[i][1], original_angle_list2[i]])
                    # xzangle2_list = torch.tensor(xzangle2_list)

                    # Adding camera and prepare id_list

                    if len(main_points_xyangle_list) == 0:
                        main_points_xyangle_list = xzangle1_list
                        main_bbox_list = bbox1

                    # recover id_list
                    id_list_view1 = id_list_view1[:-1]
                    id_list_view2 = id_list_view2[:-1]

                    ######     evaluation re-id resutl      ########
                    # reid_test
                    reid_output1 = model_reid(batch_data[2].squeeze())
                    reid_output2 = model_reid(batch_data[3].squeeze())

                    # record_feature
                    if len(view_reid_feature_list) == 0:
                        view_reid_feature_list.append(reid_output1)
                    view_reid_feature_list.append(reid_output2)

                    if len(person_num_list) == 0:
                        person_num_list.append(reid_output1.shape[0])
                    person_num_list.append(reid_output2.shape[0])

                    # match_matrix = get_cos_distance_matrix(reid_output1, reid_output2)
                    match_matrix = get_eu_distance_mtraix(reid_output1, reid_output2)
                    selected_pair_list = get_top_pair_from_matrix(match_matrix, 3)

                    ######     evaluation re-id resutl      ########

                    ######          rotation here       ######

                    # mono about
                    # Using selected pair index of 2 to rotate

                    theta_list, deltax_list, deltay_list = get_theta_delx_dely_from_pairs(output_dict1['angles'],
                                                                                          output_dict2['angles'],
                                                                                          output_dict1['xz_pred'],
                                                                                          output_dict2['xz_pred'],
                                                                                          selected_pair_list)

                    # creating centroid
                    x_centroid = 0
                    y_centroid = 0

                    for i in range(len(deltax_list)):
                        x_centroid += deltax_list[i].item()
                        y_centroid += deltay_list[i].item()
                    x_centroid /= len(deltay_list)
                    y_centroid /= len(deltay_list)

                    select_index = 0
                    min_distance = 9999
                    for i in range(len(theta_list)):
                        distance = math.sqrt((deltax_list[i] - x_centroid) ** 2 + (deltay_list[i] - y_centroid) ** 2)
                        if distance < min_distance:
                            select_index = i
                            min_distance = distance

                    ######          rotation here       ######

                    ######          caculating cam xy and r loss       ######

                    i = select_index
                    theta_output = theta_list[i] * 57.3
                    if theta_output < 0:
                        theta_output += 360
                    theta_output = 360 - theta_output

                    theta_output -= 90
                    if theta_output > 180:
                        theta_output = theta_output - 360
                    r_pred = theta_output / 57.3
                    x_pred = deltax_list[i]
                    y_pred = deltay_list[i]
                    cam_xy_pred_list.append([x_pred.item(), y_pred.item()])
                    cam_r_pred_list.append(r_pred.item())

                    ######          caculating xy and r loss       ######

                    ######          caculating xy and r Hit probability       ######

                    new_xzpred2, new_angle2 = rotate_x_y_by_point(output_dict2['xz_pred'], output_dict2['angles'],
                                                                  delta_x=deltax_list[select_index],
                                                                  delta_y=deltay_list[select_index],
                                                                  delta_theta=theta_list[select_index])

                    if len(batch_data[9].shape) >= 2:
                        tmp_batch = batch_data[9].squeeze()
                    else:
                        tmp_batch = batch_data[9]
                    frame_id = int(tmp_batch[0].item())

                    # record rotated information
                    rotated_points_xy_list.append(new_xzpred2.tolist())
                    rotated_points_r_list.append(new_angle2.tolist())
                    rotated_bbox_list.append(bbox2)
                    rotated_id_list.append(id_list_view2)
                    main_id_list = id_list_view1
                    # record rotated information

                    # gt aggregation
                    # [[x, y, r], [x, y, r], ...]
                    aggregated_points = []
                    bbox_list = []
                    aggregated_id_list = []
                    # using by gt

                    # gt select point pairs

                    # creating matrix statistic unit without gt
                    pseudo_dis_matrix = torch.zeros_like(match_matrix)
                    pseudo_r_matrix = torch.zeros_like(match_matrix)
                    n, m = pseudo_dis_matrix.shape
                    for i in range(n):
                        for j in range(m):
                            # here has no grad, because new_xzpred2 has no gard and original_xz_list.item()
                            pseudo_dis_matrix[i][j] = math.sqrt(((new_xzpred2[i][0].item() - original_xz_list1[j][
                                0].item()) ** 2 + (new_xzpred2[i][1].item() - original_xz_list1[j][1].item()) ** 2))
                            pseudo_r_matrix[i][j] = clac_rad_distance(new_angle2[i], original_angle_list1[j]).item()

                    reid_val_matrix = match_matrix.clone().detach()
                    # dis_mask = pseudo_dis_matrix <= cfg.distance_threshold
                    reid_val_mask = reid_val_matrix >= matrix_threshold
                    # mask = torch.logical_and(dis_mask, reid_val_mask)
                    mask = reid_val_mask

                    n, m = mask.shape

                    pair_list = []
                    for i in range(n):
                        for j in range(m):
                            if mask[i][j] == True:
                                pair_list.append([[j, i], pseudo_dis_matrix[i][j].item(), reid_val_matrix[i][j].item(),
                                                  pseudo_r_matrix[i][j].item()])

                    pair_list.sort(key=lambda x: (x[1], x[2]))

                    # used in view1
                    used_list1 = set()
                    # used in view2
                    used_list2 = set()
                    pair_list_filterd = []
                    for index, pair in enumerate(pair_list):
                        i_tmp, j_tmp = pair[0]
                        if i_tmp in used_list1 or j_tmp in used_list2:
                            pass
                        else:
                            pair_list_filterd.append(pair)
                            used_list1.add(pair[0][0])
                            used_list2.add(pair[0][1])
                    # creating matrix statistic unit

                    # making aggregation from points pair

                    view1_used_index = set()
                    view2_used_index = set()
                    # pair points
                    # i : view1
                    # j : view2
                    for (i, j), _, _, _ in pair_list_filterd:
                        x_view1 = original_xz_list1[i][0].item()
                        y_view1 = original_xz_list1[i][1].item()
                        r_view1 = original_angle_list1[i].item()

                        x_view2 = new_xzpred2[j][0].item()
                        y_view2 = new_xzpred2[j][1].item()
                        r_view2 = new_angle2[j].item()

                        r_tmp = r_view1

                        # aggregated_points.append([(x_view1 + x_view2) / 2, (y_view1 + y_view2) / 2, r_tmp])
                        aggregated_points.append([x_view1, y_view1, r_tmp])
                        box_tmp = [bbox1[i], bbox2[j]]
                        bbox_list.append(box_tmp)
                        aggregated_id_list.append(id_list_view1[i])

                        view1_used_index.add(i)
                        view2_used_index.add(j)

                    # cauculating single points
                    view1_all_index = set([k for k in range(len(original_xz_list1))])
                    view2_all_index = set([k for k in range(len(new_xzpred2))])

                    view1_left_index = view1_all_index - view1_used_index
                    view2_left_index = view2_all_index - view2_used_index
                    # adding single points to aggregated point collections
                    for index in view1_left_index:
                        aggregated_points.append(
                            [original_xz_list1[index][0].item(), original_xz_list1[index][1].item(),
                             original_angle_list1[index].item()])
                        bbox_list.append(bbox1[index])
                        aggregated_id_list.append(id_list_view1[index])
                    for index in view2_left_index:
                        aggregated_points.append(
                            [new_xzpred2[index][0].item(), new_angle2[index].item(), new_angle2[index].item()])
                        bbox_list.append(bbox2[index])
                        aggregated_id_list.append(id_list_view2[index])

                    ################## statistic pair cosine similarity with gt    ###########################
                    top_box_gt_all = batch_data[20].tolist()
                    top_box_id_all = batch_data[21].tolist()
                    aggregated_top_gt_position = []
                    aggregated_top_bbox_gt_list = []
                    for id in aggregated_id_list:
                        x1, y1, x2, y2 = top_box_gt_all[top_box_id_all.index(id)]
                        aggregated_top_gt_position.append([(x1 + x2) / 2, (y1 + y2) / 2])
                        aggregated_top_bbox_gt_list.append([x1, y1, x2, y2])
                    aggregated_top_pred_position = [[x, y] for x, y, r in aggregated_points]

                    matrix_aggregated_1 = torch.tensor(aggregated_top_pred_position).reshape((-1, 1, 2))
                    matrix_aggregated_2 = torch.tensor(aggregated_top_pred_position).reshape((1, -1, 2))
                    self_distance_matrix_of_aggregation = torch.sqrt(
                        torch.sum((matrix_aggregated_1 - matrix_aggregated_2) ** 2, axis=-1) + 1e-6)
                    self_distance_matrix_of_aggregation_reg = self_distance_matrix_of_aggregation / (
                            torch.max(self_distance_matrix_of_aggregation).item() + 1e-6)

                    matrix_aggregated_1_top = torch.tensor(aggregated_top_gt_position).reshape((-1, 1, 2))
                    matrix_aggregated_2_top = torch.tensor(aggregated_top_gt_position).reshape((1, -1, 2))
                    self_distance_matrix_of_aggregated_top = torch.sqrt(
                        torch.sum((matrix_aggregated_1_top - matrix_aggregated_2_top) ** 2, axis=-1) + 1e-6)
                    self_distance_matrix_of_aggregated_top_reg = self_distance_matrix_of_aggregated_top / (
                            torch.max(self_distance_matrix_of_aggregated_top).item() + 1e-6)
                    cos_similarity_aggregation = calc_cos_similarity_from_two_matrix(
                        self_distance_matrix_of_aggregated_top_reg, self_distance_matrix_of_aggregation_reg)
                    pair_cos_list.append(cos_similarity_aggregation.item())

                    with open(os.path.join(cfg.result_path, 'cos_log.txt'), 'a') as f:
                        f.write(
                            f'Frame[{frame_id}] View({view_id1},{view_id2}): {round(cos_similarity_view1.item() * 100, 3)}, {round(cos_similarity_view2.item() * 100, 3)}, {round(cos_similarity_aggregation.item() * 100, 3)}\n')

                    ################## statistic pair cosine similarity with gt    ###########################

                # 4 (main view is not here)
                comp_view_num = len(view_candidate_dataset_list)

                # --------- aggregated these points ------------

                # preparing total data list
                # id
                # xy and r
                xyr_total_list = copy.copy(main_points_xyangle_list)
                bbox_total_list = main_bbox_list
                top_pred_xy_total_list = [[xyr[0].item(), xyr[1].item()] for xyr in xyr_total_list]
                top_pred_r_total_list = [xyr[2].item() for xyr in xyr_total_list]
                for i in range(len(rotated_points_xy_list)):
                    top_pred_xy_total_list.extend(rotated_points_xy_list[i])
                    top_pred_r_total_list.extend(rotated_points_r_list[i])
                    bbox_total_list.extend(rotated_bbox_list[i])
                top_pred_id_total_list = main_id_list
                for id_list in rotated_id_list:
                    for id in id_list:
                        top_pred_id_total_list.append(id)

                person_interval_list = []
                tmp_sum = 0
                for person in person_num_list:
                    person_interval_list.append(tmp_sum)
                    tmp_sum += person
                person_interval_list.append(tmp_sum)

                # preparing total data list

                # creating reid matrix
                total_feature_cat = torch.cat(view_reid_feature_list, dim=0)
                match_total_matrix = get_eu_distance_mtraix(total_feature_cat, total_feature_cat)

                # creating distance matrix
                distance_total_matrix = torch.zeros_like(match_total_matrix)
                r_total_matrix = torch.zeros_like(match_total_matrix)

                # creating r matrix

                # creating gt matrix, distance matrix and r matrix

                dis_mask = distance_total_matrix <= cfg.distance_threshold
                reid_val_mask = match_total_matrix >= matrix_threshold
                # mask = torch.logical_and(dis_mask, reid_val_mask)
                mask = reid_val_mask
                # Simplify the matrix by symmetry
                mask_up = torch.triu(mask, 1)
                # remove self matrix mask
                person_num_sum = 0
                for person_num in person_num_list:
                    mask_up[person_num_sum:person_num_sum + person_num,
                    person_num_sum:person_num_sum + person_num] = False
                    person_num_sum += person_num

                # creating gt matrix, distance matrix and r matrix

                # aggregate points
                pair_list = []
                # [[index, score]]
                id_total_list = [i for i in range(person_num_sum)]
                union_find = UnionFind(len(id_total_list))
                pair_dict = {i: [] for i in range(len(id_total_list))}

                for i in range(len(id_total_list)):
                    for j in range(len(id_total_list)):
                        if mask_up[i][j] == True:
                            pair_dict[i].append([j, match_total_matrix[i][j]])
                            union_find.union(i, j)

                union_collection_dict = {i: [] for i in range(len(id_total_list))}
                for i in range(len(id_total_list)):
                    union_collection_dict[union_find.find(i)].append(i)
                # filter the union_collection
                aggregated_list = []
                for key, val in union_collection_dict.items():
                    if len(val) > 1:
                        aggregated_list.append(val)

                pop_index_list = []
                for index, aggregated_sub_list in enumerate(aggregated_list):
                    split_list = []
                    person_interval_counter = [[] for i in range(len(person_interval_list) - 1)]
                    for i in range(len(person_interval_list) - 1):
                        for j in range(len(aggregated_sub_list)):
                            if aggregated_sub_list[j] >= person_interval_list[i] and aggregated_sub_list[j] < \
                                    person_interval_list[i + 1]:
                                person_interval_counter[i].append(aggregated_sub_list[j])
                    person_counter_max = max(
                        [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                    if person_counter_max > 1:
                        # split
                        while person_counter_max > 1:
                            for i in range(len(person_interval_counter)):
                                tmp_split = []
                                if len(person_interval_counter[i]) > 0:
                                    pivot = person_interval_counter[i][0]
                                    tmp_split.append(person_interval_counter[i].pop(0))
                                    for j in range(i + 1, len(person_interval_counter)):
                                        score_list = [match_total_matrix[pivot][elem_index] for elem_index in
                                                      person_interval_counter[j]]
                                        if len(score_list) == 0:
                                            continue
                                        if max(score_list) < cfg.matrix_threshold:
                                            break
                                        else:
                                            max_index = score_list.index(max(score_list))
                                            tmp_split.append(person_interval_counter[j].pop(max_index))
                                if tmp_split != []:
                                    aggregated_list.append(tmp_split)

                            person_counter_max = max(
                                [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                            # collecting splited list
                        # collecting remaining list
                        pop_index_list.append(index)
                        remaining_list = []
                        for i in range(len(person_interval_counter)):
                            for j in range(len(person_interval_counter[i])):
                                remaining_list.append(person_interval_counter[i][j])
                        if remaining_list != []:
                            aggregated_list.append(remaining_list)



                    else:
                        pass

                aggregated_list_new = []
                for i in range(len(aggregated_list)):
                    if i in pop_index_list:
                        continue
                    else:
                        aggregated_list_new.append(aggregated_list[i])

                aggregated_list = aggregated_list_new

                index_used_set = set()
                total_index_set = set([i for i in range(len(id_total_list))])

                # mean choosing
                # final_xy_total_list = []
                # final_r_total_list = []

                # centroid 1 choosing
                final_only_one_xy_total_list = []
                final_only_one_r_total_list = []
                final_only_one_bbox_total_list = []
                final_only_one_feature_total_list = []

                final_only_one_top_bbox_total_list = []
                final_only_one_top_position_total_list = []

                real_id_list = []

                for aggregated_sub_list in aggregated_list:
                    x_sum = 0
                    y_sum = 0
                    r_sum = 0
                    # used to caculate centroid
                    x_cache = []
                    y_cache = []
                    r_cache = []

                    id_cache = []

                    for index_sub in range(len(aggregated_sub_list)):
                        index_used_set.add(aggregated_sub_list[index_sub])

                        x_cache.append(top_pred_xy_total_list[index_sub][0])
                        y_cache.append(top_pred_xy_total_list[index_sub][1])
                        r_cache.append(top_pred_r_total_list[index_sub])

                        id_cache.append(top_pred_id_total_list[aggregated_sub_list[index_sub]])

                        x_sum += top_pred_xy_total_list[index_sub][0]
                        y_sum += top_pred_xy_total_list[index_sub][1]
                        r_sum += top_pred_r_total_list[index_sub]
                    x_mean = x_sum / len(aggregated_sub_list)
                    y_mean = y_sum / len(aggregated_sub_list)
                    r_mean = r_sum / len(aggregated_sub_list)

                    # centroid about
                    centroid_x = x_mean
                    centroid_y = y_mean
                    to_centroid_distance_list = [math.sqrt((x_cache[i] - x_mean) ** 2 + (y_cache[i] - y_mean) ** 2) for
                                                 i in range(len(x_cache))]
                    rank_list = torch.argsort(torch.tensor(to_centroid_distance_list)).tolist()
                    rank1_index = rank_list.index(0)
                    # rank2_index = rank_list.index(1)

                    total1_index = aggregated_sub_list[rank1_index]
                    # total2_index = aggregated_sub_list[rank2_index]

                    # TODO change index to main viwe1
                    total1_index = aggregated_sub_list[0]
                    real_id_list.append(id_cache)

                    final_only_one_xy_total_list.append(top_pred_xy_total_list[total1_index])
                    final_only_one_r_total_list.append(top_pred_r_total_list[total1_index])
                    final_only_one_bbox_total_list.append(
                        [bbox_total_list[aggregated_sub_list[i]] for i in range(len(aggregated_sub_list))])
                    final_only_one_feature_total_list.append(total_feature_cat[total1_index])
                    corresponding_bbox = top_bbox_gt_all[top_id_list_all.index(top_pred_id_total_list[total1_index])]

                    final_only_one_top_bbox_total_list.append(corresponding_bbox)
                    final_only_one_top_position_total_list.append([(corresponding_bbox[0] + corresponding_bbox[2]) / 2,
                                                                   (corresponding_bbox[1] + corresponding_bbox[3]) / 2])

                # aggregated more view points here

                # aggregated single points here
                single_index_list = sorted(list(total_index_set - index_used_set))

                for single_index in single_index_list:
                    id_used_here = id_total_list[single_index]
                    xy_to_be_added = [top_pred_xy_total_list[single_index][0], top_pred_xy_total_list[single_index][1]]
                    r_to_be_added = top_pred_r_total_list[single_index]

                    # final_xy_total_list.append([top_pred_xy_total_list[single_index][0], top_pred_xy_total_list[single_index][1]])
                    final_only_one_xy_total_list.append(xy_to_be_added)
                    # final_two_mean_xy_total_list.append(xy_to_be_added)
                    # final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])

                    # final_r_total_list.append(top_pred_r_total_list[single_index])
                    final_only_one_r_total_list.append(r_to_be_added)
                    final_only_one_bbox_total_list.append([bbox_total_list[single_index]])
                    final_only_one_feature_total_list.append(total_feature_cat[single_index])

                    real_id_list.append([top_pred_id_total_list[single_index]])

                    corresponding_bbox = top_bbox_gt_all[top_id_list_all.index(top_pred_id_total_list[single_index])]
                    final_only_one_top_bbox_total_list.append(corresponding_bbox)
                    final_only_one_top_position_total_list.append([(corresponding_bbox[0] + corresponding_bbox[2]) / 2,
                                                                   (corresponding_bbox[1] + corresponding_bbox[3]) / 2])

                # update statistic

                ############################             calc multi view aggregation self distance result          ######################
                # calc target
                frame_id_str = '0' * (5 - len(str(frame_id))) + str(frame_id)
                frame_str = '0' * (5 - len(str(frame_id))) + str(frame_id)
                counter = Counter(top_pred_id_total_list)
                sum_counter = len(counter)
                target = 0
                for key, num in counter.items():
                    tmp = [key] * num
                    if tmp in real_id_list:
                        target += 1
                target_ratio = target / sum_counter

                with open(os.path.join(cfg.result_path, 'target_ratio.txt'), 'a') as f:
                    f.write(f'Frame {frame_str}: {target_ratio}\n')

                target_dict[frame_str] = target_ratio
                torch.save(target_dict, os.path.join(cfg.result_path, 'target_dict.pth'))

                matrix_aggregated_1 = torch.tensor(final_only_one_xy_total_list).reshape((-1, 1, 2))
                matrix_aggregated_2 = torch.tensor(final_only_one_xy_total_list).reshape((1, -1, 2))
                self_distance_matrix_of_aggregation = torch.sqrt(
                    torch.sum((matrix_aggregated_1 - matrix_aggregated_2) ** 2, axis=-1) + 1e-6)
                self_distance_matrix_of_aggregation_reg = self_distance_matrix_of_aggregation / (
                        torch.max(self_distance_matrix_of_aggregation).item() + 1e-6)

                matrix_aggregated_1_top = torch.tensor(final_only_one_top_position_total_list).reshape((-1, 1, 2))
                matrix_aggregated_2_top = torch.tensor(final_only_one_top_position_total_list).reshape((1, -1, 2))
                self_distance_matrix_of_aggregated_top = torch.sqrt(
                    torch.sum((matrix_aggregated_1_top - matrix_aggregated_2_top) ** 2, axis=-1) + 1e-6)
                self_distance_matrix_of_aggregated_top_reg = self_distance_matrix_of_aggregated_top / (
                        torch.max(self_distance_matrix_of_aggregated_top).item() + 1e-6)
                cos_similarity_aggregation_more_view = calc_cos_similarity_from_two_matrix(
                    self_distance_matrix_of_aggregated_top_reg,
                    self_distance_matrix_of_aggregation_reg)
                total_cos_list.append(cos_similarity_aggregation_more_view.item())
                saved_cos_list = [view1_cos_list, view2_cos_list, pair_cos_list, total_cos_list]
                torch.save(saved_cos_list, os.path.join(cfg.result_path, 'cos_list.path'))

                if len(total_cos_list) > 0 and len(view1_cos_list) > 0 and len(pair_cos_list) > 0:
                    view1_avg = sum(view1_cos_list) / len(view1_cos_list)
                    view2_avg = sum(view2_cos_list) / len(view2_cos_list)
                    pair_avg = sum(pair_cos_list) / len(pair_cos_list)
                    agg_avg = sum(total_cos_list) / len(total_cos_list)

                    tqdm_line.set_postfix({'COS': round(agg_avg * 100, 2)})

                    with open(os.path.join(cfg.result_path, 'cos_log.txt'), 'a') as f:
                        f.write(
                            f'Frame[{frame_id}] Avg Result: view1: {view1_avg}, view2: {view2_avg}, pair:{pair_avg}, view-{comp_view_num + 1}:{agg_avg}\n')

                with open(os.path.join(cfg.result_path, 'cos_log.txt'), 'a') as f:
                    f.write(
                        f'Frame[{frame_id}] View({",".join([str(i) for i in range(1, 1 + 1 + comp_view_num)])}): {round(cos_similarity_view1.item() * 100, 3)}, {round(cos_similarity_view2.item() * 100, 3)}, {round(cos_similarity_aggregation.item() * 100, 3)}\n')


            except Exception as e:
                continue

        test_info = {
            'view_avg': agg_avg,
            'pair_avg': pair_avg,
            'single_view_avg': (view1_avg + view2_avg) / 2,
            'view_num': comp_view_num + 1
        }

        return test_info


def _test_monoreid_cam_gt_centroid_soft_choose_more_view(view_candidate_dataset_list, model_mono, model_reid, device,
                                                         epoch, cfg):
    # Input augmented data,
    # train vindicator
    # save vindicator and return

    view_pair_num = len(view_candidate_dataset_list)

    cfg.epoch = epoch
    if cfg.matrix_threshold > 0.5:
        matrix_threshold = cfg.matrix_threshold / 2
    else:
        matrix_threshold = cfg.matrix_threshold

    epoch_timer = Timer()

    loss_meter = AverageMeter_MonoReid()
    hit_meter = HitProbabilityMeter_Monoreid()
    hit_meter_gt = HitProbabilityMeter_Monoreid()

    hit_total_mean_meter = HitProbabilityMeter_Monoreid()
    hit_total_one_meter = HitProbabilityMeter_Monoreid()
    hit_total_two_meter = HitProbabilityMeter_Monoreid()

    camera_meter = HitProbabilityMeter_Monoreid()
    total_f1_sum = 0
    # calculating pair point IOU
    pair_point_intersection = 0
    pair_point_union = 0
    # [[frame_str, person_prob, id_prob, repeat_tag]]
    prob_statistic_cache = []

    model_mono.eval()
    model_reid.eval()

    with torch.no_grad():
        for iter in range(len(view_candidate_dataset_list[0])):
            if cfg.dataset_name == 'Tracking1000' and iter == 768:
                continue
            print(f'processing test case {iter}')
            draw_gt_id_xyangle_dict_total = {}
            cam_xy_pred_list = []
            cam_xy_gt_list = []
            cam_r_pred_list = []
            cam_r_gt_list = []

            rotated_points_xy_list = []
            rotated_points_r_list = []
            rotated_points_id_list = []
            rotated_bbox_list = []

            main_points_xyangle_list = []
            main_points_id_list = []
            main_bbox_list = []

            view_reid_feature_list = []

            person_num_list = []

            for view_pair_id in range(len(view_candidate_dataset_list)):

                batch_data = view_candidate_dataset_list[view_pair_id][iter]

                # prepare batch data
                batch_data = [try_to(b.unsqueeze(dim=0), device) for b in batch_data]

                output_dict1 = model_mono(batch_data[4].squeeze(), batch_data[5].squeeze())
                output_dict2 = model_mono(batch_data[6].squeeze(), batch_data[7].squeeze())

                # monoloco output
                original_xz_list1 = output_dict1['xz_pred']
                original_xz_list2 = output_dict2['xz_pred']
                original_angle_list1 = output_dict1['angles']
                original_angle_list2 = output_dict2['angles']

                # record ground truth coverage
                fp_dict = cfg.fp_dict
                frame_id = int(batch_data[9][0][0].item())
                id1_used_here = batch_data[10].squeeze().tolist()
                id2_used_here = batch_data[11].squeeze().tolist()
                id1_set = set(id1_used_here)
                id2_set = set(id2_used_here)
                id_list = list(id1_set.union(id2_set))
                gt_xyangle_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id_list, cfg.gt_ratio)
                # record id xyangle_gt dict used to draw figure
                for index_draw_gt, id_draw_gt in enumerate(id_list):
                    draw_gt_id_xyangle_dict_total[id_draw_gt] = gt_xyangle_list[index_draw_gt]
                    # record id xyangle_gt dict used to draw figure
                # record ground truth coverage

                # record ground truth coverage
                frame_id = int(batch_data[9][0][0].item())
                view_id1 = int(batch_data[9][0][1].item())
                view_id2 = int(batch_data[9][0][3].item())

                try:
                    bbox1 = [[view_id1, bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]] for bbox_score in
                             batch_data[5].squeeze().cpu().tolist()]
                    bbox2 = [[view_id2, bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]] for bbox_score in
                             batch_data[7].squeeze().cpu().tolist()]
                except:
                    continue

                # creating xzangle_list() from xz_list() and angle_list()
                # view 1
                n = len(original_xz_list1)
                # xzangle_list of view 1
                xzangle1_list = []
                for i in range(n):
                    xzangle1_list.append([original_xz_list1[i][0], original_xz_list1[i][1], original_angle_list1[i]])
                # xzangle1_list = torch.tensor(xzangle1_list)

                # view 2
                n = len(original_xz_list2)
                # xzangle_list of view 2
                xzangle2_list = []
                for i in range(n):
                    xzangle2_list.append([original_xz_list2[i][0], original_xz_list2[i][1], original_angle_list2[i]])
                # xzangle2_list = torch.tensor(xzangle2_list)

                # Adding camera and prepare id_list

                # id 1
                id1 = batch_data[10]
                # id 2
                id2 = batch_data[11]

                if len(main_points_xyangle_list) == 0:
                    main_points_xyangle_list = xzangle1_list
                    main_points_id_list = id1.squeeze().tolist()
                    main_bbox_list = bbox1

                ######     evaluation re-id resutl      ########
                # reid_test
                reid_output1 = model_reid(batch_data[2].squeeze())
                reid_output2 = model_reid(batch_data[3].squeeze())

                # record_feature
                if len(view_reid_feature_list) == 0:
                    view_reid_feature_list.append(reid_output1)
                view_reid_feature_list.append(reid_output2)

                if len(person_num_list) == 0:
                    person_num_list.append(reid_output1.shape[0])
                person_num_list.append(reid_output2.shape[0])

                # match_matrix = get_cos_distance_matrix(reid_output1, reid_output2)
                match_matrix = get_eu_distance_mtraix(reid_output1, reid_output2)
                selected_pair_list = get_top_pair_from_matrix(match_matrix, 3)

                # create matrix gt
                matrix_gt = torch.zeros_like(match_matrix)
                for i in range(id2.squeeze().shape[0]):
                    for j in range(id1.squeeze().shape[0]):
                        if id1.squeeze()[j] == id2.squeeze()[i]:
                            matrix_gt[i][j] = 1.0

                re_id_loss = torch.nn.MSELoss()(match_matrix, matrix_gt)

                # calculating matrix iou
                preidct = match_matrix.clone().detach()
                preidct[preidct > matrix_threshold] = 1.0
                preidct[preidct <= matrix_threshold] = 0.0
                gt = matrix_gt.clone().detach()

                preidct = preidct.int()
                gt = gt.int()

                intersection_num = torch.sum(preidct * gt)
                union_num = torch.sum(torch.bitwise_or(preidct, gt))

                # caculating f1 score
                reid_f1 = calc_f1_loss_by_matrix(gt, preidct)

                # calculating matrix iou

                ######     evaluation re-id resutl      ########

                ###### [SAM]         rotation here       ######

                # mono about
                # Using selected pair index of 2 to rotate

                theta_list, deltax_list, deltay_list = get_theta_delx_dely_from_pairs(output_dict1['angles'],
                                                                                      output_dict2['angles'],
                                                                                      output_dict1['xz_pred'],
                                                                                      output_dict2['xz_pred'],
                                                                                      selected_pair_list)

                # centroid selection strategy
                x_centroid = 0
                y_centroid = 0

                for i in range(len(deltax_list)):
                    x_centroid += deltax_list[i].item()
                    y_centroid += deltay_list[i].item()
                x_centroid /= len(deltay_list)
                y_centroid /= len(deltay_list)

                select_index = 0
                min_distance = 9999
                for i in range(len(theta_list)):
                    distance = math.sqrt((deltax_list[i] - x_centroid) ** 2 + (deltay_list[i] - y_centroid) ** 2)
                    if distance < min_distance:
                        select_index = i
                        min_distance = distance

                ######          rotation here       ######

                ######          caculating cam xy and r loss       ######

                x_gt, y_gt, r_gt = batch_data[8][0][0].item(), batch_data[8][0][1].item(), batch_data[8][0][2].item()
                x_gt *= cfg.gt_ratio
                y_gt *= cfg.gt_ratio
                r_gt -= 90
                if r_gt > 180:
                    r_gt = r_gt - 360
                r_gt /= 57.3

                cam_xy_loss = 0
                cam_r_loss = 0

                # selecting the min centroid distance camera to calculate the loss
                i = select_index
                theta_output = theta_list[i] * 57.3
                if theta_output < 0:
                    theta_output += 360
                theta_output = 360 - theta_output

                theta_output -= 90
                if theta_output > 180:
                    theta_output = theta_output - 360
                r_pred = theta_output / 57.3
                x_pred = deltax_list[i]
                y_pred = deltay_list[i]

                cam_xy_loss += torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
                cam_r_loss += clac_rad_distance(r_gt, r_pred)

                loss_meter.update(re_id_loss.item(), cam_xy_loss.item(), cam_r_loss.item())

                # record camera predection
                cam_xy_pred_list.append([x_pred.item(), y_pred.item()])
                cam_xy_gt_list.append([x_gt, y_gt])
                cam_r_pred_list.append(r_pred.item())
                cam_r_gt_list.append(r_gt)
                ######          caculating xy and r loss       ######

                ######          caculating xy and r Hit probability       ######

                new_xzpred2, new_angle2 = rotate_x_y_by_point(output_dict2['xz_pred'], output_dict2['angles'],
                                                              delta_x=deltax_list[select_index],
                                                              delta_y=deltay_list[select_index],
                                                              delta_theta=theta_list[select_index])

                fp_dict = cfg.fp_dict
                if len(batch_data[9].shape) >= 2:
                    tmp_batch = batch_data[9].squeeze()
                else:
                    tmp_batch = batch_data[9]
                frame_id = int(tmp_batch[0].item())
                id1_used_here = batch_data[10].squeeze().tolist()
                id2_used_here = batch_data[11].squeeze().tolist()
                gt_xyangle_view1_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id1_used_here, cfg.gt_ratio)
                gt_xyangle_view2_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id2_used_here, cfg.gt_ratio)

                # record rotated information
                rotated_points_xy_list.append(new_xzpred2.tolist())
                rotated_points_r_list.append(new_angle2.tolist())
                rotated_points_id_list.append(id2_used_here)
                rotated_bbox_list.append(bbox2)
                # record rotated information

                # gt aggregation
                # [[x, y, r], [x, y, r], ...]
                aggregated_points = []
                bbox_list = []
                gt_points = []  # the same shape as aggregated points
                id_added_list = []
                # record person id pair added to aggregated pair
                # [[id_view1, id_view2], [id_view1, id_view2], [], []]
                id_pair_list = []

                # using by gt
                id_pair_list2 = []

                # gt select point pairs

                # choosing point pairs by using gt
                aggregated_points2 = []
                gt_points2 = []  # the same shape as aggregated points
                id_added_list2 = []

                for i in range(len(id1_used_here)):
                    for j in range(len(id2_used_here)):

                        if id1_used_here[i] in id2_used_here and id1_used_here[i] == id2_used_here[j]:
                            r_tmp = (original_angle_list1[i].item() + new_angle2[j].item()) / 2
                            if r_tmp > math.pi:
                                r_tmp = r_tmp - 2 * math.pi
                            if r_tmp < -math.pi:
                                r_tmp = 2 * math.pi + r_tmp

                            aggregated_points2.append([(original_xz_list1[i][0].item() + new_xzpred2[j][0].item()) / 2,
                                                       (original_xz_list1[i][1].item() + new_xzpred2[j][1].item()) / 2,
                                                       r_tmp])
                            gt_points2.append(gt_xyangle_view1_list[i])
                            id_added_list2.append(id1_used_here[i])

                            id_pair_list2.append([id1_used_here[i], id2_used_here[j]])

                        else:
                            # person only in view 1 and not in view2
                            if id1_used_here[i] not in id_added_list2 and id1_used_here[i] not in id2_used_here:
                                id_added_list2.append(id1_used_here[i])
                                aggregated_points2.append(
                                    [original_xz_list1[i][0].item(), original_xz_list1[i][1].item(),
                                     original_angle_list1[i].item()])
                                gt_points2.append(gt_xyangle_view1_list[i])
                            # person only in view2 and not in view1
                            if id2_used_here[j] not in id_added_list2 and id2_used_here[j] not in id1_used_here:
                                id_added_list2.append(id2_used_here[j])
                                aggregated_points2.append([new_xzpred2[j][0].item(), new_xzpred2[j][1].item(),
                                                           new_angle2[j].item()])
                                gt_points2.append(gt_xyangle_view2_list[j])

                id_added_set = set(id_added_list2)
                id_need_to_add = set(id1_used_here).union(set(id2_used_here))
                # make sure that the resutl is reasonable
                assert (id_need_to_add - id_added_set) == set() and len(id_need_to_add) == len(
                    aggregated_points2) == len(
                    gt_points2)
                # gt aggregation

                # creating matrix statistic unit without gt
                pseudo_dis_matrix = torch.zeros_like(match_matrix)
                pseudo_r_matrix = torch.zeros_like(match_matrix)
                n, m = pseudo_dis_matrix.shape
                for i in range(n):
                    for j in range(m):
                        # here has no grad, because new_xzpred2 has no gard and original_xz_list.item()
                        pseudo_dis_matrix[i][j] = math.sqrt(((new_xzpred2[i][0].item() - original_xz_list1[j][
                            0].item()) ** 2 + (new_xzpred2[i][1].item() - original_xz_list1[j][1].item()) ** 2))
                        pseudo_r_matrix[i][j] = clac_rad_distance(new_angle2[i], original_angle_list1[j]).item()

                reid_val_matrix = match_matrix.clone().detach()
                dis_mask = pseudo_dis_matrix <= cfg.distance_threshold
                reid_val_mask = reid_val_matrix >= matrix_threshold
                mask = torch.logical_and(dis_mask, reid_val_mask)

                n, m = mask.shape

                pair_list = []
                for i in range(n):
                    for j in range(m):
                        if mask[i][j] == True:
                            pair_list.append([[j, i], pseudo_dis_matrix[i][j].item(), reid_val_matrix[i][j].item(),
                                              pseudo_r_matrix[i][j].item()])

                pair_list.sort(key=lambda x: (x[1], x[2]))

                # used in view1
                used_list1 = set()
                # used in view2
                used_list2 = set()
                pair_list_filterd = []
                for index, pair in enumerate(pair_list):
                    i_tmp, j_tmp = pair[0]
                    if i_tmp in used_list1 or j_tmp in used_list2:
                        pass
                    else:
                        pair_list_filterd.append(pair)
                        used_list1.add(pair[0][0])
                        used_list2.add(pair[0][1])
                # creating matrix statistic unit

                # making aggregation from points pair

                view1_used_index = set()
                view2_used_index = set()
                # pair points
                # i : view1
                # j : view2
                try:
                    for (i, j), _, _, _ in pair_list_filterd:
                        x_view1 = original_xz_list1[i][0].item()
                        y_view1 = original_xz_list1[i][1].item()
                        r_view1 = original_angle_list1[i].item()

                        x_view2 = new_xzpred2[j][0].item()
                        y_view2 = new_xzpred2[j][1].item()
                        r_view2 = new_angle2[j].item()

                        r_tmp = (r_view1 + r_view2) / 2
                        if r_tmp > math.pi:
                            r_tmp = r_tmp - 2 * math.pi
                        if r_tmp < -math.pi:
                            r_tmp = 2 * math.pi + r_tmp

                        aggregated_points.append([(x_view1 + x_view2) / 2, (y_view1 + y_view2) / 2, r_tmp])
                        gt_points.append(gt_xyangle_view1_list[i])
                        id_added_list.append(id1_used_here[i])
                        box_tmp = [bbox1[i], bbox2[j]]
                        bbox_list.append(box_tmp)

                        view1_used_index.add(i)
                        view2_used_index.add(j)

                        id_pair_list.append([id1_used_here[i], id2_used_here[j]])

                    # cauculating single points
                    view1_all_index = set([k for k in range(len(original_xz_list1))])
                    view2_all_index = set([k for k in range(len(new_xzpred2))])

                    view1_left_index = view1_all_index - view1_used_index
                    view2_left_index = view2_all_index - view2_used_index
                    # adding single points to aggregated point collections
                    for index in view1_left_index:
                        aggregated_points.append(
                            [original_xz_list1[index][0].item(), original_xz_list1[index][1].item(),
                             original_angle_list1[index].item()])
                        gt_points.append(gt_xyangle_view1_list[index])
                        id_added_list.append(id1_used_here[index])
                        bbox_list.append(bbox1[index])
                    for index in view2_left_index:
                        aggregated_points.append(
                            [new_xzpred2[index][0].item(), new_angle2[index].item(), new_angle2[index].item()])
                        gt_points.append(gt_xyangle_view2_list[index])
                        id_added_list.append(id2_used_here[index])
                        bbox_list.append(bbox2[index])
                except:
                    continue
                id_added_set = set(id_added_list)
                id_need_to_add = set(id1_used_here).union(set(id2_used_here))

                # only caculating pair iou, without considering single one
                # pair_point_intersection += len(id_added_set)
                # pair_point_union += len(id_need_to_add)

                pair_point_intersection += sum([1 if pair[0] == pair[1] else 0 for pair in id_pair_list])
                pair_point_union += len(id_pair_list2)
                # iou = len(id_added_set)/ len(id_need_to_add)

                hit_meter.update(aggregated_points, gt_points, intersection_num, union_num)
                hit_meter_gt.update(aggregated_points2, gt_points2, intersection_num, union_num)

                camera_meter.update([[x_pred, y_pred, r_pred]], [[x_gt, y_gt, r_gt]], 0, 0)

                # f1 about
                hit_meter.add_f1_score(reid_f1.item())
                hit_meter_gt.add_f1_score(reid_f1.item())
                ######          caculating xy and r Hit probability       ######

            # 4 (main view is not here)
            comp_view_num = len(view_candidate_dataset_list)

            # --------- aggregated these points ------------

            # preparing total data list
            # id
            id_total_list = main_points_id_list[:]
            for i in range(comp_view_num):
                id_total_list.extend(rotated_points_id_list[i])
            # xy and r
            xyr_total_list = copy.copy(main_points_xyangle_list)
            bbox_total_list = main_bbox_list
            top_pred_xy_total_list = [[xyr[0].item(), xyr[1].item()] for xyr in xyr_total_list]
            top_pred_r_total_list = [xyr[2].item() for xyr in xyr_total_list]
            for i in range(len(rotated_points_xy_list)):
                top_pred_xy_total_list.extend(rotated_points_xy_list[i])
                top_pred_r_total_list.extend(rotated_points_r_list[i])
                bbox_total_list.extend(rotated_bbox_list[i])

            person_interval_list = []
            tmp_sum = 0
            for person in person_num_list:
                person_interval_list.append(tmp_sum)
                tmp_sum += person
            person_interval_list.append(tmp_sum)

            # preparing total data list

            # creating reid matrix
            total_feature_cat = torch.cat(view_reid_feature_list, dim=0)
            match_total_matrix = get_eu_distance_mtraix(total_feature_cat, total_feature_cat)

            # creating distance matrix
            distance_total_matrix = torch.zeros_like(match_total_matrix)
            r_total_matrix = torch.zeros_like(match_total_matrix)

            # creating r matrix

            # creating gt matrix, distance matrix and r matrix
            match_total_matrix_gt = torch.zeros_like(match_total_matrix)
            for i in range(len(id_total_list)):
                for j in range(len(id_total_list)):
                    if id_total_list[i] == id_total_list[j]:
                        match_total_matrix_gt[i][j] = 1
                    distance_total_matrix[i][j] = math.sqrt(((top_pred_xy_total_list[i][0] - top_pred_xy_total_list[j][
                        0]) ** 2 + (top_pred_xy_total_list[i][1] - top_pred_xy_total_list[j][1]) ** 2))
                    r_total_matrix[i][j] = clac_rad_distance_with_no_tensor(top_pred_r_total_list[i],
                                                                            top_pred_r_total_list[j])

            dis_mask = distance_total_matrix <= cfg.distance_threshold
            reid_val_mask = match_total_matrix >= matrix_threshold
            mask = torch.logical_and(dis_mask, reid_val_mask)
            # Simplify the matrix by symmetry
            mask_up = torch.triu(mask, 1)
            total_f1 = calc_f1_loss_by_matrix(torch.triu(match_total_matrix_gt, 1), mask_up.int()).item()
            total_f1_sum += total_f1
            # remove self matrix mask
            person_num_sum = 0
            for person_num in person_num_list:
                mask_up[person_num_sum:person_num_sum + person_num, person_num_sum:person_num_sum + person_num] = False
                person_num_sum += person_num

            # creating gt matrix, distance matrix and r matrix

            # aggregate points
            pair_list = []
            # [[index, score]]
            union_find = UnionFind(len(id_total_list))
            pair_dict = {i: [] for i in range(len(id_total_list))}

            for i in range(len(id_total_list)):
                for j in range(len(id_total_list)):
                    if mask_up[i][j] == True:
                        pair_dict[i].append([j, match_total_matrix[i][j]])
                        union_find.union(i, j)

            union_collection_dict = {i: [] for i in range(len(id_total_list))}
            for i in range(len(id_total_list)):
                union_collection_dict[union_find.find(i)].append(i)
            # filter the union_collection
            aggregated_list = []
            for key, val in union_collection_dict.items():
                if len(val) > 1:
                    aggregated_list.append(val)

            original_aggregated_list = copy.copy(aggregated_list)
            original_aggregated_real_id_list = [list(map(lambda x: id_total_list[x], elem)) for elem in aggregated_list]

            # using sub-graph algorithm
            pop_index_list = []
            for index, aggregated_sub_list in enumerate(aggregated_list):
                split_list = []
                person_interval_counter = [[] for i in range(len(person_interval_list) - 1)]
                for i in range(len(person_interval_list) - 1):
                    for j in range(len(aggregated_sub_list)):
                        if aggregated_sub_list[j] >= person_interval_list[i] and aggregated_sub_list[j] < \
                                person_interval_list[i + 1]:
                            person_interval_counter[i].append(aggregated_sub_list[j])
                person_counter_max = max(
                    [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                if person_counter_max > 1:
                    # split
                    while person_counter_max > 1:
                        for i in range(len(person_interval_counter)):
                            tmp_split = []
                            if len(person_interval_counter[i]) > 0:
                                pivot = person_interval_counter[i][0]
                                tmp_split.append(person_interval_counter[i].pop(0))
                                for j in range(i + 1, len(person_interval_counter)):
                                    score_list = [match_total_matrix[pivot][elem_index] for elem_index in
                                                  person_interval_counter[j]]
                                    if len(score_list) == 0:
                                        continue
                                    if max(score_list) < cfg.matrix_threshold:
                                        break
                                    else:
                                        max_index = score_list.index(max(score_list))
                                        tmp_split.append(person_interval_counter[j].pop(max_index))
                            if tmp_split != []:
                                aggregated_list.append(tmp_split)

                        person_counter_max = max(
                            [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                        # collecting splited list
                    # collecting remaining list
                    pop_index_list.append(index)
                    remaining_list = []
                    for i in range(len(person_interval_counter)):
                        for j in range(len(person_interval_counter[i])):
                            remaining_list.append(person_interval_counter[i][j])
                    if remaining_list != []:
                        aggregated_list.append(remaining_list)



                else:
                    pass

            aggregated_list_new = []
            for i in range(len(aggregated_list)):
                if i in pop_index_list:
                    continue
                else:
                    aggregated_list_new.append(aggregated_list[i])

            aggregated_list = aggregated_list_new
            aggregated_real_id_list = [list(map(lambda x: id_total_list[x], elem)) for elem in aggregated_list]

            # show statistic result
            id_list_counter = Counter(id_total_list)

            target_ratio_sum = 0

            for aggregated_sub_list in aggregated_list:
                target_ratio = len(aggregated_sub_list) / id_list_counter[id_total_list[aggregated_sub_list[0]]]
                # print(f'{id_total_list[aggregated_sub_list[0]]}: {target_ratio}')
                target_ratio_sum += target_ratio

            final_target_ratio = round(target_ratio_sum / len(id_list_counter.keys()), 4)
            # show statistic result

            index_used_set = set()
            total_index_set = set([i for i in range(len(id_total_list))])

            # mean choosing
            final_xy_total_list = []
            final_r_total_list = []

            # centroid 1 choosing
            final_only_one_xy_total_list = []
            final_only_one_r_total_list = []
            final_only_one_bbox_total_list = []

            # centroid 2 mean choosing
            final_two_mean_xy_total_list = []
            final_two_mean_r_total_list = []

            final_xy_total_gt_list = []
            final_r_total_gt_list = []
            final_id_total_list = []
            # aggregated more view points here
            for aggregated_sub_list in aggregated_list:
                x_sum = 0
                y_sum = 0
                r_sum = 0
                # used to caculate centroid
                x_cache = []
                y_cache = []
                r_cache = []

                for index_sub in range(len(aggregated_sub_list)):
                    index_used_set.add(aggregated_sub_list[index_sub])

                    x_cache.append(top_pred_xy_total_list[index_sub][0])
                    y_cache.append(top_pred_xy_total_list[index_sub][1])
                    r_cache.append(top_pred_r_total_list[index_sub])

                    x_sum += top_pred_xy_total_list[index_sub][0]
                    y_sum += top_pred_xy_total_list[index_sub][1]
                    r_sum += top_pred_r_total_list[index_sub]
                x_mean = x_sum / len(aggregated_sub_list)
                y_mean = y_sum / len(aggregated_sub_list)
                r_mean = r_sum / len(aggregated_sub_list)

                # centroid about
                centroid_x = x_mean
                centroid_y = y_mean
                to_centroid_distance_list = [math.sqrt((x_cache[i] - x_mean) ** 2 + (y_cache[i] - y_mean) ** 2) for i in
                                             range(len(x_cache))]
                rank_list = torch.argsort(torch.tensor(to_centroid_distance_list)).tolist()
                rank1_index = rank_list.index(0)
                if 1 in rank_list:
                    rank2_index = rank_list.index(1)
                else:
                    rank2_index = rank1_index

                total1_index = aggregated_sub_list[rank1_index]
                total2_index = aggregated_sub_list[rank2_index]

                final_only_one_xy_total_list.append(top_pred_xy_total_list[total1_index])
                final_only_one_r_total_list.append(top_pred_r_total_list[total1_index])
                final_only_one_bbox_total_list.append(
                    [bbox_total_list[aggregated_sub_list[i]] for i in range(len(aggregated_sub_list))])

                final_two_mean_xy_total_list.append(
                    [(top_pred_xy_total_list[total1_index][0] + top_pred_xy_total_list[total2_index][0]) / 2,
                     (top_pred_xy_total_list[total1_index][1] + top_pred_xy_total_list[total2_index][1]) / 2])
                new_two_mean_r = (top_pred_r_total_list[total1_index] + top_pred_r_total_list[total2_index]) / 2
                if new_two_mean_r > math.pi:
                    new_two_mean_r = new_two_mean_r - 2 * math.pi
                if new_two_mean_r < -math.pi:
                    new_two_mean_r = 2 * math.pi + new_two_mean_r

                final_two_mean_r_total_list.append(new_two_mean_r)

                # centroid about

                if r_mean > math.pi:
                    r_mean = r_mean - 2 * math.pi
                if r_mean < -math.pi:
                    r_mean = 2 * math.pi + r_mean
                id_used_here = id_total_list[aggregated_sub_list[0]]
                final_xy_total_list.append([x_mean, y_mean])
                final_r_total_list.append(r_mean)
                final_id_total_list.append(id_used_here)
                final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])
                final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])
            # aggregated more view points here

            # aggregated single points here
            single_index_list = sorted(list(total_index_set - index_used_set))

            for single_index in single_index_list:
                id_used_here = id_total_list[single_index]
                xy_to_be_added = [top_pred_xy_total_list[single_index][0], top_pred_xy_total_list[single_index][1]]
                r_to_be_added = top_pred_r_total_list[single_index]

                final_xy_total_list.append(
                    [top_pred_xy_total_list[single_index][0], top_pred_xy_total_list[single_index][1]])
                final_only_one_xy_total_list.append(xy_to_be_added)
                final_two_mean_xy_total_list.append(xy_to_be_added)
                final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])

                final_r_total_list.append(top_pred_r_total_list[single_index])
                final_only_one_r_total_list.append(r_to_be_added)
                final_two_mean_r_total_list.append(r_to_be_added)
                final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])

                final_only_one_bbox_total_list.append([bbox_total_list[single_index]])
                final_id_total_list.append(id_used_here)
            # aggregated single points here

            # update final target ration here and statistic final prob
            statistic_counter_dict = {id: [] for id in list(set(id_total_list))}
            for index_val, id_val in enumerate(id_total_list):
                statistic_counter_dict[id_val].append(index_val)
            # sort list
            for key in statistic_counter_dict.keys():
                statistic_counter_dict[key].sort()
            gt_aggregated_list = [val for val in statistic_counter_dict.values()]
            # statistic every person's probability
            prob_counter = 0
            for i in range(len(aggregated_list)):
                if aggregated_list[i] in gt_aggregated_list:
                    prob_counter += 1

            iter_str = (5 - len(str(iter))) * '0' + str(iter)
            frame_str = (5 - len(str(frame_id))) * '0' + str(frame_id)
            expand_ratio = 1.3

            # this is point target ration in top view
            original_final_id_total_list = final_id_total_list[:]
            final_target_ratio = len(set(final_id_total_list).intersection(set(id_total_list))) / len(
                set(id_total_list))
            final_person_ratio = prob_counter / len(aggregated_list)
            no_repeat_tag = len(set(final_id_total_list)) == len(final_id_total_list)

            # [[frame_str, person_prob, id_prob, repeat_tag]]
            prob_statistic_cache.append([frame_str, final_person_ratio, final_target_ratio, no_repeat_tag])
            hit_total_mean_meter.update(
                [[final_xy_total_list[i][0], final_xy_total_list[i][1], final_r_total_list[i]] for i in
                 range(len(final_xy_total_list))],
                [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1], final_r_total_gt_list[i]] for i in
                 range(len(final_xy_total_gt_list))], 0, 0)

            hit_total_one_meter.update([[final_only_one_xy_total_list[i][0], final_only_one_xy_total_list[i][1],
                                         final_only_one_r_total_list[i]] for i in range(len(final_xy_total_list))],
                                       [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1],
                                         final_r_total_gt_list[i]] for i in range(len(final_xy_total_gt_list))], 0, 0)

            hit_total_two_meter.update([[final_two_mean_xy_total_list[i][0], final_two_mean_xy_total_list[i][1],
                                         final_two_mean_r_total_list[i]] for i in range(len(final_xy_total_list))],
                                       [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1],
                                         final_r_total_gt_list[i]] for i in range(len(final_xy_total_gt_list))], 0, 0)
            # update statistic

            # aggregate points

            # --------- aggregated these points ------------

            ################   Visualization Code   ##############################
            # draw gt coverage
            gt_xyangle_list = []
            id_list = []
            for key, val in draw_gt_id_xyangle_dict_total.items():
                id_list.append(key)
                gt_xyangle_list.append(val)

            # filter  subjects with cameras
            while max(id_list) > 20:
                index_max = id_list.index(max(id_list))
                del id_list[index_max]
                del gt_xyangle_list[index_max]

            # adding gt camera
            # adding main camera
            gt_xyangle_list.append([0, 0, -90 / 57.3])
            id_list.append(-1)
            # adding more view cameras
            for i in range(len(cam_xy_pred_list)):
                gt_xyangle_list.append([cam_xy_gt_list[i][0], cam_xy_gt_list[i][1], cam_r_gt_list[i]])
                id_list.append(-2 - i)

            for i in range(len(gt_xyangle_list)):
                gt_xyangle_list[i][0] *= expand_ratio
                gt_xyangle_list[i][1] *= expand_ratio

            cfg.generator.get_heatmap_from_xzangle_id(gt_xyangle_list, id_list, False, adding_border=True)
            cfg.generator.save_img(os.path.join(cfg.result_path, f"figs/{frame_str}_2_gt_top_view.png"))
            gt_coverage = (cfg.generator.board * 2).int()
            # draw gt coverage
            # -----------------------------------------
            # draw single view

            # combine xy, angle to xyr
            final_xyr = [[final_xy_total_list[i][0], final_xy_total_list[i][1], final_r_total_list[i]] for i in
                         range(len(final_id_total_list))]
            final_xyr_one = [
                [final_only_one_xy_total_list[i][0], final_only_one_xy_total_list[i][1], final_only_one_r_total_list[i]]
                for i in range(len(final_id_total_list))]
            final_xyr_two = [
                [final_two_mean_xy_total_list[i][0], final_two_mean_xy_total_list[i][1], final_two_mean_r_total_list[i]]
                for i in range(len(final_id_total_list))]

            # draw combine pred images
            if_cropped = False

            # adding pred camera
            # adding main camera
            final_xyr.append([0, 0, -90 / 57.3])
            final_xyr_one.append([0, 0, -90 / 57.3])
            final_xyr_two.append([0, 0, -90 / 57.3])
            final_id_total_list.append(-1)
            # adding more view cameras
            for i in range(len(cam_xy_pred_list)):
                final_xyr.append([cam_xy_pred_list[i][0], cam_xy_pred_list[i][1], cam_r_pred_list[i]])
                final_xyr_one.append([cam_xy_pred_list[i][0], cam_xy_pred_list[i][1], cam_r_pred_list[i]])
                final_xyr_two.append([cam_xy_pred_list[i][0], cam_xy_pred_list[i][1], cam_r_pred_list[i]])
                final_id_total_list.append(-2 - i)

            # cfg.generator.get_heatmap_from_xzangle_id(final_xyr, final_id_total_list, if_cropped=if_cropped)
            # cfg.generator.board += gt_coverage
            # cfg.generator.save_img(os.path.join(cfg.result_path, f"figs/frame_{iter_str}_mean_{final_target_ratio}.png"))

            for i in range(len(final_xyr_one)):
                final_xyr_one[i][0] *= expand_ratio
                final_xyr_one[i][1] *= expand_ratio

            while max(final_id_total_list) > 20:
                index_max = final_id_total_list.index(max(final_id_total_list))
                del final_xyr_one[index_max]
                del final_id_total_list[index_max]

            cfg.generator.get_heatmap_from_xzangle_id(final_xyr_one, final_id_total_list, if_cropped=if_cropped)
            # cfg.generator.board += gt_coverage
            cfg.generator.save_img(os.path.join(cfg.result_path, f"figs/{frame_str}_1_virtual_top_view.png"))

            # draw different view with bbox and real top view
            bbox_view_1 = []
            bbox_view_1_id = []
            bbox_view_2 = []
            bbox_view_2_id = []
            bbox_view_3 = []
            bbox_view_3_id = []
            bbox_view_4 = []
            bbox_view_4_id = []
            bbox_view_5 = []
            bbox_view_5_id = []

            for id in range(1, len(final_only_one_bbox_total_list) + 1):
                index = id - 1
                bbox_list = final_only_one_bbox_total_list[index]
                for bbox in bbox_list:
                    if bbox[0] == 1:
                        bbox_view_1.append(bbox[1:])
                        bbox_view_1_id.append(original_final_id_total_list[index])
                    elif bbox[0] == 2:
                        bbox_view_2.append(bbox[1:])
                        bbox_view_2_id.append(original_final_id_total_list[index])
                    elif bbox[0] == 3:
                        bbox_view_3.append(bbox[1:])
                        bbox_view_3_id.append(original_final_id_total_list[index])
                    elif bbox[0] == 4:
                        bbox_view_4.append(bbox[1:])
                        bbox_view_4_id.append(original_final_id_total_list[index])
                    elif bbox[0] == 5:
                        bbox_view_5.append(bbox[1:])
                        bbox_view_5_id.append(original_final_id_total_list[index])
                    else:
                        # print('no such view!')
                        pass

            try:
                view1_color = [tuple(rgb_table[id]) for id in bbox_view_1_id]
                view2_color = [tuple(rgb_table[id]) for id in bbox_view_2_id]
                view3_color = [tuple(rgb_table[id]) for id in bbox_view_3_id]
                view4_color = [tuple(rgb_table[id]) for id in bbox_view_4_id]
                view5_color = [tuple(rgb_table[id]) for id in bbox_view_5_id]
            except:
                continue

            img1_path, img2_path = view_candidate_dataset_list[0].img_pair_path[iter]
            _, img3_path = view_candidate_dataset_list[1].img_pair_path[iter]
            _, img4_path = view_candidate_dataset_list[2].img_pair_path[iter]
            _, img5_path = view_candidate_dataset_list[3].img_pair_path[iter]
            tmp = img1_path.split('/')
            tmp[-2] = 'top_video'
            img_top_path = '/'.join(tmp)

            img1 = torchvision.io.read_image(img1_path)
            img1 = draw_bounding_box_with_color_or_labels(img1, bbox_view_1, colors=view1_color)
            img2 = torchvision.io.read_image(img2_path)
            img2 = draw_bounding_box_with_color_or_labels(img2, bbox_view_2, colors=view2_color)
            img3 = torchvision.io.read_image(img3_path)
            img3 = draw_bounding_box_with_color_or_labels(img3, bbox_view_3, colors=view3_color)
            img4 = torchvision.io.read_image(img4_path)
            img4 = draw_bounding_box_with_color_or_labels(img4, bbox_view_4, colors=view4_color)
            img5 = torchvision.io.read_image(img5_path)
            img5 = draw_bounding_box_with_color_or_labels(img5, bbox_view_5, colors=view5_color)
            img_top = torchvision.io.read_image(img_top_path)
            # top bbox
            top_bbox_id = torch.load(os.path.join(cfg.label, 'f_top_bbox_pid.pth'))[str(int(frame_str))]
            top_id_list = list(set(original_final_id_total_list))
            top_color = [tuple(rgb_table[id]) for id in top_id_list]
            top_bbox_list = []
            for bbox, id in top_bbox_id:
                if id in top_id_list:
                    top_bbox_list.append(bbox)
            img_top = draw_bounding_box_with_color_or_labels(img_top, top_bbox_list, colors=top_color, width=2)

            img_list = [img1, img2, img3, img4, img5, img_top]
            imgs = torch.stack(img_list, dim=0)
            imgs = make_grid(imgs, padding=4, pad_value=255.0, nrow=3)
            torchvision.io.write_png(imgs,
                                     os.path.join(cfg.result_path, f"figs/{frame_str}_0_original_cmp_img_grid.png"))

            # sort and save cache used to select the better cases
            prob_statistic_cache.sort(key=lambda elem: (elem[1], elem[2]), reverse=True)
            torch.save(prob_statistic_cache, os.path.join(cfg.result_path, 'prob_cache_sorted.pth'))

            # torch.save(cfg.statistic_cache, os.path.join(cfg.result_path, f"statistic_cache.pth"))

        test_info = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            're_id_loss': loss_meter.reid_loss_avg,
            'mono_xy_loss': loss_meter.mono_xy_loss_avg,
            'mono_r_loss': loss_meter.mono_r_loss_avg,
            'loss': loss_meter.total_loss,
            'prob': hit_meter.get_xy_r_prob_dict(),
            'iou': hit_meter.get_f1_score(),
            'prob_gt': hit_meter_gt.get_xy_r_prob_dict(),
            'pair_point_iou': pair_point_intersection / pair_point_union,
            'person_xy_mean_loss': hit_meter.get_xy_mean_error(),
            'person_r_mean_loss': hit_meter.get_r_mean_error(),
            'person_xy_mean_gt_loss': hit_meter_gt.get_xy_mean_error(),
            'person_r_mean_gt_loss': hit_meter_gt.get_r_mean_error(),
            'cam_prob': camera_meter.get_xy_r_prob_dict(),
            'total_prob': hit_total_one_meter.get_xy_r_prob_dict(),
            'total_mean_xy': hit_total_one_meter.get_xy_mean_error(),
            'total_mean_r': hit_total_one_meter.get_r_mean_error(),
            'total_f1': total_f1_sum / len(view_candidate_dataset_list[0])

        }

        return test_info


def _test_monoreid_cam_gt_centroid_soft_choose_more_view_without_fig(view_candidate_dataset_list, model_mono,
                                                                     model_reid, device, epoch, cfg):
    cfg.epoch = epoch
    if cfg.matrix_threshold > 0.5:
        matrix_threshold = cfg.matrix_threshold / 2
    else:
        matrix_threshold = cfg.matrix_threshold

    epoch_timer = Timer()

    loss_meter = AverageMeter_MonoReid()
    hit_meter = HitProbabilityMeter_Monoreid()
    hit_meter_gt = HitProbabilityMeter_Monoreid()

    # different types of subject aggregation, the centroid selection one is `hit_total_one_meter`
    hit_total_mean_meter = HitProbabilityMeter_Monoreid()
    hit_total_one_meter = HitProbabilityMeter_Monoreid()
    hit_total_two_meter = HitProbabilityMeter_Monoreid()

    print(len(view_candidate_dataset_list[0]) * len(view_candidate_dataset_list))

    camera_meter = HitProbabilityMeter_Monoreid()
    total_f1_sum = 0
    # calculating pair point IOU
    pair_point_intersection = 0
    pair_point_union = 0
    # [[frame_str, person_prob, id_prob, repeat_tag]]
    prob_statistic_cache = []

    model_mono.eval()
    model_reid.eval()

    with torch.no_grad():
        # traverse all the view pairs
        for iter in range(len(view_candidate_dataset_list[0])):
            print(f'processing test case {iter}')
            draw_gt_id_xyangle_dict_total = {}
            cam_xy_pred_list = []
            cam_xy_gt_list = []
            cam_r_pred_list = []
            cam_r_gt_list = []

            rotated_points_xy_list = []
            rotated_points_r_list = []
            rotated_points_id_list = []
            rotated_bbox_list = []

            main_points_xyangle_list = []
            main_points_id_list = []
            main_bbox_list = []

            view_reid_feature_list = []

            person_num_list = []

            inner_loss_meter = AverageMeter_MonoReid()

            # start time
            # traverse all the frames in this view pair
            for view_pair_id in range(len(view_candidate_dataset_list)):

                batch_data = view_candidate_dataset_list[view_pair_id][iter]

                # prepare batch data
                batch_data = [try_to(b.unsqueeze(dim=0), device) for b in batch_data]

                t1 = time.time()

                output_dict1 = model_mono(batch_data[4].squeeze(), batch_data[5].squeeze())
                output_dict2 = model_mono(batch_data[6].squeeze(), batch_data[7].squeeze())

                # loconet output
                original_xz_list1 = output_dict1['xz_pred']
                original_xz_list2 = output_dict2['xz_pred']
                original_angle_list1 = output_dict1['angles']
                original_angle_list2 = output_dict2['angles']

                # getting ground truth
                fp_dict = cfg.fp_dict
                frame_id = int(batch_data[9][0][0].item())
                id1_used_here = batch_data[10].squeeze().tolist()
                id2_used_here = batch_data[11].squeeze().tolist()
                id1_set = set(id1_used_here)
                id2_set = set(id2_used_here)
                id_list = list(id1_set.union(id2_set))
                gt_xyangle_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id_list, cfg.gt_ratio)
                for index_draw_gt, id_draw_gt in enumerate(id_list):
                    draw_gt_id_xyangle_dict_total[id_draw_gt] = gt_xyangle_list[index_draw_gt]

                frame_id = int(batch_data[9][0][0].item())
                view_id1 = int(batch_data[9][0][1].item())
                view_id2 = int(batch_data[9][0][3].item())

                # getting boxes
                try:
                    bbox1 = [[view_id1, bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]] for bbox_score in
                             batch_data[5].squeeze().cpu().tolist()]
                    bbox2 = [[view_id2, bbox_score[0], bbox_score[1], bbox_score[2], bbox_score[3]] for bbox_score in
                             batch_data[7].squeeze().cpu().tolist()]
                except:
                    continue

                # create xzangle_list() from xz_list() and angle_list()
                # view 1
                n = len(original_xz_list1)
                # xzangle_list of view 1
                xzangle1_list = []
                for i in range(n):
                    xzangle1_list.append([original_xz_list1[i][0], original_xz_list1[i][1], original_angle_list1[i]])

                # view 2
                n = len(original_xz_list2)
                # xzangle_list of view 2
                xzangle2_list = []
                for i in range(n):
                    xzangle2_list.append([original_xz_list2[i][0], original_xz_list2[i][1], original_angle_list2[i]])

                # id 1
                id1 = batch_data[10]
                # id 2
                id2 = batch_data[11]

                if len(main_points_xyangle_list) == 0:
                    main_points_xyangle_list = xzangle1_list
                    main_points_id_list = id1.squeeze().tolist()
                    main_bbox_list = bbox1

                ######     evaluating re-id resutl      ########
                # reid_test
                n1 = batch_data[2].shape[1]
                reid_total_batch = torch.cat([batch_data[2].squeeze(), batch_data[3].squeeze()], dim=0)
                reid_output = model_reid(reid_total_batch)
                reid_output1 = reid_output[:n1, :]
                reid_output2 = reid_output[n1:, :]
                t_feat_end = time.time()

                # recording features
                if len(view_reid_feature_list) == 0:
                    view_reid_feature_list.append(reid_output1)
                view_reid_feature_list.append(reid_output2)

                if len(person_num_list) == 0:
                    person_num_list.append(reid_output1.shape[0])
                person_num_list.append(reid_output2.shape[0])

                # getting top3 pairs
                match_matrix = get_eu_distance_mtraix(reid_output1, reid_output2)
                selected_pair_list = get_top_pair_from_matrix(match_matrix, 3)

                # creating matrix gt
                matrix_gt = torch.zeros_like(match_matrix)
                for i in range(id2.squeeze().shape[0]):
                    for j in range(id1.squeeze().shape[0]):
                        if id1.squeeze()[j] == id2.squeeze()[i]:
                            matrix_gt[i][j] = 1.0

                re_id_loss = torch.nn.MSELoss()(match_matrix, matrix_gt)

                # calculating matrix iou
                predict = match_matrix.clone().detach()
                predict[predict > matrix_threshold] = 1.0
                predict[predict <= matrix_threshold] = 0.0
                gt = matrix_gt.clone().detach()

                predict = predict.int()
                gt = gt.int()

                intersection_num = torch.sum(predict * gt)
                union_num = torch.sum(torch.bitwise_or(predict, gt))

                # calculating f1 score
                reid_f1 = calc_f1_loss_by_matrix(gt, predict)

                ###### [SAM]          rotation here       ######
                t1_rotation = time.time()
                # using selected pair indices to rotate
                theta_list, deltax_list, deltay_list = get_theta_delx_dely_from_pairs(output_dict1['angles'],
                                                                                      output_dict2['angles'],
                                                                                      output_dict1['xz_pred'],
                                                                                      output_dict2['xz_pred'],
                                                                                      selected_pair_list)

                # centroid selection strategy
                x_centroid = 0
                y_centroid = 0

                for i in range(len(deltax_list)):
                    x_centroid += deltax_list[i].item()
                    y_centroid += deltay_list[i].item()
                x_centroid /= len(deltay_list)
                y_centroid /= len(deltay_list)

                select_index = 0
                min_distance = 9999
                for i in range(len(theta_list)):
                    distance = math.sqrt((deltax_list[i] - x_centroid) ** 2 + (deltay_list[i] - y_centroid) ** 2)
                    if distance < min_distance:
                        select_index = i
                        min_distance = distance

                ######          caculating cam xy and r loss       ######
                x_gt, y_gt, r_gt = batch_data[8][0][0].item(), batch_data[8][0][1].item(), batch_data[8][0][2].item()
                x_gt *= cfg.gt_ratio
                y_gt *= cfg.gt_ratio
                r_gt -= 90
                if r_gt > 180:
                    r_gt = r_gt - 360
                r_gt /= 57.3

                cam_xy_loss = 0
                cam_r_loss = 0

                # selecting the min centroid distance camera to calculate the loss
                i = select_index
                theta_output = theta_list[i] * 57.3
                if theta_output < 0:
                    theta_output += 360
                theta_output = 360 - theta_output

                theta_output -= 90
                if theta_output > 180:
                    theta_output = theta_output - 360
                r_pred = theta_output / 57.3
                x_pred = deltax_list[i]
                y_pred = deltay_list[i]

                t2_rotation = time.time()

                cam_xy_loss += torch.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2)
                cam_r_loss += clac_rad_distance(r_gt, r_pred)

                loss_meter.update(re_id_loss.item(), cam_xy_loss.item(), cam_r_loss.item())
                inner_loss_meter.update(re_id_loss.item(), cam_xy_loss.item(), cam_r_loss.item())

                # recording camera prediction
                cam_xy_pred_list.append([x_pred.item(), y_pred.item()])
                cam_xy_gt_list.append([x_gt, y_gt])
                cam_r_pred_list.append(r_pred.item())
                cam_r_gt_list.append(r_gt)

                ######          caculating xy and r Hit probability       ######

                new_xzpred2, new_angle2 = rotate_x_y_by_point(output_dict2['xz_pred'], output_dict2['angles'],
                                                              delta_x=deltax_list[select_index],
                                                              delta_y=deltay_list[select_index],
                                                              delta_theta=theta_list[select_index])

                fp_dict = cfg.fp_dict
                if len(batch_data[9].shape) >= 2:
                    tmp_batch = batch_data[9].squeeze()
                else:
                    tmp_batch = batch_data[9]
                frame_id = int(tmp_batch[0].item())
                id1_used_here = batch_data[10].squeeze().tolist()
                id2_used_here = batch_data[11].squeeze().tolist()
                gt_xyangle_view1_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id1_used_here, cfg.gt_ratio)
                gt_xyangle_view2_list = get_xyangle_gt_from_fp(fp_dict, frame_id, id2_used_here, cfg.gt_ratio)

                # recording rotated information
                rotated_points_xy_list.append(new_xzpred2.tolist())
                rotated_points_r_list.append(new_angle2.tolist())
                rotated_points_id_list.append(id2_used_here)
                rotated_bbox_list.append(bbox2)

                # using gt to finish subject aggregation, only used to analyse
                # [[x, y, r], [x, y, r], ...]
                aggregated_points = []
                bbox_list = []
                gt_points = []  # the same shape as aggregated points
                id_added_list = []
                # record person id pair added to aggregated pair
                # [[id_view1, id_view2], [id_view1, id_view2], [], []]
                id_pair_list = []

                # using by gt
                id_pair_list2 = []

                # gt select point pairs

                # choosing point pairs by using gt
                aggregated_points2 = []
                gt_points2 = []  # the same shape as aggregated points
                id_added_list2 = []

                for i in range(len(id1_used_here)):
                    for j in range(len(id2_used_here)):

                        if id1_used_here[i] in id2_used_here and id1_used_here[i] == id2_used_here[j]:
                            r_tmp = (original_angle_list1[i].item() + new_angle2[j].item()) / 2
                            if r_tmp > math.pi:
                                r_tmp = r_tmp - 2 * math.pi
                            if r_tmp < -math.pi:
                                r_tmp = 2 * math.pi + r_tmp

                            aggregated_points2.append([(original_xz_list1[i][0].item() + new_xzpred2[j][0].item()) / 2,
                                                       (original_xz_list1[i][1].item() + new_xzpred2[j][1].item()) / 2,
                                                       r_tmp])
                            gt_points2.append(gt_xyangle_view1_list[i])
                            id_added_list2.append(id1_used_here[i])

                            id_pair_list2.append([id1_used_here[i], id2_used_here[j]])

                        else:
                            # person only in view 1 and not in view2
                            if id1_used_here[i] not in id_added_list2 and id1_used_here[i] not in id2_used_here:
                                id_added_list2.append(id1_used_here[i])
                                aggregated_points2.append(
                                    [original_xz_list1[i][0].item(), original_xz_list1[i][1].item(),
                                     original_angle_list1[i].item()])
                                gt_points2.append(gt_xyangle_view1_list[i])
                            # person only in view2 and not in view1
                            if id2_used_here[j] not in id_added_list2 and id2_used_here[j] not in id1_used_here:
                                id_added_list2.append(id2_used_here[j])
                                aggregated_points2.append([new_xzpred2[j][0].item(), new_xzpred2[j][1].item(),
                                                           new_angle2[j].item()])
                                gt_points2.append(gt_xyangle_view2_list[j])

                # [re_id_predict, re_id_gt, [[select_i, select_j]], [select_re_id_val], [select_dis], [select_r]]
                # cfg.statistic_cache.append([match_matrix.cpu(), matrix_gt.cpu(), select_ij_list, select_reid_val_list, select_dis_list, select_r_list])

                id_added_set = set(id_added_list2)
                id_need_to_add = set(id1_used_here).union(set(id2_used_here))
                # make sure that the resutl is reasonable
                assert (id_need_to_add - id_added_set) == set() and len(id_need_to_add) == len(
                    aggregated_points2) == len(
                    gt_points2)
                # gt aggregation

                t1_two_asso = time.time()

                # creating matrix statistic unit without gt
                pseudo_dis_matrix = torch.zeros_like(match_matrix)
                pseudo_r_matrix = torch.zeros_like(match_matrix)
                n, m = pseudo_dis_matrix.shape
                for i in range(n):
                    for j in range(m):
                        # here has no grad, because new_xzpred2 has no gard and original_xz_list.item()
                        pseudo_dis_matrix[i][j] = math.sqrt(((new_xzpred2[i][0].item() - original_xz_list1[j][
                            0].item()) ** 2 + (new_xzpred2[i][1].item() - original_xz_list1[j][1].item()) ** 2))
                        pseudo_r_matrix[i][j] = clac_rad_distance(new_angle2[i], original_angle_list1[j]).item()

                reid_val_matrix = match_matrix.clone().detach()
                dis_mask = pseudo_dis_matrix <= cfg.distance_threshold
                reid_val_mask = reid_val_matrix >= matrix_threshold
                mask = torch.logical_and(dis_mask, reid_val_mask)

                n, m = mask.shape

                pair_list = []
                for i in range(n):
                    for j in range(m):
                        if mask[i][j] == True:
                            pair_list.append([[j, i], pseudo_dis_matrix[i][j].item(), reid_val_matrix[i][j].item(),
                                              pseudo_r_matrix[i][j].item()])

                pair_list.sort(key=lambda x: (x[1], x[2]))

                # used in view1
                used_list1 = set()
                # used in view2
                used_list2 = set()
                pair_list_filterd = []
                for index, pair in enumerate(pair_list):
                    i_tmp, j_tmp = pair[0]
                    if i_tmp in used_list1 or j_tmp in used_list2:
                        pass
                    else:
                        pair_list_filterd.append(pair)
                        used_list1.add(pair[0][0])
                        used_list2.add(pair[0][1])

                # making aggregation from points pair
                view1_used_index = set()
                view2_used_index = set()
                # pair points
                # i : view1
                # j : view2
                try:
                    for (i, j), _, _, _ in pair_list_filterd:
                        x_view1 = original_xz_list1[i][0].item()
                        y_view1 = original_xz_list1[i][1].item()
                        r_view1 = original_angle_list1[i].item()

                        x_view2 = new_xzpred2[j][0].item()
                        y_view2 = new_xzpred2[j][1].item()
                        r_view2 = new_angle2[j].item()

                        r_tmp = (r_view1 + r_view2) / 2
                        if r_tmp > math.pi:
                            r_tmp = r_tmp - 2 * math.pi
                        if r_tmp < -math.pi:
                            r_tmp = 2 * math.pi + r_tmp

                        aggregated_points.append([(x_view1 + x_view2) / 2, (y_view1 + y_view2) / 2, r_tmp])
                        gt_points.append(gt_xyangle_view1_list[i])
                        id_added_list.append(id1_used_here[i])
                        box_tmp = [bbox1[i], bbox2[j]]
                        bbox_list.append(box_tmp)

                        view1_used_index.add(i)
                        view2_used_index.add(j)

                        id_pair_list.append([id1_used_here[i], id2_used_here[j]])

                    # cauculating single points
                    view1_all_index = set([k for k in range(len(original_xz_list1))])
                    view2_all_index = set([k for k in range(len(new_xzpred2))])

                    view1_left_index = view1_all_index - view1_used_index
                    view2_left_index = view2_all_index - view2_used_index
                    # adding single points to aggregated point collections
                    for index in view1_left_index:
                        aggregated_points.append(
                            [original_xz_list1[index][0].item(), original_xz_list1[index][1].item(),
                             original_angle_list1[index].item()])
                        gt_points.append(gt_xyangle_view1_list[index])
                        id_added_list.append(id1_used_here[index])
                        bbox_list.append(bbox1[index])
                    for index in view2_left_index:
                        aggregated_points.append(
                            [new_xzpred2[index][0].item(), new_xzpred2[index][1].item(), new_angle2[index].item()])
                        gt_points.append(gt_xyangle_view2_list[index])
                        id_added_list.append(id2_used_here[index])
                        bbox_list.append(bbox2[index])
                except:
                    continue

                t2 = time.time()

                pair_point_intersection += sum([1 if pair[0] == pair[1] else 0 for pair in id_pair_list])
                pair_point_union += len(id_pair_list2)
                # iou = len(id_added_set)/ len(id_need_to_add)

                hit_meter.update(aggregated_points, gt_points, intersection_num, union_num)
                hit_meter_gt.update(aggregated_points2, gt_points2, intersection_num, union_num)

                camera_meter.update([[x_pred, y_pred, r_pred]], [[x_gt, y_gt, r_gt]], 0, 0)

                # f1 about
                hit_meter.add_f1_score(reid_f1.item())
                hit_meter_gt.add_f1_score(reid_f1.item())

            # 4 (main view is not here, only containing cmp views)
            comp_view_num = len(view_candidate_dataset_list)

            # --------- aggregated these points ------------

            # preparing total data list
            # id
            id_total_list = main_points_id_list[:]
            for i in range(comp_view_num):
                id_total_list.extend(rotated_points_id_list[i])
            # xy and r
            xyr_total_list = copy.copy(main_points_xyangle_list)
            bbox_total_list = main_bbox_list
            top_pred_xy_total_list = [[xyr[0].item(), xyr[1].item()] for xyr in xyr_total_list]
            top_pred_r_total_list = [xyr[2].item() for xyr in xyr_total_list]
            for i in range(len(rotated_points_xy_list)):
                top_pred_xy_total_list.extend(rotated_points_xy_list[i])
                top_pred_r_total_list.extend(rotated_points_r_list[i])
                bbox_total_list.extend(rotated_bbox_list[i])

            person_interval_list = []
            tmp_sum = 0
            for person in person_num_list:
                person_interval_list.append(tmp_sum)
                tmp_sum += person
            person_interval_list.append(tmp_sum)

            t1_asso = time.time()
            # creating reid matrix
            total_feature_cat = torch.cat(view_reid_feature_list, dim=0)
            match_total_matrix = get_eu_distance_mtraix(total_feature_cat, total_feature_cat)

            # creating distance matrix
            distance_total_matrix = torch.zeros_like(match_total_matrix)
            r_total_matrix = torch.zeros_like(match_total_matrix)

            # creating r matrix
            # creating gt matrix, distance matrix and r matrix
            match_total_matrix_gt = torch.zeros_like(match_total_matrix)
            for i in range(len(id_total_list)):
                for j in range(len(id_total_list)):
                    if id_total_list[i] == id_total_list[j]:
                        match_total_matrix_gt[i][j] = 1
                    distance_total_matrix[i][j] = math.sqrt(((top_pred_xy_total_list[i][0] - top_pred_xy_total_list[j][
                        0]) ** 2 + (top_pred_xy_total_list[i][1] - top_pred_xy_total_list[j][1]) ** 2))
                    r_total_matrix[i][j] = clac_rad_distance_with_no_tensor(top_pred_r_total_list[i],
                                                                            top_pred_r_total_list[j])

            dis_mask = distance_total_matrix <= cfg.distance_threshold
            reid_val_mask = match_total_matrix >= matrix_threshold
            mask = torch.logical_and(dis_mask, reid_val_mask)
            # Simplify the matrix by symmetry
            mask_up = torch.triu(mask, 1)
            total_f1 = calc_f1_loss_by_matrix(torch.triu(match_total_matrix_gt, 1), mask_up.int()).item()
            total_f1_sum += total_f1
            # remove self matrix mask
            person_num_sum = 0
            for person_num in person_num_list:
                mask_up[person_num_sum:person_num_sum + person_num, person_num_sum:person_num_sum + person_num] = False
                person_num_sum += person_num

            # aggregating subject points
            pair_list = []
            # [[index, score]]
            union_find = UnionFind(len(id_total_list))
            pair_dict = {i: [] for i in range(len(id_total_list))}

            for i in range(len(id_total_list)):
                for j in range(len(id_total_list)):
                    if mask_up[i][j] == True:
                        pair_dict[i].append([j, match_total_matrix[i][j]])
                        union_find.union(i, j)

            union_collection_dict = {i: [] for i in range(len(id_total_list))}
            for i in range(len(id_total_list)):
                union_collection_dict[union_find.find(i)].append(i)
            # filter the union_collection
            aggregated_list = []
            for key, val in union_collection_dict.items():
                if len(val) > 1:
                    aggregated_list.append(val)

            t2_asso = time.time()

            # using hierarchical sub-graph algorithm
            pop_index_list = []
            for index, aggregated_sub_list in enumerate(aggregated_list):
                split_list = []
                person_interval_counter = [[] for i in range(len(person_interval_list) - 1)]
                for i in range(len(person_interval_list) - 1):
                    for j in range(len(aggregated_sub_list)):
                        if aggregated_sub_list[j] >= person_interval_list[i] and aggregated_sub_list[j] < \
                                person_interval_list[i + 1]:
                            person_interval_counter[i].append(aggregated_sub_list[j])
                person_counter_max = max(
                    [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                if person_counter_max > 1:
                    # split
                    while person_counter_max > 1:
                        for i in range(len(person_interval_counter)):
                            tmp_split = []
                            if len(person_interval_counter[i]) > 0:
                                pivot = person_interval_counter[i][0]
                                tmp_split.append(person_interval_counter[i].pop(0))
                                for j in range(i + 1, len(person_interval_counter)):
                                    score_list = [match_total_matrix[pivot][elem_index] for elem_index in
                                                  person_interval_counter[j]]
                                    if len(score_list) == 0:
                                        continue
                                    if max(score_list) < cfg.matrix_threshold:
                                        break
                                    else:
                                        max_index = score_list.index(max(score_list))
                                        tmp_split.append(person_interval_counter[j].pop(max_index))
                            if tmp_split != []:
                                aggregated_list.append(tmp_split)

                        person_counter_max = max(
                            [len(person_interval_counter[i]) for i in range(len(person_interval_list) - 1)])
                        # collecting splited list
                    # collecting remaining list
                    pop_index_list.append(index)
                    remaining_list = []
                    for i in range(len(person_interval_counter)):
                        for j in range(len(person_interval_counter[i])):
                            remaining_list.append(person_interval_counter[i][j])
                    if remaining_list != []:
                        aggregated_list.append(remaining_list)

                else:
                    pass
            aggregated_list_new = []
            for i in range(len(aggregated_list)):
                if i in pop_index_list:
                    continue
                else:
                    aggregated_list_new.append(aggregated_list[i])

            aggregated_list = aggregated_list_new

            # aggregated_real_id_list = [list(map(lambda x: id_total_list[x], elem)) for elem in aggregated_list]

            # show statistic result
            id_list_counter = Counter(id_total_list)

            target_ratio_sum = 0

            for aggregated_sub_list in aggregated_list:
                target_ratio = len(aggregated_sub_list) / id_list_counter[id_total_list[aggregated_sub_list[0]]]
                # print(f'{id_total_list[aggregated_sub_list[0]]}: {target_ratio}')
                target_ratio_sum += target_ratio

            final_target_ratio = round(target_ratio_sum / len(id_list_counter.keys()), 4)
            # show statistic result

            index_used_set = set()
            total_index_set = set([i for i in range(len(id_total_list))])

            # mean choosing
            final_xy_total_list = []
            final_r_total_list = []

            # centroid 1 choosing
            final_only_one_xy_total_list = []
            final_only_one_r_total_list = []
            final_only_one_bbox_total_list = []

            # centroid 2 mean choosing
            final_two_mean_xy_total_list = []
            final_two_mean_r_total_list = []

            final_xy_total_gt_list = []
            final_r_total_gt_list = []
            final_id_total_list = []
            # aggregated more view points here
            for aggregated_sub_list in aggregated_list:
                x_sum = 0
                y_sum = 0
                r_sum = 0
                # used to caculate centroid
                x_cache = []
                y_cache = []
                r_cache = []

                for index_sub in range(len(aggregated_sub_list)):
                    index_used_set.add(aggregated_sub_list[index_sub])

                    x_cache.append(top_pred_xy_total_list[index_sub][0])
                    y_cache.append(top_pred_xy_total_list[index_sub][1])
                    r_cache.append(top_pred_r_total_list[index_sub])

                    x_sum += top_pred_xy_total_list[index_sub][0]
                    y_sum += top_pred_xy_total_list[index_sub][1]
                    r_sum += top_pred_r_total_list[index_sub]
                x_mean = x_sum / len(aggregated_sub_list)
                y_mean = y_sum / len(aggregated_sub_list)
                r_mean = r_sum / len(aggregated_sub_list)

                # centroid about
                centroid_x = x_mean
                centroid_y = y_mean
                to_centroid_distance_list = [math.sqrt((x_cache[i] - x_mean) ** 2 + (y_cache[i] - y_mean) ** 2) for i in
                                             range(len(x_cache))]
                rank_list = torch.argsort(torch.tensor(to_centroid_distance_list)).tolist()
                rank1_index = rank_list.index(0)
                if 1 in rank_list:
                    rank2_index = rank_list.index(1)
                else:
                    rank2_index = rank1_index

                total1_index = aggregated_sub_list[rank1_index]
                total2_index = aggregated_sub_list[rank2_index]

                final_only_one_xy_total_list.append(top_pred_xy_total_list[total1_index])
                final_only_one_r_total_list.append(top_pred_r_total_list[total1_index])
                final_only_one_bbox_total_list.append(
                    [bbox_total_list[aggregated_sub_list[i]] for i in range(len(aggregated_sub_list))])

                final_two_mean_xy_total_list.append(
                    [(top_pred_xy_total_list[total1_index][0] + top_pred_xy_total_list[total2_index][0]) / 2,
                     (top_pred_xy_total_list[total1_index][1] + top_pred_xy_total_list[total2_index][1]) / 2])
                new_two_mean_r = (top_pred_r_total_list[total1_index] + top_pred_r_total_list[total2_index]) / 2
                if new_two_mean_r > math.pi:
                    new_two_mean_r = new_two_mean_r - 2 * math.pi
                if new_two_mean_r < -math.pi:
                    new_two_mean_r = 2 * math.pi + new_two_mean_r

                final_two_mean_r_total_list.append(new_two_mean_r)

                # centroid about

                if r_mean > math.pi:
                    r_mean = r_mean - 2 * math.pi
                if r_mean < -math.pi:
                    r_mean = 2 * math.pi + r_mean
                id_used_here = id_total_list[aggregated_sub_list[0]]
                final_xy_total_list.append([x_mean, y_mean])
                final_r_total_list.append(r_mean)
                final_id_total_list.append(id_used_here)
                final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])
                final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])
            # aggregated more view points here

            # aggregated single points here
            single_index_list = sorted(list(total_index_set - index_used_set))

            for single_index in single_index_list:
                id_used_here = id_total_list[single_index]
                xy_to_be_added = [top_pred_xy_total_list[single_index][0], top_pred_xy_total_list[single_index][1]]
                r_to_be_added = top_pred_r_total_list[single_index]

                final_xy_total_list.append(
                    [top_pred_xy_total_list[single_index][0], top_pred_xy_total_list[single_index][1]])
                final_only_one_xy_total_list.append(xy_to_be_added)
                final_two_mean_xy_total_list.append(xy_to_be_added)
                final_xy_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][:2])

                final_r_total_list.append(top_pred_r_total_list[single_index])
                final_only_one_r_total_list.append(r_to_be_added)
                final_two_mean_r_total_list.append(r_to_be_added)
                final_r_total_gt_list.append(draw_gt_id_xyangle_dict_total[id_used_here][2])

                final_only_one_bbox_total_list.append([bbox_total_list[single_index]])
                final_id_total_list.append(id_used_here)
            # aggregated single points here

            # update final target ration here and statistic final prob
            statistic_counter_dict = {id: [] for id in list(set(id_total_list))}
            for index_val, id_val in enumerate(id_total_list):
                statistic_counter_dict[id_val].append(index_val)
            # sort list
            for key in statistic_counter_dict.keys():
                statistic_counter_dict[key].sort()
            gt_aggregated_list = [val for val in statistic_counter_dict.values()]
            # statistic every person's probability
            prob_counter = 0
            for i in range(len(aggregated_list)):
                if aggregated_list[i] in gt_aggregated_list:
                    prob_counter += 1

            iter_str = (5 - len(str(iter))) * '0' + str(iter)
            frame_str = (5 - len(str(frame_id))) * '0' + str(frame_id)
            expand_ratio = 1.3

            # frame_num = int(frame_str)
            # person_num = len(gt_xyangle_list)

            hit_tmp = HitProbabilityMeter_Monoreid()
            hit_tmp.update([[final_only_one_xy_total_list[i][0], final_only_one_xy_total_list[i][1],
                             final_only_one_r_total_list[i]] for i in range(len(final_xy_total_list))],
                           [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1], final_r_total_gt_list[i]] for i
                            in range(len(final_xy_total_gt_list))], 0, 0)
            # person_dis_avg = hit_tmp.get_xy_mean_error()
            # person_ang_avg = hit_tmp.get_r_mean_error()
            #
            # cam_dis_avg = inner_loss_meter.mono_xy_loss_avg
            # cam_ang_avg = inner_loss_meter.mono_r_loss_avg

            # calc person distribution
            person_distribution_sum = 0
            counter_here = 0

            for i in range(len(gt_xyangle_list)):
                for j in range(len(gt_xyangle_list)):
                    if i == j:
                        continue
                    else:
                        counter_here += 1
                        x1, y1 = gt_xyangle_list[i][0], gt_xyangle_list[i][1]
                        x2, y2 = gt_xyangle_list[j][0], gt_xyangle_list[j][1]

                        person_distribution_sum += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            person_distribution = person_distribution_sum / counter_here

            # this is point target ration in top view
            original_final_id_total_list = final_id_total_list[:]
            final_target_ratio = len(set(final_id_total_list).intersection(set(id_total_list))) / len(
                set(id_total_list))
            final_person_ratio = prob_counter / len(aggregated_list)
            no_repeat_tag = len(set(final_id_total_list)) == len(final_id_total_list)

            # update statistic
            # [[frame_str, person_prob, id_prob, repeat_tag]]
            prob_statistic_cache.append([frame_str, final_person_ratio, final_target_ratio, no_repeat_tag])
            hit_total_mean_meter.update(
                [[final_xy_total_list[i][0], final_xy_total_list[i][1], final_r_total_list[i]] for i in
                 range(len(final_xy_total_list))],
                [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1], final_r_total_gt_list[i]] for i in
                 range(len(final_xy_total_gt_list))], 0, 0)

            hit_total_one_meter.update([[final_only_one_xy_total_list[i][0], final_only_one_xy_total_list[i][1],
                                         final_only_one_r_total_list[i]] for i in range(len(final_xy_total_list))],
                                       [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1],
                                         final_r_total_gt_list[i]] for i in range(len(final_xy_total_gt_list))], 0, 0)

            hit_total_two_meter.update([[final_two_mean_xy_total_list[i][0], final_two_mean_xy_total_list[i][1],
                                         final_two_mean_r_total_list[i]] for i in range(len(final_xy_total_list))],
                                       [[final_xy_total_gt_list[i][0], final_xy_total_gt_list[i][1],
                                         final_r_total_gt_list[i]] for i in range(len(final_xy_total_gt_list))], 0, 0)

        test_info = {
            'epoch': epoch,
            'time': epoch_timer.timeit(),
            're_id_loss': loss_meter.reid_loss_avg,
            'mono_xy_loss': loss_meter.mono_xy_loss_avg,
            'mono_r_loss': loss_meter.mono_r_loss_avg,
            'loss': loss_meter.total_loss,
            'prob': hit_meter.get_xy_r_prob_dict(),
            'iou': hit_meter.get_f1_score(),
            'prob_gt': hit_meter_gt.get_xy_r_prob_dict(),
            'pair_point_iou': pair_point_intersection / pair_point_union,
            'person_xy_mean_loss': hit_meter.get_xy_mean_error(),
            'person_r_mean_loss': hit_meter.get_r_mean_error(),
            'person_xy_mean_gt_loss': hit_meter_gt.get_xy_mean_error(),
            'person_r_mean_gt_loss': hit_meter_gt.get_r_mean_error(),
            'cam_prob': camera_meter.get_xy_r_prob_dict(),
            'total_prob': hit_total_one_meter.get_xy_r_prob_dict(),
            'total_mean_xy': hit_total_one_meter.get_xy_mean_error(),
            'total_mean_r': hit_total_one_meter.get_r_mean_error(),
            'total_f1': total_f1_sum / len(view_candidate_dataset_list[0])

        }

        return test_info


def try_to(ts, device):
    if ts is not None:
        return ts.to(device)
    else:
        return None


def adjust_lr(optimizer, new_lr):
    print('change learning rate:', new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
