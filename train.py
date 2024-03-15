import os

os.environ['CUDA_VISIBLE_DEVICES'] = "3"
from train_net import *
from config.config import *
import sys

sys.path.append(".")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

cfg = Config('CSRD_Virtual')  # setting the dataset
cfg.channel = 3  # control the channel of input image

# extra note
cfg.model_name = 'mono21'
cfg.train_test_split = '11'  # split ratio
cfg.core_task = 'bevsee'
cfg.extra = 'csrd2_train'
cfg.exp_note = '+'.join([cfg.model_name, cfg.train_test_split, cfg.core_task, cfg.extra])

# deveice about
cfg.num_workers = 8
cfg.device_list = "3"

# test_about
cfg.test_batch_size = 1
cfg.test_interval_epoch = 1
cfg.save_model_interval_epoch = 50
cfg.test_before_train = False

# visualization
cfg.draw_fig = False
cfg.test_draw_fig_num = 20  # frame nums to visualize per epoch
cfg.draw_fig_interval_epoch = 2  # epoch interval of loss curves
cfg.is_colorful = True

# train about
cfg.batch_size = 1  # batch size only support 1
cfg.train_learning_rate = 1e-6
cfg.train_dropout_prob = 0
cfg.weight_decay = 0
# cfg.lr_plan = {500: 1e-7, 800: 1e-8, 1100: 1e-10}
cfg.max_epoch = 200
cfg.train_pin_memory = True

# continue training setting
cfg.iscontinue = False
cfg.continue_path = './models/csrd2.pth'
cfg.continue_dir = ''  # not necessary

# loss coefficient
cfg.xy_ratio = 1
cfg.r_ratio = 1
cfg.reid_ratio = 1

# dataset about
cfg.train_num = 1000
cfg.test_num = 1000
cfg.gt_ratio = 0.212  # the ratio of synthetic to real

# trian about
# dis_pesudo_ratio + r_pesudo_ratio = 1.0
cfg.dis_pseudo_ratio = 0.5
cfg.matrix_threshold = 0.23
cfg.distance_threshold = 2.0
cfg.reid_f1_threshold = 0.7
cfg.view_num = 2

# whether loading pretrained loconet
cfg.without_load_model = False
cfg.loconet_pretrained_model_path = 'models/pretrained_loconet.pth'

train_monoreid_net_with_xy_and_r(cfg)
