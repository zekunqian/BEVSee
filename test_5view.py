import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from train_net import *
import sys
from config.config import *
import os

sys.path.append(".")
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

cfg = Config('CSRD_Virtual')  # setting the dataset
cfg.channel = 3  # control the channel of input image

cfg.model_name = 'mono21'
cfg.train_test_split = '11'  # split ratio
cfg.core_task = 'bevsee'
cfg.extra = 'test_csrd5'
cfg.exp_note = '+'.join([cfg.model_name, cfg.train_test_split, cfg.core_task, cfg.extra])

# deveice about
cfg.num_workers = 8
cfg.device_list = "3"

# test_about
cfg.test_batch_size = 1
cfg.test_interval_epoch = 1
cfg.save_model_interval_epoch = 50
cfg.test_before_train = True

# visualization
cfg.draw_fig = False
# cfg.draw_fig_interval_epoch = 2
cfg.is_colorful = True

# train about
cfg.batch_size = 1  # batch size only support 1
cfg.train_learning_rate = 1e-6
cfg.train_dropout_prob = 0
cfg.weight_decay = 0
cfg.is_iteration = False
cfg.iteration_time = 5
# cfg.lr_plan = {500: 1e-7, 800: 1e-8, 1100: 1e-10}
cfg.max_epoch = 200
cfg.train_pin_memory = True

# continue training setting
cfg.iscontinue = True
# cfg.continue_path = '/data1/clark/dataset/data1002/log/main_expriment/centroid/epoch40.pth'
cfg.continue_path = 'models/csrd5.pth'
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
cfg.matrix_threshold = 0.46
cfg.distance_threshold = 2.0  # origin is 2.0
cfg.reid_f1_threshold = 0.7
cfg.view_num = 5  # scrd5 has 5 views to test

# whether loading pretrained loconet
cfg.without_load_model = True
cfg.loconet_pretrained_model_path = 'models/pretrained_loconet.pth'

if cfg.draw_fig:
    test_more_view_aggregation_net(cfg)  # drawing figures methods is much slower than the without figure one
else:
    test_more_view_aggregation_without_fig_net(cfg)
