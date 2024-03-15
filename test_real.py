import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from train_net import *
import sys

sys.path.append(".")
from config.config import *
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

cfg = Config('CSRDR_V2_G1_tracking')

# channle mode
cfg.channel = 3

# cfg.model_name = 'mono21'
cfg.model_name = 'mono21'
cfg.train_test_split = '11'  # split ratio
cfg.core_task = 'bevsee'
cfg.extra = 'test_csrdr'
# generating dir name
cfg.exp_note = '+'.join([cfg.model_name, cfg.train_test_split, cfg.core_task, cfg.extra])

# deveice about
cfg.num_workers = 1
cfg.device_list = "3"
cfg.use_multi_gpu = False

# test_about
cfg.test_batch_size = 1
cfg.test_interval_epoch = 1
cfg.test_draw_fig_num = 0
cfg.save_model_interval_epoch = 0
cfg.test_before_train = False
cfg.draw_fig = False
cfg.draw_fig_interval_epoch = 2
cfg.is_colorful = True

# train about
cfg.batch_size = 1  # 8
cfg.train_learning_rate = 1e-6
cfg.train_dropout_prob = 0
cfg.weight_decay = 0
cfg.max_epoch = 1800
cfg.train_pin_memory = True

# continue setting
cfg.iscontinue = True
cfg.continue_path = 'models/csrdr.pth'  # *
cfg.continue_dir = ''  # not necessary

cfg.xy_ratio = 1
cfg.r_ratio = 1
cfg.reid_ratio = 1

# dataset about
cfg.train_num = 1000
cfg.test_num = 1000
cfg.gt_ratio = 0.212

# trian about
# dis_pesudo_ratio + r_pesudo_ratio = 1.0
cfg.dis_pseudo_ratio = 0.5

cfg.matrix_threshold = 0.25
cfg.distance_threshold = 5
cfg.reid_f1_threshold = 0.7
cfg.aggregate_view_num = (cfg.view_num + 1) if cfg.view_num == 2 else cfg.view_num

cfg.without_load_model = True
cfg.loconet_pretrained_model_path = ''

# reid models
cfg.reload_reid_path = 'models/csrdr_reid.pth'  # using pretrained reid models
# cfg.reload_reid_path = ''
cfg.is_only_used_pretrained_reid = False

cfg.real_dataset_name_list = ['CSRDR_V2_G1_tracking', 'CSRDR_V2_G3_tracking', 'CSRDR_V3_G2_tracking',
                              'CSRDR_V3_G3_tracking', 'CSRDR_V3_G5_tracking', 'CSRDR_V4_G1_tracking',
                              'CSRDR_V4_G2_tracking']

test_more_view_aggregation_cosine_similarity(cfg)
