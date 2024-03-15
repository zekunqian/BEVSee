import time
import os
from utils.utils import readJson
from utils.make_sector_board import Generator
import torch


class Config(object):
    """
    class to save config parameter
    """

    def __init__(self, dataset_name, json_path='config/config.json'):
        # Global
        # self.ori_image_size=1088,1920
        self.image_size = 631, 630  # input image size
        self.batch_size = 32  # train batch size
        self.test_batch_size = 8  # test batch size
        self.num_workers = 12
        self.generator = Generator(10, cache_path='models/sector_cache_torch_45_thread.pth')
        config_file = readJson(json_path)
        self.config_file = config_file

        # Gpu
        self.use_gpu = True
        self.use_multi_gpu = False

        # Dataset
        assert "Virtual" in dataset_name or 'CIP' in dataset_name or 'Tracking' in dataset_name or 'CSRDR' in dataset_name
        self.dataset_name = dataset_name
        self.force = False

        self.root_view = config_file['Datasets'][dataset_name]['root_view']
        self.comp_view_list = config_file['Datasets'][dataset_name]['comp_view_list']
        self.view_num = config_file['Datasets'][dataset_name]['view']
        if 'person_num' in config_file['Datasets'][dataset_name]:
            self.person_num = config_file['Datasets'][dataset_name]['person_num']

        # (view1_dir, view2_dir)
        self.input_list = [(self.root_view, view2) for view2 in self.comp_view_list]
        self.all_input_list = [self.root_view]
        self.all_input_list.extend(self.comp_view_list)

        if 'json' in config_file['Datasets'][dataset_name].keys():
            self.json_path = config_file['Datasets'][dataset_name]['json']
        if 'annotation' in config_file['Datasets'][dataset_name].keys():
            if 'CSRDR' in dataset_name and 'tracking' in dataset_name:
                self.annotation_txt_dir_path = config_file['Datasets'][dataset_name]['annotation']
            elif 'CSRDR' in dataset_name:
                self.annotation_txt_dir_path = config_file['Datasets'][dataset_name]['annotation']
            else:
                self.fv_dict = torch.load(config_file['Datasets'][dataset_name]['annotation'] + '/fv.pth')

        self.label = config_file['Datasets'][dataset_name]['label']
        self.output_path = config_file['Datasets'][dataset_name]['output']
        self.skip_list = []

        self.train_num = -1
        self.test_num = -1
        self.channel = 3
        self.monoloco_prefix = 'out_'
        self.monoloco_suffix = '.monoloco.json'
        self.z_max = 25  # max subject distance

        # Backbone
        self.output_size = 3  # (x, y, rotation)

        # Training Parameters
        self.train_random_seed = 0
        self.train_learning_rate = 2e-4  # initial learning rate
        self.lr_plan = {500: 1e-7, 800: 1e-8, 1100: 1e-10}  # change learning rate in these epochs
        self.train_dropout_prob = 0.2  # dropout probability
        self.weight_decay = 0  # l2 weight decay

        self.max_epoch = 150  # max training epoch
        self.test_interval_epoch = 10
        self.only_test = False

    def init_config(self, need_new_folder=True):
        # if self.exp_name is None:
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.exp_name = '<%s>at<%s>' % (self.exp_note, time_str)

        self.result_path = os.path.join(self.output_path, self.exp_name)
        self.log_path = os.path.join(self.output_path, '%s/log.txt' % self.exp_name)
        if self.iscontinue and self.continue_dir != '':
            self.result_path = self.continue_dir
            self.log_path = os.path.join(self.continue_dir, 'log.txt')
        # print(self.result_path)
        # print(self.log_path)

        if need_new_folder and not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
            # mk figure dir
            os.mkdir(os.path.join(self.result_path, 'figs'))
