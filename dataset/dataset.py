try:
    from dataset.virtual_dataset import *
    from dataset.real_dataset import *
except:
    from virtual_dataset import *
    from real_dataset import *


def return_dataset(cfg):
    # train_and_test_restriction = [3000, 500]
    if 'Virtual' in cfg.dataset_name or 'Tracking' in cfg.dataset_name:
        train_list, train_label, test_list, test_label = [], [], [], []
        for i in range(1, cfg.view_num + 1):
            # skip some views, start from index 0
            if i + 1 in cfg.skip_list:
                continue
            if i <= cfg.view_num - 1:
                # todo add counter annotation here
                train_list.append(cfg.input_list[i - 1])
                if 'Tracking' in cfg.dataset_name:
                    with open(cfg.label + '/' + f"camerabias{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]) * 5, float(tmp[2]) * 5, float(tmp[3])]
                                train_label.append(tmp)
                else:
                    with open(cfg.label + '/' + f"camera{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                                train_label.append(tmp)
            else:
                test_list.append(cfg.input_list[i - 1])

                if 'Tracking' in cfg.dataset_name:
                    with open(cfg.label + '/' + f"camerabias{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]) * 5, float(tmp[2]) * 5, float(tmp[3])]
                                test_label.append(tmp)
                else:
                    with open(cfg.label + '/' + f"camera{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                                test_label.append(tmp)

        cfg.dataset_num = -1

        if cfg.train_num != -1:
            cfg.dataset_num = cfg.train_num
        training_dataset = VirtualDataset(cfg, train_list, train_label)

        if cfg.test_num != -1:
            cfg.dataset_num = cfg.test_num

        valid_dataset = VirtualDataset(cfg, test_list, test_label)

        return training_dataset, valid_dataset


def return_more_view_dataset(cfg):
    # train_and_test_restriction = [3000, 500]
    if 'Virtual' in cfg.dataset_name or 'Tracking' in cfg.dataset_name:
        train_list, train_label, test_list, test_label = [], [], [], []
        for i in range(1, cfg.view_num + 1):
            if i <= cfg.view_num - 1:
                # todo add counter annotation here
                train_list.append(cfg.input_list[i - 1])
                if 'Tracking' in cfg.dataset_name:
                    with open(cfg.label + '/' + f"camerabias{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]) * 5, float(tmp[2]) * 5, float(tmp[3])]
                                train_label.append(tmp)

                else:
                    with open(cfg.label + '/' + f"camera{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                                train_label.append(tmp)
            else:
                test_list.append(cfg.input_list[i - 1])
                if 'Tracking' in cfg.dataset_name:
                    with open(cfg.label + '/' + f"camerabias{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]) * 5, float(tmp[2]) * 5, float(tmp[3])]
                                test_label.append(tmp)
                else:
                    with open(cfg.label + '/' + f"camera{i + 1}.txt") as f:
                        for line in f:
                            if len(line) > 3:
                                tmp = line.split()
                                tmp = [float(tmp[1]), float(tmp[2]), float(tmp[3])]
                                test_label.append(tmp)

        cfg.dataset_num = -1

        if cfg.train_num != -1:
            cfg.dataset_num = cfg.train_num

        if cfg.view_num > 2:
            training_dataset = None
        else:
            training_dataset = VirtualDataset(cfg, train_list, train_label)

        if cfg.test_num != -1:
            cfg.dataset_num = cfg.test_num

        valid_dataset = VirtualDataset(cfg, test_list, test_label)

        return training_dataset, valid_dataset


def return_more_view_dataset_real_without_annotation(cfg):
    if 'CIP' in cfg.dataset_name:
        train_list, train_label, test_list, test_label = [], [], [], []
        for i in range(1, cfg.view_num + 1):
            if i <= cfg.view_num - 1:
                # todo add counter annotation here
                train_list.append(cfg.input_list[i - 1])
            else:
                test_list.append(cfg.input_list[i - 1])

        cfg.dataset_num = -1

        if cfg.train_num != -1:
            cfg.dataset_num = cfg.train_num

        if cfg.view_num > 2:
            training_dataset = None
        else:
            training_dataset = RealDataset(cfg, train_list, train_label)

        if cfg.test_num != -1:
            cfg.dataset_num = cfg.test_num

        valid_dataset = RealDataset(cfg, test_list, test_label)

        return training_dataset, valid_dataset

    elif 'CSRDR' in cfg.dataset_name:
        train_list, train_label, test_list, test_label = [], [], [], []
        for i in range(1, cfg.view_num + 1):
            if i <= cfg.view_num - 1:
                # todo add counter annotation here
                train_list.append(cfg.input_list[i - 1])
            else:
                if i - 1 >= len(cfg.input_list):
                    break
                test_list.append(cfg.input_list[i - 1])

        cfg.dataset_num = -1

        if cfg.train_num != -1:
            cfg.dataset_num = cfg.train_num

        if cfg.view_num > 2:
            training_dataset = None
        else:
            training_dataset = RealDataset_CvHMTB(cfg, train_list, train_label)

        if cfg.test_num != -1:
            cfg.dataset_num = cfg.test_num

        if (len(test_list)) <= 0:
            valid_dataset = None
        else:
            valid_dataset = RealDataset_CvHMTB(cfg, test_list, test_label)

        return training_dataset, valid_dataset
