import torch
from torch.utils import data
import torchvision as tv
import torchvision.transforms as transforms
import os

from utils.utils import get_matchid_from_fv_id_start_from_1
from utils.utils import crop_img_by_boxes, crop_img_by_boxes_without_norm


class VirtualDataset(data.Dataset):

    def __init__(self, cfg, input_dir_list, label_list):
        super(VirtualDataset, self).__init__()
        self.cfg = cfg
        self.img_pair_path = []
        self.label_list = label_list
        for dir_pair in input_dir_list:
            v0_list = sorted(os.listdir(dir_pair[0]))
            v1_list = sorted(os.listdir(dir_pair[1]))

            v0_list = [dir_pair[0] + '/' + v0 for v0 in v0_list]
            v1_list = [dir_pair[1] + '/' + v1 for v1 in v1_list]

            self.img_pair_path.extend(list(zip(v0_list, v1_list)))

        self.fv_dict = torch.load(os.path.join(cfg.label, 'fv.pth'))
        self.fv_sk_box = torch.load(os.path.join(cfg.label, 'fv_sk_box.pth'))
        # frame person id: lable
        # key is int
        self.fps = torch.load(os.path.join(cfg.label, 'fps.pth'))

        self.f_top_bbox_pid = torch.load(os.path.join(cfg.label, 'f_top_bbox_pid.pth'))

        if cfg.dataset_num != -1:
            self.img_pair_path = self.img_pair_path[:cfg.dataset_num]
            self.label_list = self.label_list[:cfg.dataset_num]
            # self.counter_list = self.counter_list[:cfg.dataset_num]

    def __len__(self):
        """
        Return the total number of samples
        """
        return len(self.label_list)

    def __getitem__(self, index):
        """
        :param index:
        :return:
            1. img1: [3, 768, 1024] [17, 255.0]
            2. img2: the same as img1
            3. img1_cropped: [n, 256, 128]
            4. img2_cropped:[n, 256, 128]
            5. sk1: [n, 3, 17] (skeleton coordinates from pifpaf)
            6. box1:[n, 5](x_min, y_min, x_max, y_max, conf)
            7. sk2: [n, 3, 17]
            8. box2: [n, 5](x_min, y_min, x_max, y_max, conf)
            9. label: [x, y, r] (gt label of camera x, y, r)
            10. frame_vide_pair: [ frame_id_of_img1, view_id_of_img1, frame_id_of_img2, view_id_of_img2]
            11. col_ids_list1: [id1_list]
            12. col_ids_list2: [id2_list]
            13. top_box: [n, 4]
            14. top_id: subject id of each top box
            15. view1_box: gt bbox of view1
            16. view1_ids: subject id of each view1
            17. view1_label: gt label of view1 (x, y, r)
            18. view2_box: gt bbox of view2
            19. view2_ids: subject id of each view2
            20. view2_label: gt label of view2 (x, y, r)

        """

        if self.cfg.channel == 1:
            img1 = tv.io.read_image(self.img_pair_path[index][0]).float()
            img2 = tv.io.read_image(self.img_pair_path[index][1]).float()
        else:
            img1 = tv.io.read_image(self.img_pair_path[index][0], tv.io.image.ImageReadMode.RGB).float()
            img2 = tv.io.read_image(self.img_pair_path[index][1], tv.io.image.ImageReadMode.RGB).float()

        frame_vid_pair = []

        path1 = self.img_pair_path[index][0]
        hor, img_path = path1.split('/')[-2], path1.split('/')[-1]
        view_id = int(hor[3])

        if 'mono' in self.cfg.model_name:
            frame_id = int(img_path.split('.')[0])
        else:
            frame_id = int(img_path.split('.')[0].split('_')[1])
        frame_vid_pair.append(frame_id)
        frame_vid_pair.append(view_id)

        path2 = self.img_pair_path[index][1]
        hor, img_path = path2.split('/')[-2], path2.split('/')[-1]
        view_id = int(hor[3])
        # if self.cfg.model_name == 'monoreid':
        if 'mono' in self.cfg.model_name:
            frame_id = int(img_path.split('.')[0])
        else:
            frame_id = int(img_path.split('.')[0].split('_')[1])
        frame_vid_pair.append(frame_id)
        frame_vid_pair.append(view_id)

        frame_vid_pair = torch.tensor(frame_vid_pair)

        label = torch.tensor(self.label_list[index])

        fv_str1 = "%s_%s" % (frame_vid_pair[0].item(), frame_vid_pair[1].item())
        fv_str2 = "%s_%s" % (frame_vid_pair[2].item(), frame_vid_pair[3].item())

        sk1, box1 = self.fv_sk_box[fv_str1]
        sk2, box2 = self.fv_sk_box[fv_str2]

        img1_cropped = crop_img_by_boxes(img1, box1)
        img2_cropped = crop_img_by_boxes(img2, box2)

        sk1 = torch.tensor(sk1)
        sk2 = torch.tensor(sk2)

        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)

        top_bbox_id = self.f_top_bbox_pid[str(frame_id)]
        top_box = torch.tensor([elem[0] for elem in top_bbox_id])
        top_id = torch.tensor([elem[1] for elem in top_bbox_id])

        _, col_ids_list1 = get_matchid_from_fv_id_start_from_1(box1, self.fv_dict, fv_str1)
        _, col_ids_list2 = get_matchid_from_fv_id_start_from_1(box2, self.fv_dict, fv_str2)

        col_ids_list1 = torch.tensor(col_ids_list1)
        col_ids_list2 = torch.tensor(col_ids_list2)

        view1_all_gt = self.fv_dict[fv_str1]
        view2_all_gt = self.fv_dict[fv_str2]

        view1_ids = torch.tensor([p_bbox[0] for p_bbox in view1_all_gt])
        view2_ids = torch.tensor([p_bbox[0] for p_bbox in view2_all_gt])

        view1_box = torch.tensor([p_bbox[1] for p_bbox in view1_all_gt])
        view2_box = torch.tensor([p_bbox[1] for p_bbox in view2_all_gt])

        frame_lable_dict = self.fps[frame_vid_pair[0].item()]

        view1_lable = torch.tensor([[float(frame_lable_dict[id.item()][0]), float(frame_lable_dict[id.item()][1]),
                                     float(frame_lable_dict[id.item()][2])] for id in view1_ids])
        view2_lable = torch.tensor([[float(frame_lable_dict[id.item()][0]), float(frame_lable_dict[id.item()][1]),
                                     float(frame_lable_dict[id.item()][2])] for id in view2_ids])

        return img1, img2, img1_cropped, img2_cropped, sk1, box1, sk2, box2, label, frame_vid_pair, col_ids_list1, col_ids_list2, \
            top_box, top_id, view1_box, view1_ids, view1_lable, view2_box, view2_ids, view2_lable
