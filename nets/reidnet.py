from collections import OrderedDict

import torch
from reid.utils.serialization import load_checkpoint
from reid.utils import to_torch
from reid import models
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore")


# only accept 256, 128
class reidnet(torch.nn.Module):
    def __init__(self):
        super(reidnet, self).__init__()

        self.model = models.create('resnet50', num_features=1024, dropout=0, num_classes=751)
        checkpoint = load_checkpoint('models/reid_checkpoint.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.train()

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=0)
        output = self.model(x, 'pool5')
        return output

    def loadmodel(self, filepath):
        state = torch.load(filepath)
        process_dict = OrderedDict()
        for key in state['state_dict_reid'].keys():
            if key.startswith('module.'):
                process_dict[key[7:]] = state['state_dict_reid'][key]
            elif 'Mixed_6' in key:
                continue
            else:
                process_dict[key] = state['state_dict_reid'][key]
        self.load_state_dict(process_dict)
        print('Load reidnet all parameter from: ', filepath)
