# coding=utf-8
import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict


class ONetBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def loss_function(cls, bbox_loss, cls_loss):
        return cls_loss + 0.5 * bbox_loss


class ONet(ONetBase):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)),
            ("prelu1", nn.PReLU(num_parameters=32)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ("conv2", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)),
            ("prelu2", nn.PReLU(num_parameters=64)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ("conv3", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ("prelu3", nn.PReLU(num_parameters=64)),
            ("pool3", nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)),

            ("conv4", nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2)),
            ("prelu4", nn.PReLU(num_parameters=128))
        ]))

        self.linear = nn.Sequential(OrderedDict([
            ("lin5", nn.Linear(in_features=1152, out_features=256)),
            ("dropout1", nn.Dropout(0.25)),
            ("prelu5", nn.PReLU(num_parameters=256))
        ]))

        self.bbox_reg_lin = nn.Linear(in_features=256, out_features=4)
        self.classify_lin = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        bbox_reg = self.bbox_reg_lin(x)
        classification = self.classify_lin(x)

        return bbox_reg, classification
