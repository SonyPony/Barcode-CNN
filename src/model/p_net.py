# coding=utf-8
import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict


class PNetBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def loss_function(cls, bbox_loss, cls_loss):
        return cls_loss + 0.5 * bbox_loss


class PNet(PNetBase):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)),
            ("prelu1", nn.PReLU(num_parameters=10)),
            ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),

            ("conv2", nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3)),
            ("prelu2", nn.PReLU(num_parameters=16)),

            ("conv3", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)),
            ("prelu3", nn.PReLU(num_parameters=32))
        ]))

        self.bbox_reg_conv = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.classify_conv = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)

    def forward(self, x):
        x = self.features(x)

        bbox_reg = self.bbox_reg_conv(x)
        classification = self.classify_conv(x)

        return bbox_reg, classification

class ExtPNet(PNetBase):
    def __init__(self):
        super().__init__()

        # TODO activation
        self.aconv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, dilation=4)
        self.prelu1 = nn.PReLU(num_parameters=16)

        self.aconv2 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, dilation=2)
        self.prelu2 = nn.PReLU(num_parameters=3)
        self.pnet = PNet()

    def forward(self, x):
        x = self.aconv1(x)
        x = self.prelu1(x)
        x = self.aconv2(x)
        x = self.prelu2(x)
        x = self.pnet(x)

        return x

class ExtPnetA3(PNetBase):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)),
            ("prelu1", nn.PReLU(num_parameters=10)),
            ("bn1", nn.BatchNorm2d(num_features=10)),

            ("aconv1", nn.Conv2d(in_channels=10, out_channels=14, kernel_size=3, dilation=2)),
            ("aprelu1", nn.PReLU(num_parameters=14)),
            ("bn2", nn.BatchNorm2d(num_features=14)),

            ("aconv2", nn.Conv2d(in_channels=14, out_channels=16, kernel_size=3, dilation=2)),
            ("aprelu2", nn.PReLU(num_parameters=16)),
            ("bn3", nn.BatchNorm2d(num_features=16)),

            ("aconv3", nn.Conv2d(in_channels=16, out_channels=10, kernel_size=3, dilation=2)),
            ("aprelu3", nn.PReLU(num_parameters=10)),
            ("bn4", nn.BatchNorm2d(num_features=10)),

            ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),

            ("conv2", nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3)),
            ("prelu2", nn.PReLU(num_parameters=16)),
            ("bn5", nn.BatchNorm2d(num_features=16)),

            ("conv3", nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)),
            ("prelu3", nn.PReLU(num_parameters=32)),
            ("bn6", nn.BatchNorm2d(num_features=32)),
            #("drop1", nn.Dropout2d(p=0.5))
        ]))

        self.bbox_reg_conv = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.classify_conv = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=1)

    def forward(self, x):
        x = self.features(x)

        bbox_reg = self.bbox_reg_conv(x)
        classification = self.classify_conv(x)

        return bbox_reg, classification
