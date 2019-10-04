# coding=utf-8
import torch
import torch.nn as nn
import torch.functional as F
from torchsummary import summary
from collections import OrderedDict


class RNetBase(nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def loss_function(cls, bbox_loss, cls_loss):
        return cls_loss + 1 * bbox_loss

class ExtRNet(RNetBase):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3)),
            ("prelu1", nn.PReLU(num_parameters=28)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ("conv2", nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3)),
            ("prelu2", nn.PReLU(num_parameters=48)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ("conv3", nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2)),
            ("prelu3", nn.PReLU(num_parameters=64)),

            ("conv4", nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)),
            ("prelu4", nn.PReLU(num_parameters=32)),

            ("conv5", nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)),
            ("prelu5", nn.PReLU(num_parameters=64)),

            ("conv6", nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)),
            ("prelu6", nn.PReLU(num_parameters=64)),
        ]))

        self.linear = nn.Sequential(OrderedDict([
            ("lin4", nn.Linear(in_features=576, out_features=128)),
            ("prelu4", nn.PReLU(num_parameters=128))
        ]))

        self.bbox_reg_lin = nn.Linear(in_features=128, out_features=4)
        self.classify_lin = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.features(x)

        # flatten
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        bbox_reg = self.bbox_reg_lin(x)
        classification = self.classify_lin(x)

        return bbox_reg, classification

class RNet(RNetBase):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels=3, out_channels=28, kernel_size=3)),
            ("prelu1", nn.PReLU(num_parameters=28)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ("conv2", nn.Conv2d(in_channels=28, out_channels=48, kernel_size=3)),
            ("prelu2", nn.PReLU(num_parameters=48)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)),

            ("conv3", nn.Conv2d(in_channels=48, out_channels=64, kernel_size=2)),
            ("prelu3", nn.PReLU(num_parameters=64))
        ]))

        self.linear = nn.Sequential(OrderedDict([
            ("dropout1", nn.Dropout(0.25)),
            ("lin4", nn.Linear(in_features=576, out_features=128)),
            ("prelu4", nn.PReLU(num_parameters=128))
        ]))

        self.bbox_reg_lin = nn.Linear(in_features=128, out_features=4)
        self.classify_lin = nn.Linear(in_features=128, out_features=2)

    def forward(self, x):
        x = self.features(x)

        # flatten
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        bbox_reg = self.bbox_reg_lin(x)
        classification = self.classify_lin(x)

        return bbox_reg, classification

if __name__ == "__main__":
    model = ExtRNet()
    summary(model, input_size=(3, 48, 48))