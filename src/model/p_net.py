# coding=utf-8
import torch
import torch.nn as nn
import torch.functional as F
from collections import OrderedDict


class PNet(nn.Module):
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


class RNet(nn.Module):
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
    from PIL import Image

    rules = (
        ("features.conv4.weight", "linear.lin4.weight"),
        ("features.conv4.bias", "linear.lin4.bias"),
        ("features.prelu4.weight", "linear.prelu4.weight"),

        ("conv5_2.weight", "bbox_reg_lin.weight"),
        ("conv5_2.bias", "bbox_reg_lin.bias"),
        ("conv5_1.weight", "classify_lin.weight"),
        ("conv5_1.bias", "classify_lin.bias")
    )

    weights = torch.load("weight/rnet_model.pth")
    #tuple(map(print, weights.keys()))

    print(weights["conv5_1.bias"])
    print("-"*50)
    for rule in rules:
        weights[rule[1]] = weights[rule[0]]
        del weights[rule[0]]
    torch.save(weights, "weight/rnet_model_v2.pth")
    model = RNet()
    weights = torch.load("weight/rnet_model_v2.pth")
    model.load_state_dict(weights)

    exit(0)

    model = PNet()
    weights = torch.load("model/weight/pnet_model_v2.pth")
    model.eval()

    im = Image.open("../../sample/office5.jpg")
    print(im)