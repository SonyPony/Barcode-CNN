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


if __name__ == "__main__":
    from PIL import Image

    model = PNet()
    weights = torch.load("model/weight/pnet_model_v2.pth")
    model.eval()

    im = Image.open("../../sample/office5.jpg")
    print(im)