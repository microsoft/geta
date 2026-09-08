"""
- Implementation of quantized vgg7-128 with batch normalization
- 7: 7 layers in total, 128: output channel after the first conv layer
- Create a quantized vgg7-128:
    model = QVGG7_BN([128, 128, "M", 256, 256, "M", 512, 512, "M"], True)
Reference:
[1] https://pyimagesearch.com/2021/05/22/minivggnet-going-deeper-with-cnns/
[2] https://github.com/Lornatang/VGG-PyTorch/blob/main/model.py

"""

from typing import cast

import torch
from torch import nn
from torch.nn import init  # for weight initialization.


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def _make_layers(vgg_cfg, batch_norm: bool = False):
    layers = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == "M":
            layers.append(
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            )  # kernel size, stride
        else:
            v = cast(int, v)
            if batch_norm:
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        v,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    )
                )
                layers.append(nn.BatchNorm2d(v))  # v represents num_feature
                layers.append(nn.ReLU(inplace=True))  # inplace operation
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels,
                        v,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=(1, 1),
                    )
                )
                layers.append(nn.ReLU(inplace=True))  # inplace operation
            in_channels = v

    layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
    return layers


class VGG7_BN(nn.Module):
    def __init__(
        self,
        vgg_cfg: list[int | str] = [128, 128, "M", 256, 256, "M", 512, 512, "M"],
        batch_norm: bool = True,
        num_classes: int = 10,
    ):
        super().__init__()
        self.features = _make_layers(vgg_cfg, batch_norm)

        # self.linear1 = nn.Linear(in_features=512, out_features=1024, bias=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=num_classes, bias=True),
        )

        self.apply(_weights_init)

    def forward(self, x, *args):  # Modified to accept arbitrary arguments
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def vgg7_bn(cfg=None, num_classes=None):
    return VGG7_BN(
        vgg_cfg=[128, 128, "M", 256, 256, "M", 512, 512, "M"],
        num_classes=num_classes or 10,
        batch_norm=True,
    )
