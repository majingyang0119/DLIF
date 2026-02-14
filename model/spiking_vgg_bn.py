import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer
import utils.conv_bilinear as Cb 

__all__ = [
    'SpikingVGGBN', 'spiking_vgg11_bn'
]

cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGGConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels,
                 neuron: callable, dropout: float,
                 bilinear: bool = False, **kwargs):
        super().__init__()
        self.bilinear = bilinear
        whether_bias = True

        # 线性卷积
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=whether_bias
        )

        # 二次卷积分支
        if self.bilinear:
            self.bilinear_layer = Cb.Conv2d_bilinear(
                in_channels, out_channels,
                kernel_size=3, padding=1, bias=whether_bias
            )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = neuron(**kwargs)
        self.drop = layer.Dropout(dropout)

    def forward(self, x):
        out_lin = self.conv(x)

        if self.bilinear:
            out_bilinear = self.bilinear_layer(x)
            out = out_lin + out_bilinear
        else:
            out = out_lin

        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        return out


class SpikingVGGBN(nn.Module):
    def __init__(self, vgg_name, neuron: callable = None, dropout=0.0,
                 num_classes=10, **kwargs):
        super(SpikingVGGBN, self).__init__()
        self.whether_bias = True
        self.init_channels = kwargs.get('c_in', 2)

        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout, neuron,
                                        bilinear=True,  **kwargs)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout, neuron,
                                        bilinear=True,  **kwargs)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout, neuron,
                                        bilinear=False, **kwargs)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout, neuron,
                                        bilinear=False, **kwargs)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout, neuron,
                                        bilinear=False, **kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg_list, dropout, neuron, bilinear: bool = False, **kwargs):
        layers_list = []
        for x in cfg_list:
            if x == 'M':
                layers_list.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers_list.append(
                    VGGConvBlock(self.init_channels, x,
                                 neuron=neuron, dropout=dropout,
                                 bilinear=bilinear, **kwargs)
                )
                self.init_channels = x
        return nn.Sequential(*layers_list)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)
        return out


def spiking_vgg11_bn(neuron: callable = None, num_classes=10,
                     neuron_dropout=0.0, **kwargs):
    return SpikingVGGBN('VGG11', neuron=neuron,
                        dropout=neuron_dropout,
                        num_classes=num_classes, **kwargs)
