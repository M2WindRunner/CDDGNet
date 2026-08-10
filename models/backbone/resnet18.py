import torch
import torch.nn as nn

from models.block.fft import CFAM


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


def make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.BatchNorm2d, channel_first=True):
    return nn.Sequential(
        (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
        nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
        (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
        norm_layer(out_dim),
    )


class SiameseWaveletResNet18(nn.Module):
    FREQUENCY_STAGE_OPTIONS = {
        "all_stages": (1, 2, 3, 4),
        "middle_stages": (2, 3),
        "semantic_stage": (3,),
        "no_frequency": (),
    }

    def __init__(self, frequency_strategy="all_stages", layers=(2, 2, 2, 2)):
        super().__init__()
        if frequency_strategy not in self.FREQUENCY_STAGE_OPTIONS:
            raise ValueError(
                "frequency_strategy must be one of {}".format(
                    tuple(self.FREQUENCY_STAGE_OPTIONS)
                )
            )
        self.frequency_strategy = frequency_strategy
        self.frequency_stages = set(self.FREQUENCY_STAGE_OPTIONS[frequency_strategy])

        self.expansion = 1
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block = BasicBlock
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, downdown=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, downdown=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, downdown=True)

        self.dwt1 = CFAM(64) if 1 in self.frequency_stages else None
        self.dwt2 = CFAM(128) if 2 in self.frequency_stages else None
        self.dwt3 = CFAM(256) if 3 in self.frequency_stages else None
        self.dwt4 = CFAM(512) if 4 in self.frequency_stages else None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, downdown=None):
        downsample = None

        layers = []

        if downdown is not None:
            layers.append(make_downsample_v3(planes // 2, planes))

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * self.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        x1 = self.layer1(x)
        y1 = self.layer1(y)

        if self.dwt1 is not None:
            x1, y1 = self.dwt1(x1, y1)

        x2 = self.layer2(x1)
        y2 = self.layer2(y1)

        if self.dwt2 is not None:
            x2, y2 = self.dwt2(x2, y2)

        x3 = self.layer3(x2)
        y3 = self.layer3(y2)

        if self.dwt3 is not None:
            x3, y3 = self.dwt3(x3, y3)

        x4 = self.layer4(x3)
        y4 = self.layer4(y3)

        if self.dwt4 is not None:
            x4, y4 = self.dwt4(x4, y4)

        return x1, x2, x3, x4, y1, y2, y3, y4


ResNet = SiameseWaveletResNet18


def resnet18(pretrained=False, **kwargs):
    return SiameseWaveletResNet18(**kwargs)
