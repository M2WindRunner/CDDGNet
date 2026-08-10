import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block.FA import FrequencyAttentionModule


class DWTC(nn.Module):
    def __init__(self, in_channels):
        super(DWTC, self).__init__()

        self.in_channels = in_channels

        self.linear_1 = nn.Linear(self.in_channels, 4)
        self.extract = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3, 3), padding=(1, 1),
                                               stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))

        self.extract2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1, 1),
                                               stride=(1, 1), bias=False),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU(inplace=True))

        self.fa = FrequencyAttentionModule(in_channels // 4)

    def forward(self, x):
        list1 = torch.split(x, self.in_channels // 4, dim=1)
        list1 = list(list1)
        ll = list1[0]
        lh = list1[1]
        hl = list1[2]
        hh = list1[3]

        ll_fa = self.fa(ll)
        h = lh + hl + hh
        h_fa = self.fa(h)

        n_b, n_c, h, w = x.size()
        x = self.extract(x)
        feats = F.adaptive_avg_pool2d(x, (1, 1)).view((n_b, n_c))
        feats = F.relu(self.linear_1(feats))
        feats = torch.tanh(feats)
        feats = feats.view((n_b, 4, 1, 1))
        y = torch.split(feats, 1, dim=1)
        y = list(y)
        rll = torch.mul(y[0], ll_fa)
        ll = rll * h_fa + ll
        rlh = torch.mul(y[1], lh * h_fa)
        lh = rlh + lh
        rhl = torch.mul(y[2], hl * h_fa)
        hl = rhl + hl
        rhh = torch.mul(y[3], hh * h_fa)
        hh = rhh + hh
        x = self.extract2(torch.cat([ll, lh, hl, hh], dim=1))
        out = x

        return out
