import torch
import torch.nn as nn
import torch.nn.functional as F
from models.block.FA import FrequencyAttentionModule
from models.block.torch_wavelets import DWT_2D, IDWT_2D


class CFAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")
        self.fa = FrequencyAttentionModule(in_channels)
        self.theta_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.extract = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(inplace=True),
        )
        nn.init.zeros_(self.theta_conv.weight)
        nn.init.constant_(self.theta_conv.bias, -2.0)

    def forward(self, feature1, feature2):
        original_height, original_width = feature1.shape[-2:]
        pad_height = original_height % 2
        pad_width = original_width % 2
        if pad_height or pad_width:
            feature1_padded = F.pad(
                feature1, (0, pad_width, 0, pad_height), mode="replicate"
            )
            feature2_padded = F.pad(
                feature2, (0, pad_width, 0, pad_height), mode="replicate"
            )
        else:
            feature1_padded = feature1
            feature2_padded = feature2
        bands1 = list(torch.chunk(self.dwt(feature1_padded), 4, dim=1))
        bands2 = list(torch.chunk(self.dwt(feature2_padded), 4, dim=1))

        bands1 = [self.fa(band) for band in bands1]
        bands2 = [self.fa(band) for band in bands2]

        low_difference = torch.abs(bands1[0] - bands2[0])
        change_gate = torch.sigmoid(self.theta_conv(low_difference))
        for band_index in range(1, 4):
            bands1[band_index] = bands1[band_index] * change_gate
            bands2[band_index] = bands2[band_index] * change_gate

        enhanced1 = self.idwt(self.extract(torch.cat(bands1, dim=1)))
        enhanced2 = self.idwt(self.extract(torch.cat(bands2, dim=1)))
        enhanced1 = enhanced1[..., :original_height, :original_width]
        enhanced2 = enhanced2[..., :original_height, :original_width]
        return feature1 + enhanced1, feature2 + enhanced2
