import torch
import torch.nn as nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu
from models.block.Drop import DropBlock
from models.block.Field import ASPP4
from models.block.dwtc import DWTC
from models.block.torch_wavelets import DWT_2D, IDWT_2D


class SiameseTopDownDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.context = ASPP4(c4)
        self.up4 = Conv3Relu(c4, c3)
        self.fuse3 = Conv3Relu(c3 * 2, c3)
        self.up3 = Conv3Relu(c3, c2)
        self.fuse2 = Conv3Relu(c2 * 2, c2)
        self.up2 = Conv3Relu(c2, c1)
        self.fuse1 = Conv3Relu(c1 * 2, c1)

    @staticmethod
    def resize(feature, reference):
        return F.interpolate(
            feature, size=reference.shape[-2:], mode="bilinear", align_corners=False
        )

    def decode_one(self, f1, f2, f3, f4):
        f4 = self.context(f4)
        f3 = self.fuse3(torch.cat((f3, self.up4(self.resize(f4, f3))), dim=1))
        f2 = self.fuse2(torch.cat((f2, self.up3(self.resize(f3, f2))), dim=1))
        f1 = self.fuse1(torch.cat((f1, self.up2(self.resize(f2, f1))), dim=1))
        return f1, f2, f3, f4

    def forward(self, features_a, features_b):
        return self.decode_one(*features_a), self.decode_one(*features_b)


class WaveletDifferenceFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")
        self.wavelet_refiners = nn.ModuleList(DWTC(channel * 4) for channel in channels)
        self.wavelet_fusers = nn.ModuleList(
            Conv3Relu(channel * 8, channel * 4) for channel in channels
        )
        self.output_fusers = nn.ModuleList(
            Conv3Relu(channel * 2, channel) for channel in channels
        )

    def forward(self, features_a, features_b):
        fused_differences = []
        raw_differences = []
        for feature_a, feature_b, refiner, wavelet_fuser, output_fuser in zip(
            features_a,
            features_b,
            self.wavelet_refiners,
            self.wavelet_fusers,
            self.output_fusers,
        ):
            spatial_difference = torch.abs(feature_a - feature_b)
            raw_differences.append(spatial_difference)
            original_height, original_width = feature_a.shape[-2:]
            pad_height = original_height % 2
            pad_width = original_width % 2
            if pad_height or pad_width:
                feature_a_for_dwt = F.pad(
                    feature_a, (0, pad_width, 0, pad_height), mode="replicate"
                )
                feature_b_for_dwt = F.pad(
                    feature_b, (0, pad_width, 0, pad_height), mode="replicate"
                )
            else:
                feature_a_for_dwt = feature_a
                feature_b_for_dwt = feature_b
            wavelet_a = refiner(self.dwt(feature_a_for_dwt))
            wavelet_b = refiner(self.dwt(feature_b_for_dwt))
            wavelet_difference = wavelet_fuser(torch.cat((wavelet_a, wavelet_b), dim=1))
            wavelet_difference = self.idwt(wavelet_difference)
            wavelet_difference = wavelet_difference[..., :original_height, :original_width]
            fused_differences.append(
                output_fuser(torch.cat((spatial_difference, wavelet_difference), dim=1))
            )
        return tuple(fused_differences), tuple(raw_differences)


class WeightedMultiScaleFusion(nn.Module):
    def __init__(self, channels, output_channels):
        super().__init__()
        self.projections = nn.ModuleList(
            Conv3Relu(channel, output_channels) for channel in channels
        )
        self.weights = nn.Parameter(torch.ones(len(channels), dtype=torch.float32))
        self.output = Conv3Relu(output_channels, output_channels)

    def forward(self, features):
        target_size = features[0].shape[-2:]
        projected = [
            F.interpolate(
                projection(feature),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            for projection, feature in zip(self.projections, features)
        ]
        positive_weights = F.relu(self.weights)
        normalized_weights = positive_weights / (positive_weights.sum() + 1e-6)
        fused = sum(weight * feature for weight, feature in zip(normalized_weights, projected))
        return self.output(fused)


class FPNNeck(nn.Module):
    def __init__(self, inplanes, neck_name="fpn+aspp+fuse+drop", use_wavelet=True):
        super().__init__()
        self.channels = (inplanes, inplanes * 2, inplanes * 4, inplanes * 8)
        self.temporal_decoder = SiameseTopDownDecoder(self.channels)
        self.wavelet_fusion = (
            WaveletDifferenceFusion(self.channels) if use_wavelet else None
        )
        self.context = ASPP4(self.channels[-1])
        self.change_decoder = WeightedMultiScaleFusion(self.channels, inplanes)
        if "drop" in neck_name:
            self.high_level_drop = DropBlock(rate=0.05, size=3, step=30)
        else:
            self.high_level_drop = DropBlock(rate=0, size=0, step=0)

    def forward(self, multi_scale_features):
        features_a = multi_scale_features[:4]
        features_b = multi_scale_features[4:]
        decoded_a, decoded_b = self.temporal_decoder(features_a, features_b)

        decoded_a = list(decoded_a)
        decoded_b = list(decoded_b)
        dropped = self.high_level_drop(
            [decoded_a[2], decoded_a[3], decoded_b[2], decoded_b[3]]
        )
        decoded_a[2], decoded_a[3], decoded_b[2], decoded_b[3] = dropped

        if self.wavelet_fusion is not None:
            fused_differences, raw_differences = self.wavelet_fusion(decoded_a, decoded_b)
        else:
            raw_differences = tuple(
                torch.abs(feature_a - feature_b)
                for feature_a, feature_b in zip(decoded_a, decoded_b)
            )
            fused_differences = raw_differences
        fused_differences = list(fused_differences)
        fused_differences[-1] = self.context(fused_differences[-1])
        change = self.change_decoder(fused_differences)

        diff1 = raw_differences[0]
        auxiliary_maps = [
            F.interpolate(
                difference.mean(dim=1, keepdim=True),
                size=diff1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            for difference in raw_differences[1:]
        ]
        return change, diff1, *auxiliary_maps
