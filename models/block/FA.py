import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyAttentionModule(nn.Module):
    def __init__(self, in_channels, attention_size=8):
        super(FrequencyAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.attention_size = attention_size

        self.extract = nn.Conv2d(self.in_channels, self.in_channels, (1, 1), stride=(1, 1), bias=False)
        self.ln = LayerNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        pooled_height = min(H, self.attention_size)
        pooled_width = min(W, self.attention_size)
        pooled_x = F.adaptive_avg_pool2d(x, (pooled_height, pooled_width))

        energy_spectrum = torch.pow(pooled_x, 2)
        energy_spectrum_sum = torch.sum(
            energy_spectrum, dim=(1, 2, 3), keepdim=True
        ).clamp_min(1e-6)
        energy_spectrum_normalized = energy_spectrum / energy_spectrum_sum

        energy_spectrum_reshaped = energy_spectrum_normalized.view(B, C, -1)
        position_correlation = torch.einsum('bci, bcj -> bij', energy_spectrum_reshaped, energy_spectrum_reshaped)
        position_correlation = position_correlation / (C ** 0.5)
        position_correlation = torch.softmax(position_correlation, dim=-1)

        x_reshaped = pooled_x.reshape(B, C, -1)
        x_attention = torch.einsum('bci, bij -> bcj', x_reshaped, position_correlation)
        x_attention1 = x_attention.view(B, C, pooled_height, pooled_width)

        x_conx = self.extract(pooled_x)

        x_conx_reshaped = x_conx.view(B, C, -1)
        x_conx_position_correlation = torch.einsum('bci, bcj -> bij', x_conx_reshaped, x_conx_reshaped)
        x_conx_position_correlation = x_conx_position_correlation / (C ** 0.5)
        x_conx_position_correlation = torch.softmax(x_conx_position_correlation, dim=-1)

        x_conx_reshaped = x_conx.view(B, C, -1)
        x_conx_attention = torch.einsum('bci, bij -> bcj', x_conx_reshaped, x_conx_position_correlation)
        x_attention2 = x_conx_attention.view(B, C, pooled_height, pooled_width)

        gamma = 0.5

        attention = gamma * x_attention1 + (1 - gamma) * x_attention2
        if attention.shape[-2:] != (H, W):
            attention = F.interpolate(
                attention, size=(H, W), mode="bilinear", align_corners=False
            )
        out = x + attention
        out = self.ln(out)
        out = self.relu(out)

        return out


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x
