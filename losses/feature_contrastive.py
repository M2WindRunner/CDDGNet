import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureContrastiveLoss(nn.Module):
    def __init__(
        self,
        margin=1.0,
        change_weight=1.0,
        hard_negative_ratio=0.25,
        outlier_trim_ratio=0.05,
        normalize_by_channels=True,
        label_threshold=0.5,
        eps=1e-6,
    ):
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be greater than 0")
        if change_weight < 0:
            raise ValueError("change_weight must be non-negative")
        if not 0.0 <= hard_negative_ratio <= 1.0:
            raise ValueError("hard_negative_ratio must be in [0, 1]")
        if not 0.0 <= outlier_trim_ratio < 1.0:
            raise ValueError("outlier_trim_ratio must be in [0, 1)")
        if not 0.0 <= label_threshold <= 1.0:
            raise ValueError("label_threshold must be in [0, 1]")

        self.margin = margin
        self.change_weight = change_weight
        self.hard_negative_ratio = hard_negative_ratio
        self.outlier_trim_ratio = outlier_trim_ratio
        self.normalize_by_channels = normalize_by_channels
        self.label_threshold = label_threshold
        self.eps = eps

    def _resize_labels(self, labels, spatial_size):
        if labels.dim() == 4:
            if labels.shape[1] != 1:
                raise ValueError("4D labels must have exactly one channel")
            labels = labels.squeeze(1)
        if labels.dim() != 3:
            raise ValueError("labels must have shape [B, H, W] or [B, 1, H, W]")

        if labels.shape[-2:] != spatial_size:
            label_map = labels.unsqueeze(1).float()
            is_downsampling = (
                spatial_size[0] <= labels.shape[-2]
                and spatial_size[1] <= labels.shape[-1]
            )
            if is_downsampling:
                label_map = F.interpolate(label_map, size=spatial_size, mode="area")
                labels = label_map.squeeze(1) >= self.label_threshold
            else:
                labels = F.interpolate(
                    label_map, size=spatial_size, mode="nearest"
                ).squeeze(1) >= 0.5
        return labels.long()

    def _select_unchanged(self, distances, mask):
        values = distances[mask]
        if values.numel() == 0 or self.hard_negative_ratio == 0:
            return values

        if self.outlier_trim_ratio > 0 and values.numel() > 1:
            trusted_count = max(1, int(values.numel() * (1.0 - self.outlier_trim_ratio)))
            values = torch.topk(
                values, k=trusted_count, largest=False, sorted=False
            ).values

        keep = max(1, int(values.numel() * self.hard_negative_ratio))
        return torch.topk(values, k=keep, largest=True, sorted=False).values

    def forward(self, diff_features, labels):
        if diff_features.dim() != 4:
            raise ValueError("diff_features must have shape [B, C, H, W]")

        labels = self._resize_labels(labels, diff_features.shape[-2:])
        distances = torch.linalg.vector_norm(diff_features, ord=2, dim=1)
        if self.normalize_by_channels:
            distances = distances / (diff_features.shape[1] ** 0.5 + self.eps)
        sample_losses = []

        for sample_idx in range(distances.shape[0]):
            sample_distance = distances[sample_idx]
            sample_label = labels[sample_idx]
            unchanged_mask = sample_label == 0
            change_mask = sample_label == 1
            terms = []

            unchanged_distances = self._select_unchanged(sample_distance, unchanged_mask)
            if unchanged_distances.numel() > 0:
                terms.append(
                    F.smooth_l1_loss(
                        unchanged_distances,
                        torch.zeros_like(unchanged_distances),
                        reduction="mean",
                    )
                )

            change_distances = sample_distance[change_mask]
            if change_distances.numel() > 0 and self.change_weight > 0:
                change_loss = F.relu(self.margin - change_distances).square().mean()
                terms.append(self.change_weight * change_loss)

            if terms:
                sample_losses.append(torch.stack(terms).sum())

        if not sample_losses:
            return diff_features.sum() * 0.0

        return torch.stack(sample_losses).mean()
