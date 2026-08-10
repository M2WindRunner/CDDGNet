import torch
import torch.nn as nn
import torch.nn.functional as F


class DELFLoss(nn.Module):
    def __init__(self, tau=10.0, max_kmeans_samples=2048, kmeans_iters=6, eps=1e-6):
        super().__init__()
        self.tau = tau
        self.max_kmeans_samples = max_kmeans_samples
        self.kmeans_iters = kmeans_iters
        self.eps = eps

    def _fit_kmeans2(self, x):
        n = x.shape[0]
        if n < 2:
            return None

        if n > self.max_kmeans_samples:
            idx = torch.randperm(n, device=x.device)[:self.max_kmeans_samples]
            samples = x[idx]
        else:
            samples = x

        m = samples.shape[0]
        init_ids = torch.randperm(m, device=x.device)[:2]
        centers = samples[init_ids].clone()

        for _ in range(self.kmeans_iters):
            dist = torch.norm(samples[:, None, :] - centers[None, :, :], p=2, dim=2)
            assign = torch.argmin(dist, dim=1)
            for k in range(2):
                mask = assign == k
                if mask.any():
                    centers[k] = samples[mask].mean(dim=0)

        return centers

    def forward(self, feat_map, logits, labels):
        if labels.dim() == 4:
            labels = labels.squeeze(1)

        if labels.shape[-2:] != feat_map.shape[-2:]:
            labels = F.interpolate(labels.unsqueeze(1).float(), size=feat_map.shape[-2:], mode="nearest")
            labels = labels.squeeze(1).long()

        if logits.shape[-2:] != feat_map.shape[-2:]:
            logits = F.interpolate(logits, size=feat_map.shape[-2:], mode="bilinear", align_corners=True)

        preds = torch.argmax(logits, dim=1)

        b, c, h, w = feat_map.shape
        feat_flat = feat_map.permute(0, 2, 3, 1).reshape(b, h * w, c)
        loss_list = []

        for i in range(b):
            gt = labels[i].reshape(-1)
            pred = preds[i].reshape(-1)
            feats = feat_flat[i]

            change_mask = gt == 1
            invariant_mask = gt == 0
            misclassified_mask = pred != gt

            if change_mask.sum() < 2 or invariant_mask.sum() < 2:
                continue

            invariant_feats = feats[invariant_mask]
            centers = self._fit_kmeans2(invariant_feats)
            if centers is None:
                continue

            change_center = feats[change_mask].mean(dim=0)
            center_to_change = torch.norm(centers - change_center[None, :], p=2, dim=1)
            ambiguous_cluster = torch.argmin(center_to_change)

            inv_dist = torch.norm(invariant_feats[:, None, :] - centers[None, :, :], p=2, dim=2)
            inv_assign = torch.argmin(inv_dist, dim=1)
            inv_indices = torch.where(invariant_mask)[0]
            ambiguous_domain_mask = torch.zeros_like(invariant_mask, dtype=torch.bool)
            ambiguous_domain_mask[inv_indices[inv_assign == ambiguous_cluster]] = True

            pb_mask = ambiguous_domain_mask & misclassified_mask
            if pb_mask.sum() == 0:
                continue

            pb_center = feats[pb_mask].mean(dim=0)
            invariant_center = feats[invariant_mask].mean(dim=0)
            if torch.norm(pb_center - change_center, p=2) >= torch.norm(pb_center - invariant_center, p=2):
                continue

            change_map = change_mask.view(1, 1, h, w).float()
            neighbor_mask = (F.max_pool2d(change_map, kernel_size=3, stride=1, padding=1) > 0).view(-1)
            pp_mask = neighbor_mask & invariant_mask
            if pp_mask.sum() == 0:
                pp_mask = invariant_mask

            pp_center = feats[pp_mask].mean(dim=0)
            pb_feats = feats[pb_mask]
            l2_dist = torch.norm(pb_feats - pp_center[None, :], p=2, dim=1)
            scaled_dist = torch.clamp(l2_dist / (self.tau + self.eps), max=50.0)
            loss_de = torch.exp(scaled_dist).mean()
            loss_de = torch.clamp(loss_de, max=10.0)
            loss_list.append(loss_de)

        if len(loss_list) == 0:
            return feat_map.new_tensor(0.0)

        return torch.clamp(torch.stack(loss_list).mean(), max=10.0)
