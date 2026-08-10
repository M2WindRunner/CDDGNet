import os
from copy import deepcopy
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.resnet18 import SiameseWaveletResNet18
from models.head.FCN import FCNHead
from models.neck.FPN import FPNNeck

from collections import OrderedDict

from utils.common import ScaleInOutput


class ChangeDetection(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.inplanes = 64
        self.dl = opt.dual_label
        self.frequency_strategy = getattr(opt, "frequency_strategy", "all_stages")
        self._create_backbone(opt.backbone)
        self._create_neck(opt.neck)
        self._create_heads(opt.head)

    def forward(self, xa, xb, tta=False):
        if not tta:
            return self.forward_once(xa, xb)
        else:
            return self.forward_tta(xa, xb)

    def forward_once(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."
        padded_height = ((h_input + 31) // 32) * 32
        padded_width = ((w_input + 31) // 32) * 32
        pad_height = padded_height - h_input
        pad_width = padded_width - w_input
        if pad_height or pad_width:
            xa = F.pad(xa, (0, pad_width, 0, pad_height), mode="replicate")
            xb = F.pad(xb, (0, pad_width, 0, pad_height), mode="replicate")
        features = self.backbone(xa, xb)
        change, diff1, diff2, diff3, diff4 = self.neck(features)
        out = self.head_forward(change, out_size=(padded_height, padded_width))
        if self.dl:
            out = tuple(item[..., :h_input, :w_input] for item in out)
        else:
            out = out[..., :h_input, :w_input]

        return out, diff1, diff2, diff3, diff4

    def forward_tta(self, xa, xb):
        _, _, h, w = xa.shape
        mutil_scales = [1.0, 0.75, 0.5]
        probabilities1 = []
        probabilities2 = []
        for single_scale in mutil_scales:
            single_scale = (
                max(32, int(round(h * single_scale / 32)) * 32),
                max(32, int(round(w * single_scale / 32)) * 32),
            )
            xa_size = F.interpolate(xa, single_scale, mode='bilinear', align_corners=True)
            xb_size = F.interpolate(xb, single_scale, mode='bilinear', align_corners=True)

            out_1 = self.forward_once(xa_size, xb_size)[0]

            if self.dl:
                out1_1, out1_2 = out_1
                probabilities1.append(
                    F.interpolate(F.softmax(out1_1, dim=1), size=(h, w), mode="bilinear", align_corners=False)
                )
                probabilities2.append(
                    F.interpolate(F.softmax(out1_2, dim=1), size=(h, w), mode="bilinear", align_corners=False)
                )

            else:
                out1_1 = out_1
                probabilities1.append(
                    F.interpolate(F.softmax(out1_1, dim=1), size=(h, w), mode="bilinear", align_corners=False)
                )

        mean1 = torch.stack(probabilities1).mean(dim=0)
        if self.dl:
            return mean1, torch.stack(probabilities2).mean(dim=0)
        return mean1

    def head_forward(self, change, out_size):
        out1 = F.interpolate(self.head1(change), size=out_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(change), size=out_size,
                             mode='bilinear', align_corners=True) if self.dl else None
        return (out1, out2) if self.dl else out1

    def _init_weight(self, pretrain=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            checkpoint = torch.load(pretrain, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, nn.DataParallel):
                checkpoint = checkpoint.module

            if isinstance(checkpoint, nn.Module):
                checkpoint = checkpoint.state_dict()
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint = checkpoint['state_dict']

            model_dict = self.state_dict()
            pretrained_dict = {
                key: value
                for key, value in checkpoint.items()
                if key in model_dict and model_dict[key].shape == value.shape
            }
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))

    def load_pretrained(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, nn.DataParallel):
            checkpoint = checkpoint.module
        if isinstance(checkpoint, nn.Module):
            checkpoint = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            checkpoint = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))

        model_dict = self.state_dict()
        pretrained_dict = {
            key.removeprefix('module.'): value
            for key, value in checkpoint.items()
            if key.removeprefix('module.') in model_dict
            and model_dict[key.removeprefix('module.')].shape == value.shape
        }
        self.load_state_dict(pretrained_dict, strict=False)
        print("=> Loaded {}/{} tensors from {}".format(
            len(pretrained_dict), len(model_dict), checkpoint_path
        ))
        return len(pretrained_dict)

    def _create_backbone(self, backbone):
        aliases = {
            "resnet50": "siamese_wavelet_resnet18",
            "siamese_wavelet_resnet18": "siamese_wavelet_resnet18",
        }
        if backbone in aliases:
            self.backbone = SiameseWaveletResNet18(
                frequency_strategy=getattr(self, "frequency_strategy", "all_stages")
            )
        else:
            raise ValueError(
                'error backbone, received: {}'.format(backbone)
            )

    def _create_neck(self, neck):
        if 'fpn' in neck:
            self.neck = FPNNeck(
                self.inplanes,
                neck,
                use_wavelet=self.frequency_strategy != "no_frequency",
            )

    def _select_head(self, head):
        if head == 'fcn':
            return FCNHead(self.inplanes, 2)

    def _create_heads(self, head):
        self.head1 = self._select_head(head)
        self.head2 = self._select_head(head) if self.dl else None


class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=512):
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = nn.ModuleList()
        assert isinstance(ckp_paths, list), "ckp_path must be a list: {}".format(ckp_paths)
        print("-" * 50 + "\n--Ensamble method: {}".format(method))
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                raise ValueError("ckp_path must be a checkpoint path: {}".format(ckp_path))
            print("--Load model: {}".format(ckp_path))
            checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                config = checkpoint.get("model_config", {})
                model_opt = SimpleNamespace(
                    backbone=config.get("backbone", "siamese_wavelet_resnet18"),
                    frequency_strategy=config.get("frequency_strategy", "all_stages"),
                    neck=config.get("neck", "fpn+aspp+fuse+drop"),
                    head=config.get("head", "fcn"),
                    dual_label=config.get("dual_label", False),
                    pretrain="",
                )
                model = ChangeDetection(model_opt)
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.to(device)
                model.eval()
                self.models_list.append(model)
            else:
                raise ValueError("checkpoint format not supported: {}".format(ckp_path))
        print("Total {} models loaded.".format(len(self.models_list)))
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        super().eval()
        return self

    def forward(self, xa, xb, tta=False):
        xa, xb = self.scale.scale_input((xa, xb))
        out1, out2 = 0, 0
        cd_pred1, cd_pred2 = None, None

        if len(self.models_list) == 1:
            outs, diff1, diff2, diff3, diff4 = self.models_list[0](xa, xb, tta)
            raw_output = outs
            if isinstance(outs, tuple):
                probabilities = outs if tta else tuple(F.softmax(out, dim=1) for out in outs)
                predictions = tuple(torch.argmax(probability, dim=1) for probability in probabilities)
                return predictions if self.models_list[0].dl else (
                    predictions[0], diff1, diff2, diff3, diff4, raw_output
                )
            probability = outs if tta else F.softmax(outs, dim=1)
            prediction = torch.argmax(probability, dim=1)
            return prediction, diff1, diff2, diff3, diff4, raw_output

        for i, model in enumerate(self.models_list):
            outs, diff1, diff2, diff3, diff4 = model(xa, xb, tta)
            test = outs
            if not isinstance(outs, tuple):
                outs = (outs, outs)
            if "avg" in self.method:
                if self.method == "avg2" and not tta:
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))
                out1 += outs[0]
                out2 += outs[1]
                _, cd_pred1 = torch.max(out1, 1)
                _, cd_pred2 = torch.max(out2, 1)
            elif self.method == "vote":
                _, out1_tmp = torch.max(outs[0], 1)
                _, out2_tmp = torch.max(outs[1], 1)
                out1 += out1_tmp
                out2 += out2_tmp
                cd_pred1 = out1 / (i + 1) >= 0.5
                cd_pred2 = out2 / (i + 1) >= 0.5

        if self.models_list[0].dl:
            return cd_pred1, cd_pred2
        else:
            return cd_pred1, diff1, diff2, diff3, diff4, test


class ModelEMA:
    def __init__(self, model, decay=0.96):
        self.shadow1 = deepcopy(model.module if self.is_parallel(model) else model).eval()
        self.decay = decay
        for p in self.shadow1.parameters():
            p.requires_grad_(False)

        self.shadow2 = deepcopy(self.shadow1)
        self.shadow3 = deepcopy(self.shadow1)
        self.update_count = 0

    def update(self, model):
        with torch.no_grad():
            msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
            for k, v in self.shadow1.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= self.decay
                    v += (1. - self.decay) * msd[k].detach()
            for k, v in self.shadow2.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.95
                    v += (1. - 0.95) * msd[k].detach()
            for k, v in self.shadow3.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= 0.94
                    v += (1. - 0.94) * msd[k].detach()
        self.update_count += 1

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


class ModelSWA:
    def __init__(self, total_epoch=300):
        self.update_count = 0
        self.epoch_threshold = int(total_epoch * 0.8)
        self.swa_model = None

    def update(self, model):
        if self.update_count >= self.epoch_threshold:
            with torch.no_grad():
                if self.swa_model is None:
                    self.swa_model = deepcopy(model.module) if self.is_parallel(model) else deepcopy(model)
                else:
                    msd = model.module.state_dict() if self.is_parallel(model) else model.state_dict()
                    for k, v in self.swa_model.state_dict().items():
                        if v.dtype.is_floating_point:
                            v *= (self.update_count - self.epoch_threshold)
                            v += msd[k].detach()
                            v /= (self.update_count - self.epoch_threshold + 1)
        self.update_count += 1

    def save(self, swa_ckp_dir_path):
        if self.update_count >= self.epoch_threshold:
            swa_file_path = os.path.join(swa_ckp_dir_path, "swa_{}_{}.pt".format(
                self.update_count - 1, self.update_count - self.epoch_threshold))
            torch.save(self.swa_model, swa_file_path)

    @staticmethod
    def is_parallel(model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
