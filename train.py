import os
import argparse
import random
import re
import numpy as np
import torch
from tqdm import tqdm

from eval import eval_for_metric
from losses.get_losses import SelectLoss
from losses.delf import DELFLoss
from losses.feature_contrastive import FeatureContrastiveLoss
from models.block.Drop import dropblock_step
from utils.dataloaders import get_loaders
from utils.dataloaders import compute_image_statistics
from utils.common import check_dirs, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
from models.main_model import ChangeDetection


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def save_checkpoint(model, path, opt, epoch, metric):
    torch.save({
        "model_state_dict": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "metric": float(metric),
        "model_config": {
            "backbone": opt.backbone,
            "frequency_strategy": opt.frequency_strategy,
            "neck": opt.neck,
            "head": opt.head,
            "dual_label": opt.dual_label,
        },
        "image_mean": np.asarray(opt.image_mean).tolist(),
        "image_std": np.asarray(opt.image_std).tolist(),
    }, path)


def find_best_checkpoint(path):
    if not path:
        return ""
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError("Pretrained path does not exist: {}".format(path))

    candidates = []
    for name in os.listdir(path):
        if not name.endswith(".pt"):
            continue
        match = re.search(r"_([-+]?[0-9]*\.?[0-9]+)\.pt$", name)
        score = float(match.group(1)) if match else float("-inf")
        candidates.append((score, os.path.getmtime(os.path.join(path, name)), name))
    if not candidates:
        raise FileNotFoundError("No .pt checkpoint found in {}".format(path))
    _, _, best_name = max(candidates)
    return os.path.join(path, best_name)


def get_delf_weight(epoch, step, total_steps, max_weight, warmup_epochs, ramp_epochs):
    if total_steps <= 0:
        return max_weight

    progress = epoch + (step + 1) / total_steps

    if progress < warmup_epochs:
        return 0.0

    if ramp_epochs <= 0:
        return max_weight

    ratio = (progress - warmup_epochs) / ramp_epochs
    ratio = max(0.0, min(1.0, ratio))
    return max_weight * ratio


def train(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    gpu_info()
    save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path = check_dirs(opt.run_name.strip())
    save_results = SaveResult(result_save_path)
    save_results.prepare()

    opt.image_mean, opt.image_std = compute_image_statistics(opt.dataset_dir)
    print("Image mean: {}".format(opt.image_mean))
    print("Image std: {}".format(opt.image_std))
    train_loader, val_loader = get_loaders(opt)
    scale = ScaleInOutput(opt.input_size)

    model = ChangeDetection(opt)
    if opt.pretrain:
        model.load_pretrained(find_best_checkpoint(opt.pretrain))
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    criterion = SelectLoss(opt.loss)
    val_criterion = SelectLoss(opt.loss)
    delf_criterion = DELFLoss(tau=opt.delf_tau, max_kmeans_samples=opt.delf_kmeans_samples).cuda()
    feature_criterion = FeatureContrastiveLoss(
        margin=opt.feature_margin,
        change_weight=opt.feature_change_weight,
        hard_negative_ratio=opt.feature_hard_ratio,
        outlier_trim_ratio=opt.feature_trim_ratio,
    ).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=0.01)
    if opt.pseudo_label:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate / 5, epochs=opt.epochs, up_rate=0)
    else:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs, up_rate=0)

    best_metric = 0
    train_avg_loss = 0
    train_dis_loss = 0
    train_feature_loss = 0
    total_bs = 16
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))

    optimizer.zero_grad()
    for epoch in range(opt.epochs):
        model.train()
        train_tbar = tqdm(train_loader)
        current_delf_weight = 0.0
        current_feature_weight = 0.0
        for i, (batch_img1, batch_img2, batch_label, batch_label2, _) in enumerate(train_tbar):
            train_tbar.set_description(
                "epoch {}, train_loss {:.6f}, de_loss {:.6f}, feat_loss {:.6f}".format(
                    epoch, train_avg_loss, train_dis_loss, train_feature_loss
                )
            )
            batch_img1 = batch_img1.float().cuda(non_blocking=True)
            batch_img2 = batch_img2.float().cuda(non_blocking=True)
            batch_label = batch_label.long().cuda(non_blocking=True)
            batch_label2 = batch_label2.long().cuda(non_blocking=True)
            input1_cp1, input1_cp2 = scale.scale_input((batch_img1, batch_img2))

            outs, diff1, diff2, diff3, diff4 = model(input1_cp1, input1_cp2)

            input1 = scale.scale_output(outs)

            input1_loss = criterion(input1, (batch_label,))
            input1_logits = input1[0] if isinstance(input1, tuple) else input1
            delf_loss = delf_criterion(diff1, input1_logits, batch_label)
            feature_loss = feature_criterion(diff1, batch_label)
            current_delf_weight = get_delf_weight(
                epoch,
                i,
                len(train_loader),
                opt.delf_lambda,
                opt.delf_warmup_epochs,
                opt.delf_ramp_epochs,
            )
            current_feature_weight = get_delf_weight(
                epoch,
                i,
                len(train_loader),
                opt.feature_lambda,
                opt.feature_warmup_epochs,
                opt.feature_ramp_epochs,
            )

            loss = (
                input1_loss
                + current_delf_weight * delf_loss
                + current_feature_weight * feature_loss
            )

            train_avg_loss = (train_avg_loss * i + loss.cpu().detach().numpy()) / (i + 1)
            train_dis_loss = (train_dis_loss * i + delf_loss.cpu().detach().numpy()) / (i + 1)
            train_feature_loss = (
                train_feature_loss * i + feature_loss.cpu().detach().numpy()
            ) / (i + 1)

            loss.backward()
            if ((i + 1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()

            del batch_img1, batch_img2, batch_label, batch_label2

        if len(train_loader) % accumulate_iter != 0:
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        dropblock_step(model)
        p, r, f1, miou, oa, val_avg_loss = eval_for_metric(
            model,
            val_loader,
            val_criterion,
            input_size=opt.input_size,
            save_visuals=opt.save_visuals,
        )

        refer_metric = f1
        underscore = "_"
        if refer_metric.mean() > best_metric:
            best_ckp_file = os.path.join(
                best_ckp_save_path,
                underscore.join([opt.backbone, opt.neck, opt.head, 'epoch',
                                 str(epoch), str(round(float(refer_metric.mean()), 5))]) + ".pt")
            save_checkpoint(model, best_ckp_file, opt, epoch, refer_metric.mean())
            best_metric = refer_metric.mean()

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        save_results.show(p, r, f1, miou, oa, refer_metric, best_metric, train_avg_loss, val_avg_loss, lr, epoch, train_dis_loss)

    return best_ckp_file


def set_randomness():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection train')
    parser.add_argument("--backbone", type=str, default="siamese_wavelet_resnet18")
    parser.add_argument("--frequency-strategy", type=str, default="all_stages")
    parser.add_argument("--neck", type=str, default="fpn+aspp+fuse+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")
    parser.add_argument("--run-name", type=str, default="")

    parser.add_argument("--pretrain", type=str,
                        default="")
    parser.add_argument("--cuda", type=str, default="1")
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--test-dir", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--delf-lambda", type=float, default=0.0)
    parser.add_argument("--delf-tau", type=float, default=10.0)
    parser.add_argument("--delf-kmeans-samples", type=int, default=4096)
    parser.add_argument("--delf-warmup-epochs", type=int, default=0)
    parser.add_argument("--delf-ramp-epochs", type=int, default=0)
    parser.add_argument("--feature-lambda", type=float, default=0.0)
    parser.add_argument("--feature-margin", type=float, default=1.0)
    parser.add_argument("--feature-change-weight", type=float, default=1.0)
    parser.add_argument("--feature-hard-ratio", type=float, default=0.25)
    parser.add_argument("--feature-trim-ratio", type=float, default=0.05)
    parser.add_argument("--feature-warmup-epochs", type=int, default=0)
    parser.add_argument("--feature-ramp-epochs", type=int, default=0)
    parser.add_argument("--color-jitter", action="store_true", default=False)
    parser.add_argument("--save-visuals", action="store_true")
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--pseudo-label", type=bool, default=False)

    opt = parser.parse_args()
    print(opt)
    set_randomness()
    train(opt)
