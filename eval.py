import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as T
from models.main_model import EnsembleModel
from utils.dataloaders import get_eval_loaders
from utils.dataloaders import load_normalization_statistics
from utils.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
import torch.nn.functional as F
import utils.visual as Metrics
import cv2
from datetime import datetime


def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gpu_info()
    save_path, result_save_path = check_eval_dirs()

    save_results = SaveResult(result_save_path)
    save_results.prepare()

    opt.image_mean, opt.image_std = load_normalization_statistics(
        opt.ckp_paths, opt.normalization_dir
    )
    print("Evaluation image mean: {}".format(opt.image_mean))
    print("Evaluation image std: {}".format(opt.image_std))

    model = EnsembleModel(opt.ckp_paths, device, input_size=opt.input_size)

    if model.models_list[0].head2 is None:
        opt.dual_label = False
    else:
        opt.dual_label = True
    eval_loader = get_eval_loaders(opt)

    p, r, f1, miou, oa, avg_loss = eval_for_metric(
        model,
        eval_loader,
        tta=opt.tta,
        input_size=opt.input_size,
        dataset_name=opt.dataset_name,
        save_visuals=opt.save_visuals,
    )

    save_results.show(p, r, f1, miou, oa)
    print("F1-mean: {}".format(f1.mean()))
    print("mIOU-mean: {}".format(miou.mean()))


def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=256, dataset_name="eval", save_visuals=False):
    avg_loss = 0
    val_loss = torch.tensor([0])
    scale = ScaleInOutput(input_size)

    tn_fp_fn_tp = [np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
    if save_visuals:
        to_pilimg = T.ToPILImage()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_result_path = f'Experiment/{dataset_name}_{timestamp}'
        os.makedirs(test_result_path, exist_ok=True)

    model.eval()
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader)

        current_step = 0

        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...eval_loss: {}".format(avg_loss))
            batch_img1 = batch_img1.float().cuda(non_blocking=True)
            batch_img2 = batch_img2.float().cuda(non_blocking=True)
            batch_label1 = batch_label1.long().cuda(non_blocking=True)
            batch_label2 = batch_label2.long().cuda(non_blocking=True)

            if criterion is not None:
                batch_img1, batch_img2 = scale.scale_input((batch_img1, batch_img2))

            model_out = model(batch_img1, batch_img2, tta)
            preds = None
            if isinstance(model_out, (tuple, list)):
                if len(model_out) == 6:
                    outs, diff1, diff2, diff3, diff4, preds = model_out
                elif len(model_out) == 5:
                    outs, diff1, diff2, diff3, diff4 = model_out
                elif len(model_out) > 0:
                    outs = model_out[0]
                else:
                    raise RuntimeError("model output is empty")
            else:
                outs = model_out

            if preds is None:
                preds = outs[0] if isinstance(outs, tuple) else outs
            if not isinstance(outs, tuple):
                outs = (outs, outs)

            labels = (batch_label1, batch_label2)

            if criterion is not None:
                outs = scale.scale_output(outs)
                val_loss = criterion(outs, labels)
                val_loss = val_loss
                _, cd_pred1 = torch.max(outs[0], 1)
                _, cd_pred2 = torch.max(outs[1], 1)
            else:
                cd_pred1 = outs[0]
                cd_pred2 = outs[1]

            cd_preds = (cd_pred1, cd_pred2)

            avg_loss = (avg_loss * i + val_loss.cpu().detach().numpy()) / (i + 1)

            for j, (cd_pred, label) in enumerate(zip(cd_preds, labels)):
                tn = ((cd_pred == 0) & (label == 0)).int().sum().cpu().numpy()
                fp = ((cd_pred == 1) & (label == 0)).int().sum().cpu().numpy()
                fn = ((cd_pred == 0) & (label == 1)).int().sum().cpu().numpy()
                tp = ((cd_pred == 1) & (label == 1)).int().sum().cpu().numpy()
                assert tn + fp + fn + tp == np.prod(batch_label1.shape)

                tn_fp_fn_tp[j] += [tn, fp, fn, tp]

            if save_visuals:
                img_A = Metrics.tensor2img(batch_img1[0], out_type=np.uint8, min_max=(-1, 1))
                img_B = Metrics.tensor2img(batch_img2[0], out_type=np.uint8, min_max=(-1, 1))
                pred_cm = torch.round(cd_preds[0][0]).cpu().clone().float()
                gt_cm = torch.round(labels[0][0]).cpu().clone().float()
                pred_cm = to_pilimg(pred_cm)
                gt_cm = to_pilimg(gt_cm)

                pred_img_np = np.array(pred_cm)
                gt_img_np = np.array(gt_cm)

                if preds.dim() == 4:
                    prob_map = torch.sum(preds[0], dim=0)
                elif preds.dim() == 3:
                    prob_map = preds[0].float()
                else:
                    prob_map = preds.float()
                prob_map = prob_map.squeeze().cpu().numpy()
                denom = prob_map.max() - prob_map.min()
                heatmap = (prob_map - prob_map.min()) / (denom + 1e-8)
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                diff_img = np.zeros((pred_img_np.shape[0], pred_img_np.shape[1], 3), dtype=np.uint8)
                for x in range(pred_img_np.shape[1]):
                    for y in range(pred_img_np.shape[0]):
                        pred_pixel = pred_img_np[y, x]
                        gt_pixel = gt_img_np[y, x]

                        if np.all(pred_pixel == [0, 0, 0]) and np.all(gt_pixel == [0, 0, 0]):
                            diff_img[y, x] = [0, 0, 0]
                        elif np.all(pred_pixel == [255, 255, 255]) and np.all(gt_pixel == [255, 255, 255]):
                            diff_img[y, x] = [255, 255, 255]
                        elif np.all(pred_pixel == [255, 255, 255]) and np.all(gt_pixel == [0, 0, 0]):
                            diff_img[y, x] = [0, 0, 255]
                        elif np.all(pred_pixel == [0, 0, 0]) and np.all(gt_pixel == [255, 255, 255]):
                            diff_img[y, x] = [0, 255, 0]

                Metrics.save_img(
                    img_A, '{}/img_A_{}.png'.format(test_result_path, current_step))
                Metrics.save_img(
                    img_B, '{}/img_B_{}.png'.format(test_result_path, current_step))
                filename_pred_cm = '{}/img_pred_cm{}.png'.format(test_result_path, current_step)
                filename_gt_cm = '{}/img_gt_cm{}.png'.format(test_result_path, current_step)
                filename_diff_img = '{}/img_diff_{}.png'.format(test_result_path, current_step)
                pred_cm.save(filename_pred_cm)
                gt_cm.save(filename_gt_cm)
                cv2.imwrite(filename_diff_img, diff_img)

                filename_heatmap = '{}/heatmap_{}.png'.format(test_result_path, current_step)
                cv2.imwrite(filename_heatmap, heatmap)

                current_step += 1

    p, r, f1, miou, oa = compute_p_r_f1_miou_oa(tn_fp_fn_tp)

    return p, r, f1, miou, oa, avg_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Change Detection eval')

    parser.add_argument("--ckp-paths", type=str, nargs="+",
                        default=[])

    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--dataset-dir", type=str, default="")
    parser.add_argument("--normalization-dir", type=str, default="")
    parser.add_argument("--dataset-name", type=str, default="eval")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--tta", type=bool, default=False)
    parser.add_argument("--save-visuals", action="store_true")

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)
