import os
import math
import numpy as np
import cv2
from torchvision.utils import make_grid

second_colormap = [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0], [255, 0, 0]]


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    return img_np.astype(out_type)


def Index2Color(pred, cmap=second_colormap):
    colormap = np.asarray(cmap, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def save_scdimg(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.cvtColor(np.squeeze(img, axis=0), cv2.COLOR_RGB2BGR))


def save_feat(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, cv2.applyColorMap(cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_JET))


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))
