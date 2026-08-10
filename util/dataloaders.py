import os
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np

from utils import transforms as tr


def get_loaders(opt):
    train_dataset = CDDloader(opt, 'train', aug=True)
    val_dataset = CDDloader(opt, 'val', aug=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers,
                                               pin_memory=True,
                                               worker_init_fn=seed_worker,
                                               generator=make_generator())
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers,
                                             pin_memory=True,
                                             worker_init_fn=seed_worker,
                                             generator=make_generator()
                                             )
    return train_loader, val_loader


def get_eval_loaders(opt):
    dataset_name = "test"
    print("using dataset: {} set".format(dataset_name))
    eval_dataset = CDDloader(opt, dataset_name, aug=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=opt.num_workers,
                                              pin_memory=True,
                                              worker_init_fn=seed_worker,
                                              generator=make_generator())
    return eval_loader


def get_infer_loaders(opt):
    infer_datast = CDDloadImageOnly(opt, 'test', aug=False)
    infer_loader = torch.utils.data.DataLoader(infer_datast,
                                               batch_size=opt.batch_size,
                                               shuffle=False,
                                               num_workers=opt.num_workers,
                                               pin_memory=True,
                                               worker_init_fn=seed_worker,
                                               generator=make_generator())
    return infer_loader


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)


def make_generator():
    generator = torch.Generator()
    generator.manual_seed(42)
    return generator


def compute_image_statistics(dataset_dir, max_samples=None):
    image_paths = []
    for phase in ('train',):
        for view in ('t1', 't2'):
            image_paths.extend(
                os.path.join(dataset_dir, phase, view, name)
                for name in sorted(os.listdir(os.path.join(dataset_dir, phase, view)))
                if is_img(name)
            )
    if max_samples is not None:
        image_paths = image_paths[:max_samples]
    if not image_paths:
        raise ValueError('No training images found for normalization statistics')

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sum_sq = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for path in image_paths:
        image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
        pixels = image.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_sum_sq += np.square(pixels).sum(axis=0)
        pixel_count += pixels.shape[0]

    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sum_sq / pixel_count - np.square(mean), 1e-6)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def load_normalization_statistics(checkpoint_paths, fallback_dir):
    if isinstance(checkpoint_paths, str):
        checkpoint_paths = [checkpoint_paths]
    if not checkpoint_paths:
        raise ValueError("need at least one checkpoint path for normalization")

    checkpoint_path = checkpoint_paths[0]
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            mean = checkpoint.get("image_mean")
            std = checkpoint.get("image_std")
            if mean is not None and std is not None:
                mean = np.asarray(mean, dtype=np.float32)
                std = np.asarray(std, dtype=np.float32)
                if mean.shape == (3,) and std.shape == (3,):
                    print("Using normalization statistics stored in checkpoint: {}".format(checkpoint_path))
                    return mean, std

    print(
        "Checkpoint does not contain normalization statistics; "
        "computing them from source training set: {}".format(fallback_dir)
    )
    return compute_image_statistics(fallback_dir)


class CDDloader(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        if phase == 'train':
            self.data_dir = str(opt.dataset_dir)
        if phase == 'val':
            self.data_dir = str(opt.test_dir)
        if phase == 'test':
            self.data_dir = str(opt.dataset_dir)
        self.dual_label = opt.dual_label
        self.phase = str(phase)
        self.aug = aug
        self.image_mean = getattr(opt, 'image_mean', None)
        self.image_std = getattr(opt, 'image_std', None)
        self.color_jitter = getattr(opt, 'color_jitter', False)
        self.input_size = getattr(opt, 'input_size', 256)
        self.transform = tr.build_transforms(
            mean=self.image_mean,
            std=self.image_std,
            color_jitter=self.color_jitter,
            input_size=self.input_size,
        )[0 if self.aug else 1]
        self.names = validate_dataset_files(self.data_dir, self.phase, self.dual_label)

    def __getitem__(self, index):
        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, self.phase, 't1', name)).convert('RGB')
        img2 = Image.open(os.path.join(self.data_dir, self.phase, 't2', name)).convert('RGB')
        label_name = resolve_label_name(name)
        label = Image.open(os.path.join(self.data_dir, self.phase, 'label', label_name)).convert('L')

        if self.dual_label:
            label2 = Image.open(os.path.join(self.data_dir, self.phase, 'label2', label_name)).convert('L')
        else:
            label2 = label

        if img1.size != img2.size or img1.size != label.size or label.size != label2.size:
            raise ValueError('t1, t2 and label sizes must match for {}'.format(name))
        img1, img2, label, label2 = self.transform([img1, img2, label, label2])

        return img1, img2, label, label2, name

    def __len__(self):
        return len(self.names)


def is_img(name):
    return os.path.splitext(name)[1].lower() in {'.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff'}


def resolve_label_name(image_name):
    stem, extension = os.path.splitext(image_name)
    return stem + '.png' if extension.lower() in {'.tif', '.tiff'} else image_name


def validate_dataset_files(data_dir, phase, dual_label=False, require_labels=True):
    t1_dir = os.path.join(data_dir, phase, 't1')
    t2_dir = os.path.join(data_dir, phase, 't2')
    label_dir = os.path.join(data_dir, phase, 'label')
    required_dirs = [t1_dir, t2_dir]
    if require_labels:
        required_dirs.append(label_dir)
    if require_labels and dual_label:
        required_dirs.append(os.path.join(data_dir, phase, 'label2'))
    for directory in required_dirs:
        if not os.path.isdir(directory):
            raise FileNotFoundError('Missing dataset directory: {}'.format(directory))

    names = sorted(name for name in os.listdir(t1_dir) if is_img(name))
    missing = []
    for name in names:
        label_name = resolve_label_name(name)
        paths = [os.path.join(t2_dir, name)]
        if require_labels:
            paths.append(os.path.join(label_dir, label_name))
            if dual_label:
                paths.append(os.path.join(data_dir, phase, 'label2', label_name))
        missing.extend(path for path in paths if not os.path.isfile(path))
    if missing:
        preview = '\n'.join(missing[:10])
        raise FileNotFoundError('Dataset files are not fully paired:\n{}'.format(preview))
    return names


class CDDloadImageOnly(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.phase = str(phase)
        self.aug = aug
        self.image_mean = getattr(opt, 'image_mean', None)
        self.image_std = getattr(opt, 'image_std', None)
        self.input_size = getattr(opt, 'input_size', 256)
        self.names = validate_dataset_files(
            self.data_dir, self.phase, dual_label=False, require_labels=False
        )
        self.inference_transforms = tr.build_transforms(
            mean=self.image_mean,
            std=self.image_std,
            color_jitter=False,
            input_size=self.input_size,
        )[2]

    def __getitem__(self, index):
        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, self.phase, 't1', name)).convert('RGB')
        img2 = Image.open(os.path.join(self.data_dir, self.phase, 't2', name)).convert('RGB')

        img1, img2 = self.inference_transforms([img1, img2])

        return img1, img2, name
