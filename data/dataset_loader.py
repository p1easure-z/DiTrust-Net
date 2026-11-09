import os

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, zoom
from torch.utils.data import Dataset


def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """
    弹性形变 (Elastic Deformation)
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[1:]  # (H, W)
    H, W = shape

    # 生成随机位移场
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # 使用 stack 堆叠坐标，生成标准 (2, H*W) 形状，后续 reshape 更自然
    indices = np.stack([y + dy, x + dx], axis=0).reshape(2, -1)

    # Image 插值
    transformed_img = np.zeros_like(image)
    for c in range(image.shape[0]):
        transformed_img[c] = map_coordinates(
            image[c], indices, order=3, mode='reflect').reshape(shape)

    # Mask 插值
    transformed_mask = np.zeros_like(mask)
    for c in range(mask.shape[0]):
        transformed_mask[c] = map_coordinates(
            mask[c], indices, order=0, mode='reflect').reshape(shape)

    return transformed_img, transformed_mask


def _crop_or_pad(array, out_h, out_w):
    _, h, w = array.shape
    if h > out_h:
        top = (h - out_h) // 2
        array = array[:, top:top + out_h, :]
    elif h < out_h:
        pad_top = (out_h - h) // 2
        pad_bottom = out_h - h - pad_top
        array = np.pad(array, ((0, 0), (pad_top, pad_bottom), (0, 0)), mode='constant')

    _, h, w = array.shape
    if w > out_w:
        left = (w - out_w) // 2
        array = array[:, :, left:left + out_w]
    elif w < out_w:
        pad_left = (out_w - w) // 2
        pad_right = out_w - w - pad_left
        array = np.pad(array, ((0, 0), (0, 0), (pad_left, pad_right)), mode='constant')
    return array


def random_scale_jitter(image, mask, scale_range=(0.92, 1.08)):
    scale = float(np.random.uniform(*scale_range))
    image_scaled = zoom(image, zoom=(1.0, scale, scale), order=1, mode='reflect')
    mask_scaled = zoom(mask, zoom=(1.0, scale, scale), order=0, mode='nearest')
    return _crop_or_pad(image_scaled, image.shape[1], image.shape[2]), _crop_or_pad(mask_scaled, mask.shape[1], mask.shape[2])


def random_background_cutout(image, mask, max_size=36):
    bg_mask = mask.sum(axis=0) == 0
    bg_indices = np.argwhere(bg_mask)
    if bg_indices.size == 0:
        return image

    cy, cx = bg_indices[np.random.randint(len(bg_indices))]
    patch = int(np.random.randint(12, max_size + 1))
    half = patch // 2
    y0, y1 = max(cy - half, 0), min(cy + half, image.shape[1])
    x0, x1 = max(cx - half, 0), min(cx + half, image.shape[2])
    image[:, y0:y1, x0:x1] = 0.0
    return image


def compute_case_sampling_weights(mask_paths, et_weight=3.0, tc_weight=2.0, wt_weight=1.0, bg_weight=0.35):
    weights = []
    for mask_path in mask_paths:
        mask = np.load(mask_path)
        if np.any(mask == 4):
            weights.append(et_weight)
        elif np.any(np.isin(mask, [1, 4])):
            weights.append(tc_weight)
        elif np.any(np.isin(mask, [1, 2, 4])):
            weights.append(wt_weight)
        else:
            weights.append(bg_weight)
    return np.asarray(weights, dtype=np.float32)


class BraTSDatasetLoader(Dataset):
    """
    BraTS 脑肿瘤分割数据集加载器
    """

    def __init__(self, img_paths, mask_paths, is_train=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.is_train = is_train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 加载数据 (H, W, 4) -> 范围 [0, 1]
        npimage = np.load(img_path).astype(np.float32)
        npmask = np.load(mask_path)
        npimage = np.nan_to_num(npimage, nan=0.0, posinf=0.0, neginf=0.0)

        # 强度增强 (Intensity Augmentation)
        if self.is_train:
            # Gamma 变换 (模拟磁场不均匀性和对比度差异)
            if np.random.rand() < 0.5:
                gamma = np.random.uniform(0.7, 1.5)
                npimage = np.clip(npimage, a_min=0.0, a_max=None)
                npimage = np.power(npimage + 1e-8, gamma)

        # Z-Score 归一化 (将 [0,1] 拉伸到零均值分布)
        mean = np.mean(npimage, axis=(0, 1), keepdims=True)
        std = np.std(npimage, axis=(0, 1), keepdims=True)
        npimage = (npimage - mean) / (std + 1e-8)
        npimage = np.nan_to_num(npimage, nan=0.0, posinf=0.0, neginf=0.0)

        # Intensity Shift (模拟信号漂移)
        if self.is_train:
            if np.random.rand() < 0.2:
                shift_val = np.random.uniform(-0.1, 0.1)
                npimage = npimage + shift_val

        # 维度变换 (H, W, C) -> (C, H, W)
        npimage = npimage.transpose((2, 0, 1))

        # 标签处理 (One-hot) - 需在空间变换前准备好
        wt_label = np.zeros_like(npmask)
        wt_label[np.isin(npmask, [1, 2, 4])] = 1.
        tc_label = np.zeros_like(npmask)
        tc_label[np.isin(npmask, [1, 4])] = 1.
        et_label = np.zeros_like(npmask)
        et_label[npmask == 4] = 1.
        nplabel = np.stack([wt_label, tc_label, et_label], axis=0).astype("float32")  # (3, H, W)

        # 空间增强 (Spatial Augmentation)
        if self.is_train:
            # 弹性形变
            if np.random.rand() < 0.3:
                npimage, nplabel = elastic_transform(npimage, nplabel, alpha=15, sigma=3)

            if np.random.rand() < 0.35:
                npimage, nplabel = random_scale_jitter(npimage, nplabel)

            # 几何变换
            if np.random.rand() < 0.5:
                npimage = np.flip(npimage, axis=2).copy()
                nplabel = np.flip(nplabel, axis=2).copy()
            if np.random.rand() < 0.5:
                npimage = np.flip(npimage, axis=1).copy()
                nplabel = np.flip(nplabel, axis=1).copy()
            if np.random.rand() < 0.5:
                k = np.random.randint(1, 4)
                npimage = np.rot90(npimage, k, axes=(1, 2)).copy()
                nplabel = np.rot90(nplabel, k, axes=(1, 2)).copy()

            if np.random.rand() < 0.25:
                npimage = random_background_cutout(npimage, nplabel)

        npimage = np.nan_to_num(npimage, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
        nplabel = np.nan_to_num(nplabel, nan=0.0, posinf=1.0, neginf=0.0).astype("float32")
        filename = os.path.basename(img_path).split('.')[0]

        return npimage, nplabel, filename
