# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json
from MirroPadding import mirror_padding  # 确保MirroPadding.py在相同目录


class SegmentationDataset(Dataset):
    def __init__(self, images_txt_path, labels_txt_path,mode='Unet',custom_mean=None, custom_std=None, is_grayscale=True):
        super().__init__()
        self.mirror_padding = mirror_padding

        # 读取图像和标签路径
        with open(images_txt_path, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]

        with open(labels_txt_path, 'r') as f:
            self.label_paths = [line.strip() for line in f.readlines() if line.strip()]

        # 存储文件名映射
        self.label_filenames = [os.path.basename(path) for path in self.label_paths]

        '''
        使用的Unet还是U2net Unet需要对256*256 填充到448*448 这样输出才是260*260 
        U2net 输入输出尺寸不变 256*256填充到260*260 输出就是260*260
        '''
        self.mode = mode

        # 验证路径数量是否匹配
        if len(self.image_paths) != len(self.label_paths):
            raise ValueError(f"图像数量({len(self.image_paths)})和标签数量({len(self.label_paths)})不匹配")

        # 验证文件是否存在
        for img_path in self.image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"图像文件不存在: {img_path}")

        for lbl_path in self.label_paths:
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"标签文件不存在: {lbl_path}")

        self.is_grayscale = is_grayscale

        # 设置归一化参数
        if is_grayscale:
            self.mean = custom_mean if custom_mean is not None else 0.5
            self.std = custom_std if custom_std is not None else 0.5

            # 图像预处理转换 (单通道)
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std])
            ])
        else:
            self.mean = custom_mean if custom_mean is not None else [0.485, 0.456, 0.406]
            self.std = custom_std if custom_std is not None else [0.229, 0.224, 0.225]

            # 图像预处理转换 (三通道)
            self.image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
            print(f"使用归一化参数 - 均值: {self.mean}, 标准差: {self.std}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        if self.is_grayscale:
            image = Image.open(self.image_paths[idx]).convert('L')  # 转为灰度
        else:
            image = Image.open(self.image_paths[idx]).convert('RGB')  # 转为RGB

        # 读取标签并二值化
        label = Image.open(self.label_paths[idx]).convert('L')  # 标签总是单通道
        lbl_array = np.array(label)

        # 将标签二值化 (0或1)
        lbl_array = (lbl_array > 0).astype(np.int64)

        # 验证图像尺寸
        img_array = np.array(image)
        if img_array.shape[:2] != (512, 512):
            raise ValueError(
                f"图像尺寸不是512×512: {img_array.shape[:2]} "
                f"在文件 {self.image_paths[idx]}"
            )

        if lbl_array.shape[:2] != (512, 512):
            raise ValueError(
                f"标签尺寸不是512×512: {lbl_array.shape[:2]} "
                f"在文件 {self.label_paths[idx]}"
            )

        # 转换标签
        label_t = torch.from_numpy(lbl_array).long()

        # 将图像裁剪为4个256x256块
        image_blocks = self.crop_to_blocks(img_array)
        label_blocks = self.crop_to_blocks(lbl_array)

        # 对每个图像块进行镜像填充
        padded_blocks = []
        for block in image_blocks:
            # Unet
            if self.mode == 'Unet':
                padded_block = self.mirror_padding(block, pad_size=192)  # 要填充到448 送入模型才输出260 448-256=192
            # U2net
            else:
                padded_block = self.mirror_padding(block, pad_size=4)  # 要填充到260 送入模型才输出260 260-256=4
            padded_block = Image.fromarray(padded_block)
            padded_block = self.image_transform(padded_block)
            padded_blocks.append(padded_block)

        # 转换标签块为tensor
        label_tensors = [torch.from_numpy(block).long() for block in label_blocks]

        # 将列表转换为元组以便DataLoader处理
        padded_blocks = tuple(padded_blocks)
        label_tensors = tuple(label_tensors)

        label_filename = self.label_filenames[idx]
        return padded_blocks, label_tensors, label_t, label_filename

    def crop_to_blocks(self, image):
        """将512x512图像裁剪为4个256x256块"""
        blocks = [
            image[0:256, 0:256],  # 左上
            image[0:256, 256:512],  # 右上
            image[256:512, 0:256],  # 左下
            image[256:512, 256:512]  # 右下
        ]
        return blocks


def calculate_dataset_stats(images_txt_path, is_grayscale=True, max_samples=None):
    # 读取图像路径
    with open(images_txt_path, 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]

    if max_samples is not None and len(image_paths) > max_samples:
        image_paths = image_paths[:max_samples]
        print(f"使用前 {max_samples} 个样本计算统计值")

    # 初始化累加器
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    print("计算数据集统计值...")
    for path in tqdm(image_paths):
        if is_grayscale:
            img = Image.open(path).convert('L')
        else:
            img = Image.open(path).convert('RGB')

        img_array = np.array(img) / 255.0

        # 验证尺寸
        if img_array.shape[:2] != (512, 512):
            print(f"警告: 图像 {path} 尺寸为 {img_array.shape}, 但应为 (512, 512). 跳过...")
            continue

        if is_grayscale:
            pixel_sum += img_array.sum()
            pixel_sq_sum += (img_array ** 2).sum()
            pixel_count += 512 * 512
        else:
            pixel_sum += img_array.sum(axis=(0, 1))
            pixel_sq_sum += (img_array ** 2).sum(axis=(0, 1))
            pixel_count += 512 * 512 * 3

    # 计算均值和标准差
    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)

    # 如果是RGB图像，返回每个通道的均值和标准差
    if not is_grayscale:
        mean = list(mean)
        std = list(std)

    return mean, std


def get_data_loaders(data_dir,mode,batch_size=1, use_custom_stats=True, max_samples=500, is_grayscale=True):
    # 文件路径
    train_image_txt = os.path.join(data_dir, 'train_image.txt')
    train_label_txt = os.path.join(data_dir, 'train_label.txt')
    test_image_txt = os.path.join(data_dir, 'test_image.txt')
    test_label_txt = os.path.join(data_dir, 'test_label.txt')

    # 验证文件是否存在
    for path in [train_image_txt, train_label_txt, test_image_txt, test_label_txt]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")

    # 计算自定义统计值
    custom_mean, custom_std = None, None
    if use_custom_stats:
        print("计算训练集统计值...")
        custom_mean, custom_std = calculate_dataset_stats(train_image_txt, is_grayscale, max_samples)
        print(f"自定义统计值 - 均值: {custom_mean}, 标准差: {custom_std}")

        # 保存统计值
        stats_path = os.path.join(data_dir, 'dataset_stats.json')
        with open(stats_path, 'w') as f:
            json.dump({
                'mean': custom_mean,
                'std': custom_std,
                'is_grayscale': is_grayscale
            }, f)
        print(f"统计值已保存至: {stats_path}")

    # 创建数据集
    train_dataset = SegmentationDataset(
        train_image_txt,
        train_label_txt,
        mode,
        custom_mean,
        custom_std,
        is_grayscale
    )

    test_dataset = SegmentationDataset(
        test_image_txt,
        test_label_txt,
        mode,
        custom_mean,
        custom_std,
        is_grayscale
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, (custom_mean, custom_std)


def load_dataset_stats(data_dir):
    """从文件加载数据集统计值"""
    stats_path = os.path.join(data_dir, 'dataset_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            return stats['mean'], stats['std'], stats['is_grayscale']
    return None, None, None