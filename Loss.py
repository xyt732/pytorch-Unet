# -*- coding: utf-8 -*-
import os
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, label_file="train_label.txt", w0=10, sigma=512 * 0.08, save_dir="./weight_maps", eps=1e-6):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.w0 = w0
        self.sigma = sigma
        self.sigma_sq = 2 * sigma * sigma
        self.save_dir = save_dir
        self.eps = eps  # 数值稳定系数

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 加载标签文件列表
        self.label_paths = []
        with open(label_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    self.label_paths.append(path)

        # 预计算并保存所有权重图
        self.precompute_all_weights()

        # 创建文件名到权重图的映射
        self.weights_cache = {}
        for full_path in self.label_paths:
            filename = os.path.basename(full_path)
            weight_path = os.path.join(save_dir, f"{filename.replace('.png', '')}.npy")
            if os.path.exists(weight_path):
                self.weights_cache[filename] = weight_path

    def precompute_all_weights(self):
        """预计算并保存所有90个权重图"""
        for full_path in self.label_paths:
            filename = os.path.basename(full_path)
            weight_path = os.path.join(self.save_dir, f"{filename.replace('.png', '')}.npy")

            # 如果权重图已存在则跳过
            if os.path.exists(weight_path):
                continue

            # 加载标签图像
            try:
                label_img = Image.open(full_path)
                label_array = np.array(label_img)
            except Exception as e:
                print(f"Error loading {full_path}: {e}")
                continue

            # 计算权重图
            weight_map = self.compute_weight_map(label_array)

            # 保存权重图
            np.save(weight_path, weight_map)

            # 保存可视化图像
            plt.imsave(
                weight_path.replace('.npy', '.png'),
                weight_map,
                cmap="viridis",
                vmin=weight_map.min(),
                vmax=weight_map.max()
            )
            print(f"Saved weight map for {filename}")

    def compute_weight_map(self, label_array):
        """计算单个标签的权重图"""
        height, width = label_array.shape
        unique_classes = np.unique(label_array)
        num_classes = len(unique_classes)

        # 1. 计算类别权重
        class_pixels = np.zeros(num_classes)
        for i, c in enumerate(unique_classes):
            class_pixels[i] = np.sum(label_array == c)

        # 添加epsilon防止除零
        total_pixels = height * width
        class_weights = total_pixels / (class_pixels + self.eps)
        class_weights /= class_weights.max()  # 归一化

        # 创建类别权重映射
        cw_map = np.zeros_like(label_array, dtype=np.float32)
        for i, c in enumerate(unique_classes):
            cw_map[label_array == c] = class_weights[i]

        # 2. 计算边界权重
        binary_mask = (label_array > 0).astype(np.uint8)
        dw_map = np.zeros((height, width), dtype=np.float32)

        if np.any(binary_mask):
            labeled, num_labels = ndimage.label(
                binary_mask,
                structure=np.ones((3, 3))  # 8-连通
            )

            if num_labels >= 2:
                # 计算每个实例的距离变换
                dist_maps = np.zeros((num_labels, height, width), dtype=np.float32)

                for k in range(1, num_labels + 1):
                    cell_mask = (labeled == k)
                    internal_dist = ndimage.distance_transform_edt(cell_mask)
                    external_dist = ndimage.distance_transform_edt(~cell_mask)
                    combined_dist = np.where(cell_mask, internal_dist, external_dist)
                    dist_maps[k - 1] = combined_dist

                # 计算每个像素到最近两个实例的距离
                dist_maps.sort(axis=0)
                d1 = dist_maps[0]
                d2 = dist_maps[1] if num_labels >= 2 else np.zeros_like(d1)

                # 计算边界权重
                dw_map = self.w0 * np.exp(-(d1 + d2) ** 2 / self.sigma_sq)

        # 3. 组合权重并确保数值稳定
        weight_map = cw_map + dw_map

        # 检查并修复NaN/Inf
        if np.isnan(weight_map).any() or np.isinf(weight_map).any():
            print("Warning: NaN/Inf detected in weight map. Replacing with 1.0")
            weight_map = np.nan_to_num(weight_map, nan=1.0, posinf=1.0, neginf=1.0)
            weight_map = np.clip(weight_map, 1e-3, 10 * self.w0)  # 安全范围

        return weight_map

    def forward(self, pred, target, targetid):
        """
        参数:
            pred: 模型预测值 (batch_size, num_classes, height, width)
            target: 真实标签 (batch_size, height, width)
            targetid: 标签文件名列表 (batch_size)
        返回:
            加权交叉熵损失
        """
        # 确保所有数据都在同一设备
        device = pred.device
        target = target.to(device).long()  # 移动到GPU并转换类型

        batch_size, num_classes, height, width = pred.shape

        # 裁剪logits值域 [-50, 50] 防止exp溢出
        pred = torch.clamp(pred, -50.0, 50.0)

        # 确保标签在有效范围内
        target = target.long()
        if (target.min() < 0) or (target.max() >= num_classes):
            target = torch.clamp(target, 0, num_classes - 1)

        # 加载预计算的权重图
        weight_map_batch = torch.zeros((batch_size, height, width),
                                       dtype=torch.float32,
                                       device=pred.device)

        for i, filename in enumerate(targetid):
            if filename in self.weights_cache:
                weight_map = np.load(self.weights_cache[filename])
                # 检查权重图异常值
                if np.isnan(weight_map).any() or np.isinf(weight_map).any():
                    print(f"Warning: Weight map {filename} contains NaN/Inf. Replacing with 1.0")
                    weight_map = np.nan_to_num(weight_map, nan=1.0, posinf=1.0, neginf=1.0)
                weight_map_tensor = torch.from_numpy(weight_map).float()
                weight_map_batch[i] = weight_map_tensor.to(pred.device)
            else:
                # 如果权重图未预计算，回退到实时计算
                print(f"Warning: Weight map for {filename} not precomputed. Calculating on the fly.")
                label_array = target[i].numpy()
                weight_map = self.compute_weight_map(label_array)
                weight_map = np.clip(weight_map, 1e-3, 100)
                weight_map_tensor = torch.from_numpy(weight_map).float()
                weight_map_batch[i] = weight_map_tensor.to(pred.device)

        # 计算log softmax
        log_probs = F.log_softmax(pred, dim=1)

        # 收集目标类别的log概率
        target_expanded = target.unsqueeze(1)  # 增加通道维度 (B,1,H,W)

        selected_log_probs = torch.gather(log_probs, 1, target_expanded).squeeze(1)

        # 计算加权损失
        weighted_loss = -weight_map_batch * selected_log_probs
        mean_loss = weighted_loss.mean()

        return mean_loss


if __name__ == '__main__':
    from skimage import io
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import measure, color
    import cv2

    # 读取图像并二值化
    gt = io.imread(r'C:\Users\24174\Desktop\cmmSecond\Unet-pytorch\data\ISBC2012\train\labels\frame_0001.png')
    gt = 1 * (gt > 0)

    plt.figure(figsize=(10, 10))
    plt.imshow(gt, cmap='gray')
    plt.title('Ground Truth')
    plt.colorbar()
    plt.show()

    # 【1】计算细胞和背景的像素频率
    c_weights = np.zeros(2)
    c_weights[0] = 1.0 / ((gt == 0).sum())
    c_weights[1] = 1.0 / ((gt == 1).sum())

    # 【2】归一化
    c_weights /= c_weights.max()

    # 【3】得到 class_weight map(cw_map)
    cw_map = np.where(gt == 0, c_weights[0], c_weights[1])

    plt.figure(figsize=(10, 10))
    im = plt.imshow(cw_map, cmap='viridis')
    plt.title('Class Weight Map')
    plt.colorbar(im)
    plt.show()

    # 【4】连通域分析，并彩色化
    cells = measure.label(gt, connectivity=2)
    cells_color = color.label2rgb(cells, bg_label=0, bg_color=(0, 0, 0))

    plt.figure(figsize=(20, 20))
    plt.imshow(cells_color)
    plt.title('Labeled Cells (Color)')
    plt.axis('off')  # 彩色标注图一般不加 colorbar
    plt.show()

    # 【5】计算 distance weight map (dw_map)
    w0 = 10
    sigma = 5
    dw_map = np.zeros_like(gt, dtype=float)
    maps = np.zeros((gt.shape[0], gt.shape[1], cells.max()))
    if cells.max() >= 2:
        for i in range(1, cells.max() + 1):
            maps[:, :, i - 1] = cv2.distanceTransform(1 - (cells == i).astype(np.uint8), cv2.DIST_L2, 3)
        maps = np.sort(maps, axis=2)
        d1 = maps[:, :, 0]
        d2 = maps[:, :, 1]
        dis = ((d1 + d2) ** 2) / (2 * sigma * sigma)
        dw_map = w0 * np.exp(-dis) * (cells == 0)

    plt.figure(figsize=(10, 10))
    im = plt.imshow(dw_map, cmap='jet')
    plt.title('Distance Weight Map')
    plt.colorbar(im)
    plt.show()

    # 最终权重图
    finalmap = cw_map + dw_map

    plt.figure(figsize=(10, 10))
    im = plt.imshow(finalmap, cmap='hot')
    plt.title('Final Weight Map')
    plt.colorbar(im)
    plt.show()
