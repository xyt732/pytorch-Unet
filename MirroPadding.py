# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torch.nn.functional as F

def mirror_padding(image,pad_size):
    """
    使用镜像外推（mirroring）将256x256图像扩展为448x448图像
    参数:
        image: 输入图像，形状为(256, 256)或(256, 256, channels)
    返回:
        扩展后的图像，形状为(448, 448,...)
    """
    # 计算需要扩展的尺寸 (448 - 256 = 192，每边扩展96像素)
    pad_size = pad_size // 2

    # 垂直方向镜像填充 (上下各96像素)
    top_pad = image[1:pad_size + 1, ...][::-1, ...]  # 上填充：取第1-96行并垂直翻转
    bottom_pad = image[-pad_size - 1:-1, ...][::-1, ...]  # 下填充：取倒数第97-倒数第2行并垂直翻转
    img_v = np.concatenate((top_pad, image, bottom_pad), axis=0)

    # 水平方向镜像填充 (左右各96像素)
    left_pad = img_v[:, 1:pad_size + 1, ...][:, ::-1, ...]  # 左填充：取第1-96列并水平翻转
    right_pad = img_v[:, -pad_size - 1:-1, ...][:, ::-1, ...]  # 右填充：取倒数第97-倒数第2列并水平翻转
    img_h = np.concatenate((left_pad, img_v, right_pad), axis=1)

    return img_h

def mirror_padding_torch(patch):
    """
    使用镜像填充将 (B, C, 256, 256) 扩展到 (B, C, 448, 448)
    PyTorch 实现
    """
    pad_size = 96
    # 使用反射填充模式 (镜像填充)
    return F.pad(
        patch,
        (pad_size, pad_size, pad_size, pad_size),  # 左右上下填充
        mode='reflect'
    )


if __name__ == '__main__':
    # 检查文件路径
    file_path = r"C:\Users\24174\Desktop\cmmSecond\Unet-pytorch\data\ISBC2012\train\imgs\frame_0001.png"

    if not os.path.exists(file_path):
        print(f"错误：文件不存在 - {file_path}")
    else:
        try:
            # 读取图像
            img = Image.open(file_path)
            print(f"原始图像尺寸: {img.size}, 模式: {img.mode}")

            # 转换为NumPy数组
            img_array = np.array(img)

            # 截取左上角256x256部分
            crop_size = 256
            cropped_img = img_array[:crop_size, :crop_size]
            print(f"裁剪后图像尺寸: {cropped_img.shape}")

            # 应用镜像外推扩充
            padded_img = mirror_padding(cropped_img,pad_size=192)
            print(f"扩充后图像尺寸: {padded_img.shape}")

            # 创建可视化对比
            plt.figure(figsize=(15, 8))

            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
            plt.title(f"original ({img_array.shape[1]}×{img_array.shape[0]})")
            plt.axis('off')

            # 裁剪后的图像
            plt.subplot(1, 3, 2)
            plt.imshow(cropped_img, cmap='gray' if len(cropped_img.shape) == 2 else None)
            plt.title(f"cuted ({crop_size}×{crop_size})")
            plt.axis('off')

            # 扩充后的图像
            plt.subplot(1, 3, 3)
            plt.imshow(padded_img, cmap='gray' if len(padded_img.shape) == 2 else None)
            plt.title(f"padding ({padded_img.shape[1]}×{padded_img.shape[0]})")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

            # 保存结果图像
            Image.fromarray(cropped_img).save("cropped_image.png")
            Image.fromarray(padded_img).save("padded_image.png")
            print("裁剪和扩充后的图像已保存为 cropped_image.png 和 padded_image.png")

        except Exception as e:
            print(f"处理图像时出错: {e}")