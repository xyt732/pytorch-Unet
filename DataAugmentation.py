# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import argparse
import warnings

# 忽略所有的警告
warnings.filterwarnings("ignore")


# 弹性形变函数
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    '''
    :param image: 需要进行弹性形变的数据
    :param alpha:  缩放因子  生成位移场后  乘以alpha  比如  dx = gaussian_filter(...) * alpha
    :param sigma: 生成位移场  位移值从高斯分布采样  均值为0 标准差为sigma 越大越平滑
    :param alpha_affine:  控制仿射变换的随机扰动强度 平移、旋转、缩放
    :param random_state:  None表示使用非确定性随机种子 保证每次都不一样
    :return:
    '''
    if random_state is None:
        random_state = np.random.RandomState(None)  # None表示使用非确定性随机种子 保证每次都不一样

    shape = image.shape  # [w,h,2] 0是image 1是label 要保证image和label改变的量一样
    shape_size = shape[:2]  # [w,h]

    # 随机仿射变换(包括平移、旋转、缩放、剪切)
    center_square = np.float32(shape_size) // 2  # [w/2,h/2]
    square_size = min(shape_size) // 3  # w=h w/3=h/3
    # 定义三个控制点  [w/2+w/3,h/2+h/3]  [w/2+w/3,h/2-h/3]   [w/2-w/3,h/2-h/3]
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    # 添加随机扰动
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    '''
        计算仿射变换矩阵
    使用OpenCV的getAffineTransform函数
    根据三组对应点计算仿射变换矩阵
    求解从pts1到pts2的变换关系,也就是得到M矩阵 M矩阵怎么控制变换值可以看笔记
    '''
    M = cv2.getAffineTransform(pts1, pts2)
    '''
        应用仿射变换（边界反射填充）
    使用OpenCV的warpAffine函数应用变换
    保持输出图像尺寸不变 shape_size[::-1]反转数组 符合cv2的规范
    边界处理采用反射填充模式(BORDER_REFLECT_101),因为变换过程可能越界
    '''
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # 生成弹性变形（高斯滤波生成形变场） z轴为0  因为是二维图像
    #   随机场->高斯滤波(平滑处理,sigma越大越平滑)->缩放位移量alpha
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha  # [w,h,2]
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha  # [w,h,2]
    dz = np.zeros_like(dx)  # [w,h,2]

    # 应用形变（通过 map_coordinates 实现）
    '''
    meshgrid 的作用
    将三个一维数组扩展为三维网格
    输出三个三维数组：
      x-每个点的列坐标  y-每个点的行坐标  z-每个点的通道索引
    '''
    x, y, z = np.meshgrid(np.arange(shape[1]),
                          np.arange(shape[0]),
                          np.arange(shape[2]))
    '''
    indices:
    np.reshape(..., (-1, 1))：将三维坐标数组展平为一维列向量

    因为map_coordinates 函数要求坐标参数:
    以元组形式提供 (y_coords, x_coords, z_coords)
    '''
    indices = (np.reshape(y + dy, (-1, 1)),
               np.reshape(x + dx, (-1, 1)),
               np.reshape(z, (-1, 1)))

    # order=3 : 使用双三次插值更加平滑
    return map_coordinates(image, indices, order=3, mode='reflect').reshape(shape)


# 画网格线 可以方便看见变换效果 保存图片的时候就不要加了 会把网格线保存下来
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


if __name__ == '__main__':

    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Image augmentation with elastic deformation')
    parser.add_argument('--img_path', type=str,
                        default=r'C:\Users\24174\Desktop\cmmSecond\Unet-pytorch\data\ISBC2012\train\imgs',
                        help='Path to training images directory')
    parser.add_argument('--mask_path', type=str,
                        default=r'C:\Users\24174\Desktop\cmmSecond\Unet-pytorch\data\ISBC2012\train\labels',
                        help='Path to corresponding mask images directory')
    parser.add_argument('--sum', type=int, default=2,
                        help='Number of augmented samples per image (default: 2)')
    parser.add_argument('--alpha_lv', type=float, default=2.0,
                        help='Scaling factor for deformation (default: 2.0)')
    parser.add_argument('--sigma_lv', type=float, default=0.08,
                        help='Sigma for Gaussian filter (default: 0.08)')
    parser.add_argument('--alpha_affine_lv', type=float, default=0.08,
                        help='Affine transformation intensity (default: 0.08)')
    args = parser.parse_args()

    img_path = args.img_path
    mask_path = args.mask_path

    img_list = sorted(os.listdir(img_path))
    mask_list = sorted(os.listdir(mask_path))

    img_num = len(img_list)
    mask_num = len(mask_list)

    assert img_num == mask_num, 'img nuimber is not equal to mask num.'

    # 配置参数
    sum = args.sum  # 对于每一组image和label生成2组弹性形变后的数据

    # 这三个参数都会乘以im_merge的尺寸传进elastic_transform函数
    alpha_lv = args.alpha_lv  # 缩放因子
    sigma_lv = args.sigma_lv  # 高斯分布中的sigma
    alpha_affine_lv = args.alpha_affine_lv  # 控制仿射变换的随机扰动强度（平移/旋转/缩放）

    count_total = 0
    for i in range(img_num):
        im = cv2.imread(os.path.join(img_path, img_list[i]), -1)
        im_mask = cv2.imread(os.path.join(mask_path, mask_list[i]), -1)

        # # 画网格线
        # draw_grid(im, 50)
        # draw_grid(im_mask, 50)

        # 把image和label拼接在一起   (cols, rols, 2)
        im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)

        # 得到image和label的文件名和拓展名    splitext分割文件路径中的文件名和扩展名
        (img_shotname, img_extension) = os.path.splitext(img_list[i])
        (mask_shotname, mask_extension) = os.path.splitext(mask_list[i])

        # 弹性形变
        count = 0
        while count < sum:
            # 应用弹性形变
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * alpha_lv, im_merge.shape[1] * sigma_lv,
                                           im_merge.shape[1] * alpha_affine_lv)

            # 得到image 和 label
            im_t = im_merge_t[..., 0]
            im_mask_t = im_merge_t[..., 1]

            # save the new imgs and masks
            cv2.imwrite(os.path.join(img_path, img_shotname + '-' + str(count) + img_extension), im_t)
            cv2.imwrite(os.path.join(mask_path, mask_shotname + '-' + str(count) + mask_extension), im_mask_t)

            count += 1
            count_total += 1
            if count_total % 50 == 0:
                print('Elastic deformation generated {} imgs'.format(count_total))
                # 展示结果
                plt.figure(figsize=(16, 14))
                plt.imshow(np.c_[np.r_[im, im_mask], np.r_[im_t, im_mask_t]], cmap='gray')
                plt.show()