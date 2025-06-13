# -*- coding: utf-8 -*-
import os
import shutil
from pathlib import Path

def create_dataset_txt_files(base_dir="data/ISBC2012", output_dir="TXT"):

    # 删除已存在的输出目录及其内容
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 定义路径
    train_img_dir = Path(base_dir) / "train" / "imgs"
    train_label_dir = Path(base_dir) / "train" / "labels"
    test_img_dir = Path(base_dir) / "test" / "imgs"
    test_label_dir = Path(base_dir) / "test" / "labels"

    # 验证目录是否存在
    for dir_path in [train_img_dir, train_label_dir, test_img_dir, test_label_dir]:
        if not dir_path.exists():
            print(f"警告: 目录不存在 - {dir_path}")

    # 获取所有文件路径（排序以确保匹配）
    train_img_files = sorted([str(f) for f in train_img_dir.iterdir() if f.is_file()])
    train_label_files = sorted([str(f) for f in train_label_dir.iterdir() if f.is_file()])
    test_img_files = sorted([str(f) for f in test_img_dir.iterdir() if f.is_file()])
    test_label_files = sorted([str(f) for f in test_label_dir.iterdir() if f.is_file()])

    # 写入训练集图像路径
    with open(Path(output_dir) / "train_image.txt", "w") as f:
        f.write("\n".join(train_img_files))

    # 写入训练集标签路径
    with open(Path(output_dir) / "train_label.txt", "w") as f:
        f.write("\n".join(train_label_files))

    # 写入测试集图像路径
    with open(Path(output_dir) / "test_image.txt", "w") as f:
        f.write("\n".join(test_img_files))

    # 写入测试集标签路径
    with open(Path(output_dir) / "test_label.txt", "w") as f:
        f.write("\n".join(test_label_files))

    print(f"成功创建了4个TXT文件在 {output_dir} 目录:")
    print(f"  - train_image.txt: {len(train_img_files)} 个图像")
    print(f"  - train_label.txt: {len(train_label_files)} 个标签")
    print(f"  - test_image.txt: {len(test_img_files)} 个图像")
    print(f"  - test_label.txt: {len(test_label_files)} 个标签")


if __name__ == "__main__":
    # 设置基础目录（根据您的实际结构调整）
    base_directory = "data/ISBC2012"

    # 创建TXT文件
    create_dataset_txt_files(base_dir=base_directory,output_dir="DataTxt")