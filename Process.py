# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

from Model import UNet,U2netS,U2net
from Loss import WeightedCrossEntropyLoss
from DataLoader import get_data_loaders, load_dataset_stats
from torchvision.transforms import Normalize

import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, data_dir, log_dir, checkpoint_dir, result_dir,is_grayscale,mode='Unet',n_channels=1, n_classes=2,
                 total_epochs=400, val_start_epoch=200, batch_size=1, dropout_probs=[0.1, 0.2, 0.3, 0.4],
                 bilinear=True, learning_rate=0.001, device='cuda:0', w0=10, sigma=512 * 0.08):
        # 初始化参数
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.total_epochs = total_epochs
        self.val_start_epoch = val_start_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.result_dir = result_dir
        self.dropout_probs = dropout_probs
        self.bilinear = bilinear
        self.w0 = w0
        self.sigma = sigma
        self.is_grayscale=is_grayscale
        self.mode = mode

        # 创建目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        # 设置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        self.csv_file = os.path.join(log_dir, f"training_log_{timestamp}.csv")

        # 初始化数据结构
        self.train_losses = []
        self.val_metrics = []
        self.best_metric = 0.0
        self.best_epoch = 0

        # 初始化模型、优化器和损失函数
        if mode == 'Unet':
            self.model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear,
                              dropout_probs=dropout_probs).to(self.device)
        elif mode == 'U2netS':
            self.model = U2netS(in_ch=n_channels, n_classes=n_classes).to(self.device)
        else:
            self.model = U2net(in_ch=n_channels, n_classes=n_classes).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.99, 0.999),
            weight_decay=1e-4
        )
        self.criterion = WeightedCrossEntropyLoss(
            label_file=data_dir + "/train_label.txt",
            w0=w0,
            sigma=sigma,
            save_dir="./weight_maps"
        )

        # 加载数据
        self._load_data()

        # 记录配置
        self._log_configuration()

        # 定义图像变换
        self.image_transform = Normalize(mean=[self.mean], std=[self.std])

    def _load_data(self):
        """加载数据集并获取均值和标准差"""
        print("加载数据集...")

        # 检查是否有保存的统计值
        existing_mean, existing_std, existing_grayscale = load_dataset_stats(self.data_dir)

        if existing_mean is not None:
            print(f"使用现有统计值: 均值={existing_mean}, 标准差={existing_std}")
            self.train_loader, self.val_loader, _ = get_data_loaders(
                self.data_dir,
                self.mode,
                batch_size=self.batch_size,
                use_custom_stats=False,
                is_grayscale=existing_grayscale
            )
            self.mean = existing_mean
            self.std = existing_std
        else:
            print("计算新的统计值...")
            self.train_loader, self.val_loader, stats = get_data_loaders(
                self.data_dir,
                self.mode,
                batch_size=self.batch_size,
                use_custom_stats=True,
                is_grayscale=self.is_grayscale
            )
            self.mean = stats[0]
            self.std = stats[1]

    def _log_configuration(self):
        """记录训练配置"""
        config = {
            "data_dir": self.data_dir,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "model": type(self.model).__name__,
            "criterion": type(self.criterion).__name__,
            "optimizer": type(self.optimizer).__name__,
            "is_grayscale": self.is_grayscale,
            "total_epochs": self.total_epochs,
            "val_start_epoch": self.val_start_epoch,
            "dropout_probs": self.dropout_probs,
            "bilinear": self.bilinear,
            "w0": self.w0,
            "sigma": self.sigma,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(self.log_file, 'a') as f:
            f.write("===== 训练配置 =====\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

        with open(self.csv_file, 'w') as f:
            f.write("epoch,train_loss,iou,dice,f1,epoch_time,throughput\n")

    def _log_metrics(self, epoch, train_loss,  metrics, epoch_time, throughput):
        """记录指标"""
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{self.total_epochs}\n")
            f.write(f"  Train Loss: {train_loss:.4f}\n")
            f.write(f"  epoch_time: {epoch_time:.4f}\n")
            f.write(f"  throughput: {throughput:.4f}\n")

            if epoch >= self.val_start_epoch:
                f.write(f"  IoU: {metrics['iou']:.4f}\n")
                f.write(f"  Dice: {metrics['dice']:.4f}\n")
                f.write(f"  F1: {metrics['f1']:.4f}\n")
            f.write("\n")

        with open(self.csv_file, 'a') as f:
            if epoch >= self.val_start_epoch:
                f.write(
                    f"{epoch + 1},{train_loss:.4f},{metrics['iou']:.4f},{metrics['dice']:.4f},{metrics['f1']:.4f}"
                    f",{epoch_time:.4f},{throughput:.4f}\n")
            else:
                f.write(f"{epoch + 1},{train_loss:.4f},,,,{epoch_time:.4f},{throughput:.4f}\n")

    def log_average_mid_epoch_performance(self):
        """计算并记录中间一半epoch的平均epoch_time和throughput 保留最佳性能指标"""
        # 读取CSV文件数据
        epochs = []
        epoch_times = []
        throughputs = []

        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                epoch_times.append(float(row['epoch_time']))
                throughputs.append(float(row['throughput']))

        # 计算中间一半epoch的范围
        total = len(epochs)
        mid_start = total // 4
        mid_end = 3 * total // 4

        # 计算平均值
        avg_epoch_time = np.mean(epoch_times[mid_start:mid_end])
        avg_throughput = np.mean(throughputs[mid_start:mid_end])

        # 最佳指标
        best_metrics = self.val_metrics[self.best_epoch - self.val_start_epoch]
        # 写入日志文件
        with open(self.log_file, 'a') as f:
            f.write("\n===== 平均性能指标 =====\n")
            f.write(f"中间一半epoch范围: {epochs[mid_start]} ~ {epochs[mid_end]}\n")
            f.write(f"平均epoch_time: {avg_epoch_time:.4f} 秒\n")
            f.write(f"平均throughput: {avg_throughput:.4f} images/sec\n")
            f.write("\n===== 最佳结果 =====\n")
            f.write(f"最佳epoch: {self.best_epoch + 1}\n")
            f.write(f"最佳IoU: {self.best_metric:.4f}\n")
            f.write(f"最佳Dice: {best_metrics['dice']:.4f}\n")
            f.write(f"最佳F1: {best_metrics['f1']:.4f}\n")

        print(f"\n平均性能指标和最佳结果已写入日志:")
        print(f"  Epoch范围: {epochs[mid_start]} ~ {epochs[mid_end]}")
        print(f"  平均epoch_time: {avg_epoch_time:.4f}秒")
        print(f"  平均throughput: {avg_throughput:.4f} images/sec")
        print("\n===== 最佳结果 =====")
        print(f"最佳epoch: {self.best_epoch + 1}")
        print(f"最佳IoU: {self.best_metric:.4f}")
        print(f"最佳Dice: {best_metrics['dice']:.4f}")
        print(f"最佳F1: {best_metrics['f1']:.4f}")

    def _save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点"""
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1],
            'val_metrics': self.val_metrics[-1] if epoch >= self.val_start_epoch else None,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'total_epochs': self.total_epochs,
            'val_start_epoch': self.val_start_epoch
        }

        # 保存最新模型
        torch.save(state, os.path.join(self.checkpoint_dir, "last_model.pth"))

        # 保存最佳模型
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pth"))
            print(f"保存最佳模型 (Epoch {epoch + 1}, IoU: {self.best_metric:.4f})")

    def process_batch(self, padded_blocks, label_blocks, full_labels):
        """
        处理批次数据
        :param padded_blocks: 填充后的图像块列表 (B x 4 x 448x448块)
        :param label_blocks: 对应的标签块列表 (B x 4 x 256x256块)
        :param full_labels: 完整512x512标签 (B x 512x512)
        :return: 完整预测和完整标签
        """
        blocks = []

        for sample_blocks in padded_blocks:
            sample_tensors = []
            for block in sample_blocks:  # sample_blocks (4*448*448)
                # 确保块是Tensor类型
                if isinstance(block, np.ndarray):
                    block = torch.from_numpy(block).float()
                elif block.dtype == torch.uint8:
                    block = block.float()

                # 处理无效形状
                if len(block.shape) == 1:
                    side_length = int(np.sqrt(block.size(0)))
                    if side_length * side_length == block.size(0):
                        block = block.reshape((side_length, side_length))
                    else:
                        if block.size(0) == 448 * 448:
                            block = block.reshape((448, 448))
                        elif block.size(0) == 256 * 256:
                            block = block.reshape((256, 256))
                # 确保块有正确的形状
                if len(block.shape) == 2:  # [H, W]
                    block = block.unsqueeze(0)  # 添加通道维度 [1, H, W]
                elif len(block.shape) == 3 and block.shape[0] <= 3:  # [C, H, W] - 正确格式 因为channel最大为3,所以用小于等于三判断
                    pass
                elif len(block.shape) == 3 and block.shape[1] <= 3:  # [H, C, W]
                    block = block.permute(1, 0, 2)  # 转为 [C, H, W]
                elif len(block.shape) == 3 and block.shape[2] <= 3:  # [H, W, C]
                    block = block.permute(0, 1, 2)  # 转为 [C, H, W]

                sample_tensors.append(block)  # 循环结束后 形状为4*c*448*448

            sample_tensor = torch.stack(sample_tensors)
            blocks.append(sample_tensor)  # 循环结束后 形状为 B*4*c*448*448

        # 堆叠整个批量的样本
        images_tensor = torch.stack(blocks).to(self.device)
        B, N, C, H, W = images_tensor.shape
        # 重塑为 [B*4, C, H, W] 以便模型处理
        images_tensor = images_tensor.reshape((B * N, C, H, W))

        # 模型预测
        #print('images_tensor.shape', images_tensor.shape)
        outputs = self.model(images_tensor)  # [B*4, n_classes, 260, 260]
        #print('outputs.shape:',outputs.shape)
        # 裁剪块
        cropped_outputs = torch.zeros((B * N, self.model.n_classes, 258, 258),
                                      device=self.device)  # [B*4, n_classes, 258, 258]

        # 左上块 (索引0): 裁剪顶部2行和左侧2列 -> 保留[2:, 2:]
        cropped_outputs[0::4] = outputs[0::4, :, 2:, 2:]

        # 右上块 (索引1): 裁剪顶部2行和右侧2列 -> 保留[2:, :-2]
        cropped_outputs[1::4] = outputs[1::4, :, 2:, :-2]

        # 左下块 (索引2): 裁剪底部2行和左侧2列 -> 保留[:-2, 2:]
        cropped_outputs[2::4] = outputs[2::4, :, :-2, 2:]

        # 右下块 (索引3): 裁剪底部2行和右侧2列 -> 保留[:-2, :-2]
        cropped_outputs[3::4] = outputs[3::4, :, :-2, :-2]

        # 重组输出
        cropped_outputs = cropped_outputs.reshape((B, N, self.model.n_classes, 258, 258))  # [B,4,n_classes, 258, 258]

        # 处理4个块的水平重叠
        for i in [0, 2]:
            left_block = cropped_outputs[i]
            right_block = cropped_outputs[i + 1]

            # 提取重叠区域
            left_overlap = left_block[:, :, :, -4:]
            right_overlap = right_block[:, :, :, :4]

            # 计算重叠区域平均值
            overlap_avg = (left_overlap + right_overlap) / 2.0

            # 更新左块（移除重叠区域）
            left_non_overlap = left_block[:, :, :, :-4]
            # 更新右块（移除重叠区域）
            right_non_overlap = right_block[:, :, :, 4:]

            # 重组左块：非重叠部分 + 平均重叠部分
            left_block = torch.cat([left_non_overlap, overlap_avg], dim=3)  # 258*258
            # 重组右块：平均重叠部分 + 非重叠部分
            right_block = torch.cat([overlap_avg, right_non_overlap], dim=3)  # 258*258

            cropped_outputs[i] = left_block
            cropped_outputs[i + 1] = right_block

        # 处理4个块的垂直重叠
        for i in [0, 1]:
            top_block = cropped_outputs[i]
            bottom_block = cropped_outputs[i + 2]

            top_overlap = top_block[:, :, -4:, :]
            bottom_overlap = bottom_block[:, :, :4, :]

            overlap_avg = (top_overlap + bottom_overlap) / 2.0

            top_non_overlap = top_block[:, :, :-4, :]
            bottom_non_overlap = bottom_block[:, :, 4:, :]

            top_block = torch.cat([top_non_overlap, overlap_avg], dim=2)
            bottom_block = torch.cat([overlap_avg, bottom_non_overlap], dim=2)

            cropped_outputs[i] = top_block
            cropped_outputs[i + 2] = bottom_block

        # 组合成完整图像
        full_pred = torch.zeros((1, self.model.n_classes, 512, 512))
        # 左上块
        full_pred[:, :, :258, :258] = cropped_outputs[0, :, :, :258, :258]
        # 右上块
        full_pred[:, :, :258, 258:] = cropped_outputs[1, :, :, :258, 4:]
        # 左下块
        full_pred[:, :, 258:, :258] = cropped_outputs[2, :, :, 4:, :258]
        # 右下块
        full_pred[:, :, 258:, 258:] = cropped_outputs[3, :, :, 4:, 4:]

        # 将完整标签移动到相同设备
        full_labels = full_labels.to(self.device)

        return full_pred, full_labels

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        skip_count = 0  # 记录跳过的batch数量

        start_time = time.time()
        prev_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.total_epochs} [Train]")
        for padded_blocks, label_blocks, full_labels, label_filename in loop:
            # 处理批次并拼接完整图像
            full_preds, full_labels = self.process_batch(padded_blocks, label_blocks, full_labels)

            # 重置梯度
            self.optimizer.zero_grad()

            # 计算损失
            loss = self.criterion(full_preds, full_labels, label_filename)

            # 检查梯度爆炸
            if loss.item() > 10:
                skip_count += 1
                # 恢复模型状态
                self.model.load_state_dict(prev_state)
                loop.set_postfix(loss=loss.item(), skip=skip_count)
                continue

            # 反向传播
            loss.backward()
            # 优化步骤
            self.optimizer.step()

            # 更新模型状态快照
            prev_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            # 更新统计
            running_loss += loss.item()
            total_samples += 1

            loop.set_postfix(loss=loss.item())

        # 计算性能指标
        end_time = time.time()
        epoch_time = end_time - start_time
        throughput = len(self.train_loader) / epoch_time if epoch_time > 0 else 0

        # 记录跳过情况
        if skip_count > 0:
            skip_msg = f"跳过 {skip_count} 个batch (loss > 10)"
            print(f"\n警告: {skip_msg}")
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch + 1} 警告: {skip_msg}\n")

        epoch_loss = running_loss / total_samples if total_samples > 0 else 0
        self.train_losses.append(epoch_loss)
        return epoch_loss, epoch_time, throughput

    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        total_samples = 0
        iou_sum = dice_sum = f1_sum = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validation")
            for padded_blocks, label_blocks, full_labels, label_filename in loop:
                # 处理批次并拼接完整图像
                full_preds, full_labels = self.process_batch(padded_blocks, label_blocks, full_labels)


                # 计算指标
                metrics = self.calculate_metrics(full_preds, full_labels)

                iou_sum += metrics['iou']
                dice_sum += metrics['dice']
                f1_sum += metrics['f1']
                total_samples += 1

        # 计算平均指标

        metrics = {
            'iou': iou_sum / total_samples,
            'dice': dice_sum / total_samples,
            'f1': f1_sum / total_samples
        }
        self.val_metrics.append(metrics)

        return metrics

    def calculate_metrics(self, outputs, labels):
        """计算分割指标"""
        device = outputs.device
        labels = labels.to(device)

        preds = torch.argmax(outputs, dim=1)  # (B, 512, 512)
        metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0}
        n = outputs.shape[0]

        for i in range(n):
            pred = preds[i]
            label = labels[i]

            # 计算交集和并集
            intersection = torch.logical_and(pred, label).sum()
            union = torch.logical_or(pred, label).sum()

            # 计算IoU
            iou = (intersection.item() + 1e-6) / (union.item() + 1e-6)

            # 计算Dice
            dice = (2 * intersection.item() + 1e-6) / (pred.sum().item() + label.sum().item() + 1e-6)

            # 计算F1
            precision = (intersection.item() + 1e-6) / (pred.sum().item() + 1e-6)
            recall = (intersection.item() + 1e-6) / (label.sum().item() + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

            metrics['iou'] += iou
            metrics['dice'] += dice
            metrics['f1'] += f1

        # 平均指标
        metrics['iou'] /= n
        metrics['dice'] /= n
        metrics['f1'] /= n

        return metrics

    def save_validation_results(self):
        """使用最佳模型保存验证集的分割结果"""
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_model_path):
            print("未找到最佳模型，跳过结果保存")
            return

        print(f"使用最佳模型保存验证集分割结果到: {self.result_dir}")

        # 加载最佳模型
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        os.makedirs(self.result_dir, exist_ok=True)

        with torch.no_grad():
            for padded_blocks, label_blocks, full_labels, label_filenames in tqdm(self.val_loader, desc="保存验证结果"):
                full_preds, _ = self.process_batch(padded_blocks, label_blocks, full_labels)
                preds = torch.argmax(full_preds, dim=1).cpu().numpy().astype(np.uint8)

                for i in range(preds.shape[0]):
                    base_name = os.path.basename(label_filenames[i])
                    result_name = os.path.splitext(base_name)[0] + "_pred.png"
                    save_path = os.path.join(self.result_dir, result_name)
                    plt.imsave(save_path, preds[i], cmap='gray', vmin=0, vmax=1)

        print(f"保存完成! 共处理 {len(self.val_loader.dataset)} 个验证样本")

    def run(self):
        """运行整个训练流程"""
        start_time = time.time()
        print(f"开始训练，共{self.total_epochs}个epoch (设备: {self.device})")

        for epoch in range(self.total_epochs):
            # 训练阶段
            train_loss, epoch_time, throughput = self.train_epoch(epoch)

            # 验证阶段
            metrics = {'iou': 0.0, 'dice': 0.0, 'f1': 0.0}

            if epoch >= self.val_start_epoch:
                metrics = self.validate_epoch()
                if metrics['iou'] > self.best_metric:
                    self.best_metric = metrics['iou']
                    self.best_epoch = epoch
                    self._save_checkpoint(epoch, is_best=True)

            # 记录日志
            self._log_metrics(epoch, train_loss, metrics, epoch_time, throughput)

            # 定期保存模型
            if (epoch + 1) % 50 == 0:
                self._save_checkpoint(epoch)

        # 训练结束
        total_time = time.time() - start_time
        print(f"训练完成! 总耗时: {total_time // 3600:.0f}h {(total_time % 3600) // 60:.0f}m {total_time % 60:.0f}s")
        self._save_checkpoint(self.total_epochs - 1)

        # 绘制损失曲线
        self.plot_training_curve()

        # 保存验证集结果
        if self.best_metric > 0:
            self.save_validation_results()

        # 计算并记录中间一半epoch的平均性能指标
        self.log_average_mid_epoch_performance()

    def plot_training_curve(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 8))

        # 训练损失
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        # 验证指标
        if len(self.val_metrics) > 0:
            epochs = range(self.val_start_epoch, self.total_epochs)

            plt.subplot(2, 1, 2)
            plt.plot(epochs, [m['iou'] for m in self.val_metrics], label='IoU', color='green')
            plt.plot(epochs, [m['dice'] for m in self.val_metrics], label='Dice', color='blue')
            plt.plot(epochs, [m['f1'] for m in self.val_metrics], label='F1', color='red')
            plt.xlabel('Epoch')
            plt.ylabel('Metric Value')
            plt.title('Validation Metrics')
            plt.legend(loc='upper right')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.log_dir, f'training_curve{timestamp}.png'))
        plt.close()


if __name__ == "__main__":
    # 配置参数
    DATA_DIR = r"C:\Users\24174\Desktop\cmmSecond\Unet-pytorch\DataTxt"
    LOG_DIR = r"./logs"
    CHECKPOINT_DIR = r"./checkpoints"
    RESULT_DIR = r"./results"

    # 可配置的训练参数
    TOTAL_EPOCHS = 50
    VAL_START_EPOCH = 25

    # 检查GPU是否可用
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 创建并运行训练器
    trainer = Trainer(
        data_dir=DATA_DIR,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        result_dir=RESULT_DIR,
        n_channels=1,
        n_classes=2,
        total_epochs=TOTAL_EPOCHS,
        val_start_epoch=VAL_START_EPOCH,
        batch_size=1,
        dropout_probs=[0.2, 0.3, 0.4, 0.5],
        bilinear=True,
        learning_rate=0.001,
        device=device
    )

    # 开始训练
    trainer.run()