# -*- coding: utf-8 -*-
import torch
from Process import Trainer
from MakeTxt import create_dataset_txt_files
import argparse

'''
---------------先使用DataAugmentation进行数据的填充---------------
参数:
--img_path "C:/Users/24174/Desktop/cmmSecond/Unet-pytorch/data/ISBC2012/train/imgs" \
--mask_path "C:/Users/24174/Desktop/cmmSecond/Unet-pytorch/data/ISBC2012/train/labels" \
--sum 2 \
--alpha_lv 2 \
--sigma_lv 0.08 \
--alpha_affine_lv 0.08
'''

if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='process confing')

    parser.add_argument('--Data_directory', type=str, default='./data/ISBC2012',
                        help='Data directory ')

    parser.add_argument('--Datatxt_directory', type=str, default='./DataTxt',
                        help='DataTxt directory ')

    parser.add_argument('--Log_directory', type=str, default='./logs',
                        help='Log directory ')

    parser.add_argument('--Checkpoint_directory', type=str, default='./checkpoints',
                        help='Checkpoint directory ')

    parser.add_argument('--Result_directory', type=str, default='./results',
                        help='Result directory ')

    parser.add_argument('--is_grayscale', type=str, default='True',
                        help='is_grayscale')

    parser.add_argument('--n_channels', type=int, default=1,
                        help='n_channels')

    parser.add_argument('--n_classes', type=int, default=2,
                        help='n_classes')

    parser.add_argument('--Total_epochs', type=int, default=400,
                        help='Total_epochs')

    parser.add_argument('--Val_epochs', type=int, default=200,
                        help='Val_epochs')

    parser.add_argument('--Batch_size', type=int, default=1,
                        help='Batch_size')

    parser.add_argument('--dropout_probs', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5],
                       help='List of dropout probabilities (0.2 0.3 0.4 0.5)')

    parser.add_argument('--bilinear', type=str, default='False',
                        help='bilinear')

    parser.add_argument('--Learning_rate', type=float, default=0.001,
                        help='Learning_rate')

    parser.add_argument('--Loss_w0', type=int, default=10,
                        help='Loss_w0')

    parser.add_argument('--Loss_sigma', type=float, default=512*0.08,
                        help='Loss_sigma')

    parser.add_argument('--mode', type=str, default='Unet',
                        help='mode')

    args = parser.parse_args()

    '''
    参数设置
    --Data_directory "./data/ISBC2012"
    --Datatxt_directory "./DataTxt"
    --Log_directory "./logs"
    --Checkpoint_directory "./checkpoints"
    --Result_directory "./results"
    --n_channels 1
    --n_classes 2
    --Total_epochs 100
    --Val_epochs 50
    --Batch_size 1
    --bilinear False
    --dropout_probs 0.1 0.2 0.3 0.4
    --Learning_rate 0.0001
    --Loss_w0 10
    --Loss_sigma 5   512*0.08=40.96
    --mode Unet
    '''

    # 创建保存image label路径的txt文件
    create_dataset_txt_files(args.Data_directory, args.Datatxt_directory)

    # 检查GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建Trainer类
    trainer = Trainer(
        data_dir=args.Datatxt_directory,
        log_dir=args.Log_directory,
        checkpoint_dir=args.Checkpoint_directory,
        result_dir=args.Result_directory,
        is_grayscale=(args.is_grayscale.lower() == 'true'),
        mode=args.mode,
        n_channels=args.n_channels,
        n_classes=args.n_classes,
        total_epochs=args.Total_epochs,
        val_start_epoch=args.Val_epochs,
        batch_size=args.Batch_size,
        dropout_probs=args.dropout_probs,
        bilinear=(args.bilinear.lower() == 'true'),
        learning_rate=args.Learning_rate,
        device=device,
        w0=args.Loss_w0,
        sigma=args.Loss_sigma,
    )

    # 开始训练
    trainer.run()