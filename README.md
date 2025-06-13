# Pytorch复现Unet论文 


数据集我选择的是ISBC2012和consep数据集，存放在data下。因为我使用的是Unet论文中的Vaild卷积，所以输出尺寸会小于输入尺寸， 我这个框架处理是用来处理512尺寸的，
所以我把consep(1000尺寸)中的图片都分割为了512*512的大小。

数据处理部分，复现了Unet论文中的弹性形变，可见DataAugmentation.py。
模型部分，除了Unet，还复现了U2net和U2netS，由于资源有限，我使用了Unet和U2netS进行的实验，大家有兴趣也可以尝试U2net。
损失函数部分，复现了Unet论文中的权重图(类别权重+边界强化权重)，可见Loss.py，运行的时候会事先把标签的权重图保存到./weight_maps下，避免重复运算。

另外，论文中的重铺平叠(overlap)策略、高动量(0.99)、编码器末端加入dropout层、网络初始化等都进行了复现。
项目结构中的checkpoints_consep、checkpoints_ISBC、logs_consep、logs_ISBC、results_consep、results_ISBC文件夹都是我自己整理的，
程序实际运行并不会生成这些文件夹，会生成checkpoints、DataTxt、logs、results、weight_maps文件夹。

    这个项目的设计和运行都和我的jittor框架复现Unet一样。这里只有环境配置信息。
    jittor框架链接:


## 环境配置

### 我的配置
- Python: 3.12.7
- CUDA: 12.8
- GPU: Laptop3060
- 内存: 16G
- CPU: i7-12700H


### pytorch环境配置 requirements.txt  
```
matplotlib==3.10.3
numpy==2.3.0
scipy==1.15.3
skimage==0.0
torch==2.5.0+cu124
torchsummary==1.5.1
torchvision==0.20.0+cu124
tqdm==4.67.1
```
