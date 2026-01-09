# we naturally first need to import torch and torchvision
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

def get_mnist(data_path: str = './data'):
    '''下载MNIST数据集并应用转换（转为Tensor+标准化）'''
    # 定义转换函数，将图像转为Tensor，并标准化
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    # 下载训练集，并应用转换
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    # 下载测试集，并应用转换
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    # 返回训练集和测试集
    return trainset, testset
# 加载数据
trainset, testset = get_mnist()

# 创建DataLoader（批量加载数据）
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

#%% 可视化部分
def plot_images(dataset, num_samples=9):
    '''随机绘制样本图像'''
    indices = np.random.randint(0, len(dataset), num_samples)
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        # 反标准化：(img * std) + mean
        img = img * 0.3081 + 0.1307  # 还原到[0,1]范围
        plt.subplot(3, 3, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Label: {label}", fontsize=12)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# 类别分布可视化
def plot_class_distribution(dataset, title):
    labels = [dataset[i][1] for i in range(len(dataset))]
    plt.hist(labels, bins=10, rwidth=0.8)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(range(10))
    plt.grid(axis='y')
    plt.show()


