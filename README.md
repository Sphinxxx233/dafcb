# 联邦学习客户端选择策略

本项目实现了多种联邦学习客户端选择策略，基于Fashion-MNIST数据集进行比较。项目使用Flower框架，专注于优化客户端参与度，以提高在非独立同分布（Non-IID）数据分布下的全局模型性能。

## 📋 目录

- [项目概述](#项目概述)
- [功能特性](#功能特性)
- [项目结构](#项目结构)
- [安装说明](#安装说明)
- [使用方法](#使用方法)
- [客户端选择策略](#客户端选择策略)
- [数据集](#数据集)
- [模型架构](#模型架构)
- [指标与评估](#指标与评估)
- [实验结果](#实验结果)
- [贡献指南](#贡献指南)

## 🎯 项目概述

联邦学习能够在多个去中心化的边缘设备上训练机器学习模型，同时保持数据本地化。一个关键的挑战是在每个训练轮次中选择参与训练的客户端，尤其是在数据分布不均匀的情况下。本项目实现并比较了以下几种选择策略：

- **贪婪选择（Greedy）**：选择训练损失最高的客户端
- **GAILV选择**：基于梯度的自适应重要性学习进行价值预测
- **在线学习选择**：使用在线学习算法进行自适应选择

## ✨ 功能特性

- ✅ 多种客户端选择策略
- ✅ 使用狄利克雷分布进行非IID数据划分
- ✅ 基于CNN的模型架构
- ✅ 全面的指标跟踪（PCDI、梯度范数、余弦相似度）
- ✅ 详细的日志记录和可视化
- ✅ Fashion-MNIST数据集支持
- ✅ 灵活的配置系统
- ✅ 支持tmux的自动化训练

## 📁 项目结构

```
Fedavg/
├── cnn.py                          # CNN模型架构和训练函数
├── cust.py                         # FlowerClient实现
├── fenqu.py                        # 数据划分工具
├── minist.py                       # MNIST数据集工具
├── run_greedy.py                   # 贪婪客户端选择策略
├── run_gailv.py                    # GAILV选择策略
├── run_online_learning.py          # 在线学习选择策略
├── client_selector.py              # 客户端选择工具
├── analyze_selection.py            # 选择分析工具
├── start_fl.sh                     # 自动化训练Shell脚本
├── dataset/                        # 训练日志和指标
│   ├── greedy_logs/               # 贪婪策略日志
│   ├── da_fcb_logs/               # 基线日志
│   └── ...
├── accuary/                        # 准确率结果和图表
├── data_fashion/                   # Fashion-MNIST数据集
├── eval/                           # 评估工具
└── README.md                       # 本文件
```

## 🚀 安装说明

### 环境要求

- Python 3.8+
- Conda（推荐）
- 支持CUDA的GPU（可选，当前配置为CPU）

### 安装步骤

1. 克隆仓库：
```bash
git clone <repository-url>
cd Fedavg
```

2. 创建并激活conda环境：
```bash
conda create -n fl_env python=3.9
conda activate fl_env
```

3. 安装依赖：
```bash
pip install flwr torch torchvision numpy matplotlib
```

4. 验证安装：
```bash
python -c "import flwr; import torch; print('安装成功！')"
```

## 💻 使用方法

### 运行单个策略

#### 贪婪选择策略
```bash
python run_greedy.py
```

#### GAILV选择策略
```bash
python run_gailv.py
```

#### 在线学习选择策略
```bash
python run_online_learning.py
```

### 使用Tmux自动化训练

使用提供的Shell脚本进行长时间训练：

```bash
./start_fl.sh
```

该脚本将：
- 创建名为"myjob"的tmux会话
- 激活conda环境
- 运行联邦学习训练
- 在tmux会话中保持可访问的日志

查看训练日志：
```bash
tmux attach -t myjob
```

从会话中分离：按 `Ctrl+B`，然后按 `D`

### 自定义配置

修改脚本文件中的参数：

```python
# 系统参数
NUM_ROUNDS = 150                 # 总训练轮次
NUM_CLIENTS = 100                # 客户端总数
CLIENTS_PER_ROUND = 10           # 每轮选择的客户端数
WARMUP_ROUNDS = 20               # 随机选择的热身轮次
TRAIN_EPOCHS_PER_ROUND = 30      # 本地训练轮数
BATCH_SIZE = 64                  # 批次大小
LR = 0.005                       # 学习率
```

## 🎲 客户端选择策略

### 1. 贪婪选择（最大损失）

**文件**：`run_greedy.py`

选择上一轮训练损失最高的客户端。

**优点**：
- 简单直观
- 专注于表现较差的客户端
- 无额外计算开销

**缺点**：
- 可能导致客户端饥饿
- 忽略数据质量和多样性

**使用方法**：
```python
python run_greedy.py
```

### 2. GAILV选择（基于梯度的自适应重要性学习）

**文件**：`run_gailv.py`

使用梯度信息和价值预测来选择客户端。

**优点**：
- 考虑梯度质量
- 预测客户端贡献
- 平衡探索和利用

**使用方法**：
```python
python run_gailv.py
```

### 3. 在线学习选择

**文件**：`run_online_learning.py`

使用在线学习算法进行自适应选择，根据历史性能持续改进客户端选择。

**优点**：
- 适应变化的客户端行为
- 学习最优选择模式
- 对非平稳环境鲁棒

**使用方法**：
```python
python run_online_learning.py
```

## 📊 数据集

### Fashion-MNIST

- **总样本数**：70,000（60,000训练，10,000测试）
- **图像尺寸**：28×28灰度图
- **类别数**：10（T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴）
- **非IID划分**：狄利克雷分布，α=0.1

### 数据划分

使用基于狄利克雷分布（α=0.1）的非IID方案将数据集划分到100个客户端，创建联邦学习场景中常见的真实数据异构性。

**PCDI（每类分布指数）**：通过计算客户端间标签分布的方差来衡量数据异构性。

## 🧠 模型架构

### CNN模型（NetV2）

```
输入 (1×28×28)
    ↓
模块1:
  - Conv2d(1→32, 3×3) + BatchNorm + ReLU
  - Conv2d(32→32, 3×3) + BatchNorm + ReLU
  - MaxPool2d(2×2)  [28→14]
  - Dropout(0.1)
    ↓
模块2:
  - Conv2d(32→64, 3×3) + BatchNorm + ReLU
  - Conv2d(64→64, 3×3) + BatchNorm + ReLU
  - MaxPool2d(2×2)  [14→7]
  - Dropout(0.2)
    ↓
头部:
  - Flatten
  - Linear(64×7×7→128) + ReLU
  - Dropout(0.3)
  - Linear(128→10)
    ↓
输出 (10个类别)
```

**总参数量**：约130万

## 📈 指标与评估

### 跟踪指标

对于每个客户端和轮次，记录以下指标：

- **client_id**：客户端标识符
- **current_round**：训练轮次编号
- **last_selected_round**：该客户端最后一次被选择的轮次
- **selected**：二进制标志（如果本轮被选中则为1）
- **participation**：总参与次数
- **loss**：客户端训练损失
- **accuracy**：客户端验证准确率
- **data_size**：客户端数据集大小
- **grad_norm**：梯度范数
- **cosine**：与平均更新的余弦相似度
- **local_epochs**：本地训练轮数
- **pcdi**：每类分布指数
- **contribution**：客户端贡献分数

### 全局评估

- 在完整Fashion-MNIST测试集上的测试准确率
- 全局损失
- 收敛分析

### 输出文件

- **指标CSV**：`dataset/[strategy]_logs/round_XX_metrics.csv`
- **准确率CSV**：`accuary/[strategy]/accuracy_[strategy].csv`
- **准确率图表**：`accuary/[strategy]/accuracy_plot_[strategy].png`

## 📊 实验结果

### 结果示例

运行策略后，您将找到：

1. **训练日志**：`dataset/[strategy]_logs/` 中每轮的详细指标
2. **准确率曲线**：模型性能随轮次变化的可视化比较
3. **选择分析**：客户端选择模式的见解

### 分析结果

使用分析脚本：

```bash
# 读取并绘制指标
python read_metrics.py

# 分析选择质量
python analyze_selection.py
```

## 🔬 主要发现

本研究表明：

1. **贪婪选择**可以实现有竞争力的性能，但可能遭受客户端饥饿问题
2. **GAILV**通过考虑梯度质量和预测价值改进了贪婪策略
3. **在线学习**适应客户端行为，可以实现更好的长期性能
4. **PCDI**是衡量联邦环境中数据异构性的有效指标

## 🛠️ 开发指南

### 添加新策略

1. 按照`run_greedy.py`的模式创建新脚本
2. 在主循环中实现客户端选择逻辑
3. 相应更新指标日志记录
4. 运行并比较结果

### 修改模型

编辑`cnn.py`以修改架构。确保与`cust.py`兼容以进行参数序列化。

## 📝 引用

如果您在研究中使用此代码，请引用：

```bibtex
@software{fedavg_client_selection,
  title={联邦学习客户端选择策略},
  author={[您的姓名]},
  year={2025},
  url={[仓库URL]}
}
```

## 🤝 贡献

欢迎贡献！请随时提交Pull Request。

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件。

## 🙏 致谢

- Flower框架用于联邦学习实现
- Fashion-MNIST数据集由Zalando Research提供
- PyTorch团队提供的深度学习框架

## 📧 联系方式

如有问题或建议，请在GitHub上创建issue或联系[您的邮箱]。

---

**注意**：本项目专用于研究目的。根据您的具体需求和约束调整参数和策略。
