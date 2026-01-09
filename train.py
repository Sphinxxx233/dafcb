import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import os

# ================= 配置区域 =================
DATASET_PATH = "dataset/fsmnist"  # CSV 文件夹路径
MODEL_SAVE_PATH = "value_predictor.pth"
BATCH_SIZE = 32          # 使用 BN 层时，BatchSize 建议稍微大一点(16或32)
LEARNING_RATE = 0.001
EPOCHS = 1000
TARGET_SCALE = 100.0     # 标签放大倍数，让 0.001 变成 0.1
HIDDEN_DIM = 128         # 网络隐藏层宽度
DROPOUT_RATE = 0.1       # 防止过拟合
# ===========================================

# ================= 1. 数据集处理类 =================
class ClientContributionDataset(Dataset):
    def __init__(self, csv_dir):
        # --- 读取 CSV ---
        file_pattern = os.path.join(csv_dir, "round_*_metrics.csv")
        all_files = glob.glob(file_pattern)
        
        if not all_files:
            raise ValueError(f"错误：在 {csv_dir} 没找到 CSV 文件")
            
        print(f"正在加载 {len(all_files)} 个文件...")
        
        df_list = []
        for f in all_files:
            try:
                df = pd.read_csv(f)
                # 【只训练被选中的】只有 selected=1 才有真实贡献值
                labeled_data = df[df['selected'] == 1].copy()
                if not labeled_data.empty:
                    df_list.append(labeled_data)
            except:
                pass
        
        if not df_list:
            raise ValueError("没有找到任何有效的训练数据（需要 selected=1 的行）！")
            
        self.data = pd.concat(df_list, ignore_index=True)
        
        # --- 特征工程 ---
        # 计算 Staleness (陈旧度) = 当前轮次 - 上次选中轮次
        self.data['staleness'] = self.data['current_round'] - self.data['last_selected_round']
        
        # 定义全特征列表 (9个特征)
        self.feature_cols = [
            'loss',                    # 0
            'accuracy',                # 1
            'staleness',               # 2
            'data_size',               # 3
            'grad_norm',               # 4
            'cosine_to_global_update', # 5
            'local_epochs',            # 6s
            'participation_count',     # 7
            # 'deposit'                  # 8
        ]
        
        # 提取特征矩阵 X
        self.X = self.data[self.feature_cols].values.astype(np.float32)
        
        # 【鲁棒性】将 NaN 或 无穷大 替换为 0，防止报错
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # --- 全特征归一化 (Normalization) ---
        # 防止数值差异过大影响梯度下降
        
        # 1. Loss (通常 < 5.0)
        self.X[:, 0] = self.X[:, 0] / 5.0
        
        # 2. Accuracy (已经是 0-1) -> 不动
        
        # 3. Staleness (假设最大间隔 100 轮)
        self.X[:, 2] = self.X[:, 2] / 100.0
        
        # 4. Data Size (假设最大 2000 样本)
        self.X[:, 3] = self.X[:, 3] / 2000.0
        
        # 5. Grad Norm (梯度范数通常 < 20)
        self.X[:, 4] = self.X[:, 4] / 20.0
        
        # 6. Cosine (-1 到 1) -> 不动
        
        # 7. Local Epochs (通常 < 10)
        self.X[:, 6] = self.X[:, 6] / 10.0
        
        # 8. Participation Count (假设最大 50 次)
        self.X[:, 7] = self.X[:, 7] / 50.0
        
        # 9. Deposit (0-100) -> 归一化到 0-1
        # self.X[:, 8] = self.X[:, 8] / 100.0
        
        # --- 标签处理 ---
        # 放大 100 倍
        raw_contribution = self.data['contribution'].values.astype(np.float32)
        self.y = (raw_contribution * TARGET_SCALE).reshape(-1, 1)
        
        print(f"数据集构建完成！样本数: {len(self.data)}")
        print(f"输入特征维度: {self.X.shape[1]}")
        print(f"标签已放大 {TARGET_SCALE} 倍")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ================= 2. 复杂网络模型 (ResNet-MLP) =================

class ResidualBlock(nn.Module):
    """
    残差块：Linear -> BN -> ReLU -> Dropout -> Linear -> BN
    引入 x + f(x) 结构，适合深层网络
    """
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 残差连接
        return self.activation(out)

class ComplexValuePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ComplexValuePredictor, self).__init__()
        
        # 1. 映射层：将输入映射到高维空间
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. 核心层：堆叠 3 个残差块 (共 6 层深度)
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, DROPOUT_RATE),
            ResidualBlock(hidden_dim, DROPOUT_RATE),
            ResidualBlock(hidden_dim, DROPOUT_RATE)
        )
        
        # 3. 输出头：降维并输出
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            # 【重点】最后一层保持纯 Linear，无激活函数，支持负数输出！
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_head(x)


# ================= 3. 主训练流程 =================
if __name__ == "__main__":
    # 检查 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 加载数据
        dataset = ClientContributionDataset(DATASET_PATH)
        # Shuffle=True 非常重要
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    except Exception as e:
        print(f"无法开始训练: {e}")
        exit()
    
    # 初始化复杂模型
    input_dim = dataset.X.shape[1]
    model = ComplexValuePredictor(input_dim, hidden_dim=HIDDEN_DIM).to(device)
    print("\n网络结构已加载 (ResNet-MLP):")
    print(model)
    
    # 定义 Loss 和 优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n开始训练... (Epochs: {EPOCHS})")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            
    # 保存模型
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n模型已保存为 {MODEL_SAVE_PATH}")
    
    # ================= 4. 验证环节 =================
    # 抽取一个 batch 看看真实效果，确认能不能预测负数
    model.eval()
    with torch.no_grad():
        # 获取一个 Batch
        try:
            sample_X, sample_y = next(iter(dataloader))
        except StopIteration:
             sample_X, sample_y = next(iter(DataLoader(dataset, batch_size=len(dataset))))

        sample_X, sample_y = sample_X.to(device), sample_y.to(device)
        preds = model(sample_X)
        
        print("\n--- 预测结果抽查 (数值已放大100倍) ---")
        print(f"{'真实值':<10} | {'预测值':<10} | {'绝对误差':<10}")
        print("-" * 35)
        
        count = 0
        for i in range(len(sample_y)):
            real = sample_y[i].item()
            pred = preds[i].item()
            
            # 只打印前 10 个，或者打印特定的负数样本
            print(f"{real:<10.4f} | {pred:<10.4f} | {abs(real-pred):<10.4f}")
            
            count += 1
            if count >= 10: break
