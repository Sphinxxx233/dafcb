import os
import copy
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.server.strategy.aggregate import aggregate
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# 假设 cnn.py 和 cust.py 在同一目录下
# 如果没有，请确保这两个文件存在或将相关类定义放进来
try:
    from cnn import Net, test
    from cust import FlowerClient
except ImportError:
    print("错误: 缺少 cnn.py 或 cust.py。请确保它们在当前目录下。")
    exit(1)

# ================= 配置区域 =================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制使用 CPU 运行服务端逻辑
DEVICE = torch.device("cpu")
VALUE_MODEL_PATH = Path("value_predictor.pth")
HIDDEN_DIM = 128
DROPOUT_RATE = 0.1
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
NUM_ROUNDS = 110  # 建议多跑几轮
TEMPERATURE = 0.5 # Softmax温度：越小越贪婪(倾向高分)，越大越随机

# ================= 1. 定义预测网络结构 (与训练时保持一致) =================

class ResidualBlock(nn.Module):
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
        out += residual
        return self.activation(out)

class ComplexValuePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ComplexValuePredictor, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, DROPOUT_RATE),
            ResidualBlock(hidden_dim, DROPOUT_RATE),
            ResidualBlock(hidden_dim, DROPOUT_RATE)
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # 无激活函数，支持负数
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_blocks(x)
        return self.output_head(x)

# ================= 2. 数据集准备 =================

def prepare_fashion_dataset(
    num_partitions: int = 100,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    data_path: str = "./data_fashion",
):
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST(data_path, train=True, download=True, transform=tr)
    testset = FashionMNIST(data_path, train=False, download=True, transform=tr)

    num_images = len(trainset) // num_partitions
    partition_len = [num_images] * num_partitions
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    trainloaders, valloaders = [], []
    for subset in trainsets:
        num_total = len(subset)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(subset, [num_train, num_val], torch.Generator().manual_seed(2023))
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=0))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=0))

    testloader = DataLoader(testset, batch_size=128, num_workers=0)
    return trainloaders, valloaders, testloader


def prepare_fashion_dataset_non_iid(
    num_partitions: int = 100,
    batch_size: int = 32,
    alpha: float = 0.5, # Alpha 越小，异构性越强（每个客户端拥有的类别越少）
    data_path: str = "./data_fashion",
):
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST(data_path, train=True, download=True, transform=tr)
    testset = FashionMNIST(data_path, train=False, download=True, transform=tr)

    # --- Dirichlet Non-IID Split ---
    min_size = 0
    # 确保每个客户端至少有一点数据
    labels = trainset.targets
    num_classes = 10
    
    # 重复尝试直到切分成功（防止出现空客户端）
    while min_size < 10:
        idx_batch = [[] for _ in range(num_partitions)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
            ## 调整比例以处理不平衡
            proportions = np.array([p * (len(idx_j) < len(trainset) / num_partitions) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    trainloaders, valloaders = [], []
    for idx_j in idx_batch:
        subset = torch.utils.data.Subset(trainset, idx_j)
        # 划分 Train/Val
        num_total = len(subset)
        num_val = int(0.1 * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(subset, [num_train, num_val], torch.Generator().manual_seed(2023))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False))

    testloader = DataLoader(testset, batch_size=128)
    return trainloaders, valloaders, testloader 

# 全局加载数据
trainloaders, valloaders, testloader = prepare_fashion_dataset_non_iid(num_partitions=NUM_CLIENTS)

# ================= 3. 辅助函数 =================

def set_model_parameters(model: Net, parameters) -> Net:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model

def evaluate_global(parameters) -> Tuple[float, float]:
    model = Net(num_classes=10)
    set_model_parameters(model, parameters)
    return test(model, testloader)

def load_value_predictor() -> ComplexValuePredictor | None:
    if not VALUE_MODEL_PATH.exists():
        print(f"Warning: {VALUE_MODEL_PATH} 不存在, 将回退到随机选择。")
        return None
    
    # 输入特征维度与训练时保持一致（8 个特征）
    model = ComplexValuePredictor(input_dim=8, hidden_dim=HIDDEN_DIM)
    try:
        state = torch.load(VALUE_MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        print(f"成功加载预测模型: {VALUE_MODEL_PATH}")
        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

def build_value_features(
    rnd: int,
    last_selected_round: np.ndarray,
    participation_count: np.ndarray,
    loss_arr: np.ndarray,
    acc_arr: np.ndarray,
    data_arr: np.ndarray,
    grad_arr: np.ndarray,
    cosine_arr: np.ndarray,
    epochs_arr: np.ndarray,
) -> torch.Tensor:
    """构建特征矩阵，必须与训练时的归一化逻辑完全一致"""
    
    # 计算陈旧度：从未选中过(-1)的视为从 Round 0 开始，Staleness = rnd
    staleness = rnd - np.where(last_selected_round >= 0, last_selected_round, 0)
    
    feats = np.column_stack(
        [
            loss_arr,               # 0
            acc_arr,                # 1
            staleness,              # 2
            data_arr,               # 3
            grad_arr,               # 4
            cosine_arr,             # 5
            epochs_arr,             # 6
            participation_count,    # 7
        ]
    ).astype(np.float32)

    # 清洗 NaN
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # === 归一化 (与 train.py 严格对齐) ===
    feats[:, 0] /= 5.0      # Loss
    feats[:, 2] /= 100.0    # Staleness
    feats[:, 3] /= 2000.0   # Data Size
    feats[:, 4] /= 20.0     # Grad Norm
    feats[:, 6] /= 10.0     # Epochs
    feats[:, 7] /= 50.0     # Participation

    return torch.tensor(feats, dtype=torch.float32)

def select_clients_probabilistic(
    rnd: int,
    k: int,
    model: ComplexValuePredictor,
    last_selected_round: np.ndarray,
    participation_count: np.ndarray,
    loss_arr: np.ndarray,
    acc_arr: np.ndarray,
    data_arr: np.ndarray,
    grad_arr: np.ndarray,
    cosine_arr: np.ndarray,
    epochs_arr: np.ndarray,
    temperature: float = 1.0
) -> list[int]:
    """Softmax 概率加权选择"""
    
    # 1. 构建特征
    features = build_value_features(
        rnd, last_selected_round, participation_count,
        loss_arr, acc_arr, data_arr, grad_arr, cosine_arr, epochs_arr
    )

    # 2. 预测分数
    with torch.no_grad():
        scores = model(features).squeeze(-1) # shape (100,)
    
    # 3. Softmax 转化为概率
    # 除以 temperature 控制分布平滑度
    probs = F.softmax(scores / temperature, dim=0).cpu().numpy()
    
    # 4. 修复潜在的概率和不为1的问题 (浮点误差)
    probs = probs / np.sum(probs)
    
    # 5. 根据概率采样
    selected_indices = np.random.choice(
        len(probs), 
        size=k, 
        replace=False, 
        p=probs
    )
    
    return selected_indices.tolist()

# ================= 4. 主流程逻辑 =================

def run_fedavg_value(num_rounds: int = NUM_ROUNDS) -> List[Tuple[int, float]]:
    # 初始化客户端和参数
    clients = [FlowerClient(trainloaders[i], valloaders[i]) for i in range(NUM_CLIENTS)]
    global_params = clients[0].get_parameters({})
    value_model = load_value_predictor()

    # 状态追踪变量
    last_selected_round = np.full(NUM_CLIENTS, -1, dtype=int)
    last_loss = np.full(NUM_CLIENTS, np.nan)
    last_acc = np.full(NUM_CLIENTS, np.nan)
    last_data = np.full(NUM_CLIENTS, np.nan)
    last_grad = np.full(NUM_CLIENTS, np.nan)
    last_cosine = np.full(NUM_CLIENTS, np.nan)
    last_epochs = np.full(NUM_CLIENTS, np.nan)
    last_deposit = np.full(NUM_CLIENTS, np.nan)
    participation_count = np.zeros(NUM_CLIENTS, dtype=int)

    # === Warm-up 随机排列 ===
    warmup_permutation = np.random.permutation(NUM_CLIENTS)

    history: List[Tuple[int, float]] = []
    out_dir = Path("dataset") / "fsmnist_valuepred"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始训练，总轮数: {num_rounds}, 客户端总数: {NUM_CLIENTS}")

    for rnd in range(1, num_rounds + 1):
        # --- 1. 快照：保存选人前的状态 (Input X) ---
        # 这一步至关重要，防止 Time Leakage
        snapshot_selected_round = last_selected_round.copy()
        snapshot_loss = last_loss.copy()
        snapshot_accuracy = last_acc.copy()
        snapshot_data_size = last_data.copy()
        snapshot_grad_norm = last_grad.copy()
        snapshot_participation = participation_count.copy()
        
        # 临时工作变量 (用于更新)
        loss_arr = last_loss.copy()
        acc_arr = last_acc.copy()
        data_arr = last_data.copy()
        grad_arr = last_grad.copy()
        cos_arr = last_cosine.copy()
        epochs_arr = last_epochs.copy()
        deposit_arr = last_deposit.copy()
        flag = np.zeros(NUM_CLIENTS, dtype=int) # 标记本轮谁被选中

        # --- 2. 选人策略 ---
        sampled = []
        
        # 阶段 A: Warm-up (前10轮，保证全覆盖)
        if rnd <= 10:
            start_idx = (rnd - 1) * CLIENTS_PER_ROUND
            end_idx = rnd * CLIENTS_PER_ROUND
            sampled = warmup_permutation[start_idx:end_idx].tolist()
            print(f"Round {rnd} [Warm-up]: Indexes {start_idx}-{end_idx}")
            
        # 阶段 B: AI 模型概率选择 (第11轮+)
        else:
            if value_model is not None:
                try:
                    sampled = select_clients_probabilistic(
                        rnd, CLIENTS_PER_ROUND, value_model,
                        last_selected_round, participation_count,
                        last_loss, last_acc, last_data, last_grad, last_cosine, last_epochs,
                        temperature=TEMPERATURE
                    )
                    print(f"Round {rnd} [AI-Softmax]: {sampled}")
                except Exception as e:
                    print(f"模型预测出错 ({e})，降级为随机选择")
                    sampled = np.random.choice(NUM_CLIENTS, size=CLIENTS_PER_ROUND, replace=False).tolist()
            else:
                # 如果没有模型文件，继续随机
                sampled = np.random.choice(NUM_CLIENTS, size=CLIENTS_PER_ROUND, replace=False).tolist()
        # ========================================================
        # 【在这里加这一行】打印本轮最终被选中的 10 个客户端 ID
        print(f"Round {rnd} Selected Clients: {sampled}")
        # ========================================================

        # --- 3. 训练 (Fit) ---
        fit_results = []
        base_weights = global_params
        updates = {}

        for cid in sampled:
            # 训练
            params, num_examples, info = clients[cid].fit(global_params, {})
            fit_results.append((params, num_examples, info))

            # 更新状态 (这些是下一轮的特征)
            loss_arr[cid] = info.get("loss", np.nan)
            acc_arr[cid] = info.get("accuracy", np.nan)
            data_arr[cid] = info.get("data_size", np.nan)
            grad_arr[cid] = info.get("grad_norm", np.nan)
            epochs_arr[cid] = info.get("local_epochs", np.nan)
            
            # 随机更新 Deposit (模拟信誉分变化)
            deposit_arr[cid] = np.random.uniform(0, 100)

            flag[cid] = 1
            last_selected_round[cid] = rnd
            
            # 计算更新向量 (用于 Cosine)
            # Flatten params to 1D array
            current_flat = np.concatenate([p.ravel() for p in params])
            base_flat = np.concatenate([p.ravel() for p in base_weights])
            updates[cid] = current_flat - base_flat

        # --- 4. 计算 Cosine Similarity ---
        if updates:
            # 全局平均更新向量
            U_stack = np.stack(list(updates.values()))
            global_grad = U_stack.mean(axis=0)
            ng = np.linalg.norm(global_grad)
            
            for cid, upd in updates.items():
                nl = np.linalg.norm(upd)
                if nl > 0 and ng > 0:
                    cos_arr[cid] = np.dot(upd, global_grad) / (nl * ng)
                else:
                    cos_arr[cid] = 0.0

        # 更新参与次数
        participation_count[flag == 1] += 1

        # 将更新后的状态写回全局变量
        last_loss, last_acc = loss_arr, acc_arr
        last_data, last_grad = data_arr, grad_arr
        last_cosine, last_epochs = cos_arr, epochs_arr
        last_deposit = deposit_arr

        # --- 5. 聚合与评估 ---
        global_params = aggregate([(p, n) for (p, n, _) in fit_results])
        
        # 评估全局模型
        loss_val, acc_val = evaluate_global(global_params)
        history.append((rnd, acc_val))
        print(f"--> Global Acc: {acc_val:.4f}, Loss: {loss_val:.4f}")

        # --- 6. 计算 LOO Contribution (Label Y) ---
        contribution = np.full(NUM_CLIENTS, np.nan)
        for idx, cid in enumerate(sampled):
            # 排除当前客户端 cid
            others = [(p, n) for j, (p, n, _) in enumerate(fit_results) if j != idx]
            
            if others:
                w_minus = aggregate(others)
                _, acc_minus = evaluate_global(w_minus)
            else:
                # 极端情况：只有一个客户端，去掉后用上一轮模型评估
                # 这里简单起见，假设 acc_minus 0 或者复用上一轮 acc
                # 更好的是 evaluate_global(base_weights)
                acc_minus = 0.0 
            
            # 真实贡献 = 包含我时的Acc - 没我时的Acc
            contribution[cid] = acc_val - acc_minus

        # --- 7. 保存日志 (Snapshot X + Label Y) ---
        # 准备数据用于写入 CSV
        # 注意：这里使用的是 snapshot_ 系列变量，确保是训练前的旧状态
        metrics_matrix = np.column_stack(
            [
                np.arange(NUM_CLIENTS),             # 0: client_id
                np.full(NUM_CLIENTS, rnd),          # 1: current_round
                snapshot_selected_round,            # 2: last_selected (OLD)
                flag,                               # 3: selected
                snapshot_participation,             # 4: participation
                deposit_arr,                        # 5: deposit (Current)
                np.nan_to_num(snapshot_loss, nan=0.0),      # 6: loss (OLD)
                np.nan_to_num(snapshot_accuracy, nan=0.0),  # 7: acc (OLD)
                np.nan_to_num(snapshot_data_size, nan=0.0), # 8: data
                np.nan_to_num(snapshot_grad_norm, nan=0.0), # 9: grad
                np.nan_to_num(cos_arr, nan=0.0),            # 10: cosine (Current)
                np.nan_to_num(epochs_arr, nan=0.0),         # 11: epochs
                np.nan_to_num(contribution, nan=0.0),       # 12: contribution (Label)
            ]
        )
        
        out_path = out_dir / f"round_{rnd:02d}_metrics.csv"
        header = (
            "client_id,current_round,last_selected_round,selected,participation_count,"
            "deposit,loss,accuracy,data_size,grad_norm,cosine_to_global_update,local_epochs,contribution"
        )
        # 格式化字符串
        fmt = ["%d", "%d", "%d", "%d", "%d", "%.3f", "%.4f", "%.4f", "%.0f", "%.4f", "%.4f", "%.0f", "%.6f"]
        
        np.savetxt(out_path, metrics_matrix, delimiter=",", header=header, comments="", fmt=fmt)

    return history

def plot_history(history: List[Tuple[int, float]]) -> None:
    rounds = [r for r, _ in history]
    acc = [100.0 * a for _, a in history]
    plt.plot(rounds, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("FedAvg - AI Softmax Selection")
    plt.tight_layout()
    plt.savefig("accuracy_curve_ai_softmax.png")
    plt.close()

if __name__ == "__main__":
    print(f"Flower {fl.__version__}")
    print(f"Mode: {'AI Softmax Selection' if VALUE_MODEL_PATH.exists() else 'Random Data Collection'}")
    
    hist = run_fedavg_value()
    
    # 保存结果
    np.savetxt("accuracy_ai_softmax.csv", np.array(hist), delimiter=",", header="round,accuracy", comments="")
    plot_history(hist)
    print("Done.")
