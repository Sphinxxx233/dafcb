import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 工具：对一列特征做 min-max 归一化
# ============================================================
def min_max_norm(x: np.ndarray) -> np.ndarray:
    """对一维数组做 min-max 归一化到 [0, 1]。如果常数列则全置 0。"""
    mn = x.min()
    mx = x.max()
    if mx > mn:
        return (x - mn) / (mx - mn)
    else:
        return np.zeros_like(x, dtype=np.float32)


# ============================================================
# 教师评分函数（基于你的数据列）
# ============================================================
def compute_teacher_scores_from_df(df: pd.DataFrame, max_round: int) -> np.ndarray:
    """
    根据一个 round_X_metrics.csv 对应的 DataFrame 计算教师评分。

    参数：
        df: 包含以下列（不含 client_id）：
            current_round,last_selected_round,selected,
            participation_count,deposit,loss,accuracy,data_size,
            grad_norm,cosine_to_global_update,local_epochs
        max_round: 总轮数，用于计算 round_norm（当前暂未进公式）

    返回：
        teacher_scores: (num_clients,) 的 numpy 数组
    """

    # 取出原始列（按名字，不按位置）
    current_round = df["current_round"].to_numpy(dtype=np.float32)
    last_selected_round = df["last_selected_round"].to_numpy(dtype=np.float32)
    selected = df["selected"].to_numpy(dtype=np.float32)
    participation_count = df["participation_count"].to_numpy(dtype=np.float32)
    deposit = df["deposit"].to_numpy(dtype=np.float32)
    loss = df["loss"].to_numpy(dtype=np.float32)
    accuracy = df["accuracy"].to_numpy(dtype=np.float32)
    data_size = df["data_size"].to_numpy(dtype=np.float32)
    grad_norm = df["grad_norm"].to_numpy(dtype=np.float32)
    cosine_to_global_update = df["cosine_to_global_update"].to_numpy(dtype=np.float32)
    local_epochs = df["local_epochs"].to_numpy(dtype=np.float32)

    # 派生特征
    rounds_since_last = current_round - last_selected_round          # 距离上次参与轮数
    round_norm = current_round / float(max_round)                    # 当前轮数归一化（这里先不用进公式）
    compute_cost = data_size * local_epochs                          # 通信/计算开销

    # 需要参与教师打分的特征做归一化
    loss_n = min_max_norm(loss)
    acc_n = min_max_norm(accuracy)
    grad_sim_n = min_max_norm(cosine_to_global_update)
    cost_n = min_max_norm(compute_cost)
    since_last_n = min_max_norm(rounds_since_last)
    hist_part_n = min_max_norm(participation_count)
    deposit_n = min_max_norm(deposit)

    # 教师权重（可调）
    w_acc = 0.40
    w_loss = 0.30
    w_grad_sim = 0.10
    w_cost = 0.10
    w_fair1 = 0.05
    w_fair2 = 0.03
    w_deposit = 0.02

    # 教师评分公式（所有项已归一化）
    q = (
        w_acc * acc_n
        - w_loss * loss_n
        + w_grad_sim * grad_sim_n
        - w_cost * cost_n
        + w_fair1 * since_last_n
        - w_fair2 * hist_part_n
        + w_deposit * deposit_n
    )

    return q.astype(np.float32)


# ============================================================
# Dataset：按轮读取 round_{10..110}_metrics.csv
# ============================================================
class FLRoundDataset(Dataset):
    """
    读取 ../dataset 目录下的 round_{10..110}_metrics.csv，
    构建用于训练选择模型的样本：

        X: S_t，形状 (num_clients, 11)，即去掉 client_id 后的所有列
        y: teacher_scores，形状 (num_clients,)

    一个 CSV（一个 round）视作一个样本。
    """

    def __init__(self, dataset_dir: str = "../dataset",
                 start_round: int = 10,
                 end_round: int = 110):
        self.dataset_dir = dataset_dir
        self.start_round = start_round
        self.end_round = end_round
        self.max_round = end_round  # 用于 round_norm，如果后面需要

        # 收集存在的 csv 路径
        self.paths = []
        for r in range(start_round, end_round + 1):
            path = os.path.join(dataset_dir, f"round_{r}_metrics.csv")
            if os.path.exists(path):
                self.paths.append(path)

        self.paths.sort()
        print(f"共找到 {len(self.paths)} 个 round_{{r}}_metrics.csv 文件。")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        csv_path = self.paths[idx]
        df = pd.read_csv(csv_path)

        # 1. 构造输入 X：去掉 client_id，剩下 11 列全部作为特征
        feature_df = df.drop(columns=["client_id"])
        # 特征矩阵，形状 (num_clients, 11)
        X = feature_df.to_numpy(dtype=np.float32)

        # 对每一轮内部，按列做归一化（也可以选择在网络里做，这里先做干净一点）
        X_norm = X.copy()
        for j in range(X.shape[1]):
            col = X[:, j]
            mn, mx = col.min(), col.max()
            if mx > mn:
                X_norm[:, j] = (col - mn) / (mx - mn)
            else:
                X_norm[:, j] = 0.0  # 常数列，直接归零

        # 2. 构造标签 y：教师评分（使用原始 df 计算）
        teacher_scores = compute_teacher_scores_from_df(feature_df, max_round=self.max_round)

        # 返回：(输入, 标签)
        return (
            torch.tensor(X_norm, dtype=torch.float32),         # 形状 (num_clients, 11)
            torch.tensor(teacher_scores, dtype=torch.float32)  # 形状 (num_clients,)
        )


# ============================================================
# DataLoader 构造函数
# ============================================================
def build_dataloader(batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    """
    构建用于训练的 DataLoader。
    一个 batch = 若干个 round（每个 round 是 (100, 11) 的矩阵）。
    """
    dataset = FLRoundDataset("../dataset", start_round=10, end_round=110)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
