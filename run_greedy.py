import os
import copy
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple
import random
import shutil

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from flwr.server.strategy.aggregate import aggregate
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

# ----------------------- 1. 导入 CNN 与客户端 -----------------------
try:
    from cnn import Net, test
    from cust import FlowerClient
except ImportError:
    print("⚠️ 警告: 未找到 cnn.py 或 cust.py，请确保文件在同一目录下。")
    pass

# ----------------------- 2. 系统参数 (保持一致) -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DEVICE = torch.device("cpu")

WARMUP_ROUNDS = 20
NUM_ROUNDS = 150
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10

TRAIN_EPOCHS_PER_ROUND = 30
BATCH_SIZE = 64
LR = 0.005
# REWARD_SCALE, ALPHA 等参数在Greedy中不使用，但保留变量名位置以示结构一致

# ----------------------- 3. 输出目录 (修改以区分) -----------------------
# 注意：这里改成了 greedy_logs
DATA_OUT_DIR = Path("dataset") / "greedy_logs"
if DATA_OUT_DIR.exists():
    shutil.rmtree(DATA_OUT_DIR)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

ACC_OUT_DIR = Path("accuary") / "greedy"
if ACC_OUT_DIR.exists():
    shutil.rmtree(ACC_OUT_DIR)
ACC_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- 4. (已移除 NeuralBandit) -----------------------
# Greedy 策略不需要 MLP 网络

# ----------------------- 5. 数据集 Non-IID 划分 (保持一致) -----------------------
def prepare_fashion_dataset_non_iid(
    num_partitions: int = 100,
    batch_size: int = 32,
    alpha: float = 0.1,
    data_path: str = "./data_fashion",
):
    print(f"正在准备数据 (Non-IID Alpha={alpha})...")
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = FashionMNIST(data_path, train=True, download=True, transform=tr)
    testset = FashionMNIST(data_path, train=False, download=True, transform=tr)

    min_size = 0
    labels = trainset.targets
    num_classes = 10
    
    while min_size < 10:
        idx_batch = [[] for _ in range(num_partitions)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_partitions))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    trainloaders, valloaders = [], []
    for idx_j in idx_batch:
        subset = torch.utils.data.Subset(trainset, idx_j)
        num_total = len(subset)
        num_val = int(0.1 * num_total)
        num_train = num_total - num_val
        for_train, for_val = random_split(subset, [num_train, num_val], torch.Generator().manual_seed(2023))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False))

    testloader = DataLoader(testset, batch_size=128)
    return trainloaders, valloaders, testloader 

# ----------------------- 6. PCDI (保持一致) -----------------------
def calculate_client_pcdi(dataloader, num_classes=10):
    label_counts = np.zeros(num_classes)
    for _, y in dataloader:
        y_np = y.numpy()
        label_counts += np.bincount(y_np, minlength=num_classes)
    avg_lambda = np.mean(label_counts)
    return np.sum((label_counts - avg_lambda)**2)

# ----------------------- 7. (Greedy 不需要构建 Tensor 特征) -----------------------
# 但为了 CSV 输出一致，我们会在写入时直接使用 numpy 数组

# ----------------------- 8. 主流程 (Greedy) -----------------------
def run_greedy_fl():

    print("Preparing Data...")
    trainloaders, valloaders, testloader = prepare_fashion_dataset_non_iid(NUM_CLIENTS, alpha=0.1)
    clients = [FlowerClient(tl, vl) for tl, vl in zip(trainloaders, valloaders)]
    global_params = clients[0].get_parameters({})

    print("Calculating PCDI...")
    client_pcdi_arr = np.zeros(NUM_CLIENTS)
    for i, loader in enumerate(trainloaders):
        client_pcdi_arr[i] = calculate_client_pcdi(loader)

    # 状态
    last_selected_round = np.full(NUM_CLIENTS, -1, dtype=int)
    last_loss = np.zeros(NUM_CLIENTS)  # Greedy 的核心依据
    last_acc = np.zeros(NUM_CLIENTS)
    last_data = np.zeros(NUM_CLIENTS)
    last_grad = np.zeros(NUM_CLIENTS)
    last_cosine = np.zeros(NUM_CLIENTS)
    last_epochs = np.full(NUM_CLIENTS, 1.0)
    participation_count = np.zeros(NUM_CLIENTS, dtype=int)

    warmup_perm = np.random.permutation(NUM_CLIENTS)
    history_acc = []

    print(f"=== Greedy Selection (Max Loss Strategy) ===")

    for rnd in range(1, NUM_ROUNDS + 1):

        # ---------- Step A: 选人 (核心改动) ----------
        if rnd <= WARMUP_ROUNDS:
            # 热身阶段：随机选择，为了初始化各个客户端的 Loss
            start = (rnd - 1) * CLIENTS_PER_ROUND
            sampled_indices = [warmup_perm[i % NUM_CLIENTS] for i in range(start, start + CLIENTS_PER_ROUND)]
            print(f"Round {rnd} [Warm-up]: Random")
        else:
            # 贪婪阶段：选择 Loss 最大的 Top-K
            # argsort 返回从小到大的索引，取最后 CLIENTS_PER_ROUND 个即为最大的
            # 注意：如果客户端在 Warmup 没被选中过，Loss 为 0，Greedy 永远不会选它 (Starvation)
            # 这正是 Greedy 的缺点，也是你要对比的点
            sampled_indices = np.argsort(last_loss)[-CLIENTS_PER_ROUND:].tolist()
            
            # 为了调试，打印一下选中的客户端平均 Loss
            avg_selected_loss = np.mean(last_loss[sampled_indices])
            print(f"Round {rnd} [Greedy]: Selecting Max Loss clients. Avg Loss: {avg_selected_loss:.4f}")

        # ---------- Step B: 本地训练 ----------
        fit_results = []
        updates = {}
        flag = np.zeros(NUM_CLIENTS, dtype=int)
        
        # 仅仅为了CSV记录用
        rewards_map = {} 

        for idx, cid in enumerate(sampled_indices):

            params, n_examples, info = clients[cid].fit(global_params, {})
            fit_results.append((params, n_examples, info))

            curr_local_acc = info.get("accuracy", 0.0)
            
            # Greedy 不需要计算 reward，但为了保持 CSV 格式一致，我们填 0 或填精度
            rewards_map[cid] = curr_local_acc 

            # 更新状态 (最重要的是 last_loss)
            last_loss[cid] = info.get("loss", 0.0)  # 更新 Loss，供下一轮贪婪选择使用
            last_acc[cid] = curr_local_acc
            last_data[cid] = info.get("data_size", 0.0)
            last_grad[cid] = info.get("grad_norm", 0.0)
            last_epochs[cid] = info.get("local_epochs", 1.0)
            last_selected_round[cid] = rnd
            participation_count[cid] += 1
            flag[cid] = 1

            # 记录参数更新方向（用于 cosine）
            flat_diff = np.concatenate([p.ravel() for p in params]) - \
                        np.concatenate([p.ravel() for p in global_params])
            updates[cid] = flat_diff

        # ---------- Step C: cosine 更新 (仅记录，不影响决策) ----------
        if updates:
            u_mat = np.stack(list(updates.values()))
            avg_u = u_mat.mean(axis=0)
            norm_avg = np.linalg.norm(avg_u)
            for cid, u in updates.items():
                nu = np.linalg.norm(u)
                if nu > 1e-9 and norm_avg > 1e-9:
                    last_cosine[cid] = np.dot(u, avg_u) / (nu * norm_avg)
                else:
                    last_cosine[cid] = 0.0

        # ---------- Step D: 聚合更新 & 全局评估 ----------
        global_params = aggregate([(p, n) for p, n, _ in fit_results])

        def eval_fn(params):
            net = Net()
            sd = OrderedDict({k: torch.tensor(v) for k, v in zip(net.state_dict().keys(), params)})
            net.load_state_dict(sd)
            return test(net, testloader)

        curr_loss, curr_acc = eval_fn(global_params)
        history_acc.append((rnd, curr_acc))
        print(f"---> Round {rnd} Global Acc: {curr_acc:.4f}")

        # ---------- Step E: 写入 CSV (保持一致) ----------
        contributions_map = np.zeros(NUM_CLIENTS)
        for cid in sampled_indices:
            contributions_map[cid] = rewards_map[cid]

        metrics_matrix = np.column_stack([
            np.arange(NUM_CLIENTS),
            np.full(NUM_CLIENTS, rnd, dtype=int),
            last_selected_round,
            flag,
            participation_count,
            np.nan_to_num(last_loss),
            np.nan_to_num(last_acc),
            np.nan_to_num(last_data),
            np.nan_to_num(last_grad),
            np.nan_to_num(last_cosine),
            last_epochs,
            client_pcdi_arr,
            contributions_map
        ])

        out_path = DATA_OUT_DIR / f"round_{rnd:02d}_metrics.csv"
        header = (
            "client_id,current_round,last_selected_round,selected,participation,"
            "loss,accuracy,data_size,grad_norm,cosine,local_epochs,pcdi,contribution"
        )
        fmt = ["%d", "%d", "%d", "%d", "%d", "%.4f", "%.4f", "%.0f", "%.4f",
               "%.4f", "%.1f", "%.2f", "%.6f"]

        np.savetxt(out_path, metrics_matrix, delimiter=",", header=header, comments="", fmt=fmt)

        # Greedy 没有 Step F (训练 MLP)

    print("Greedy Done.")
    return history_acc


# ----------------------- 9. 主入口 -----------------------
if __name__ == "__main__":
    hist = run_greedy_fl()

    acc_path = ACC_OUT_DIR / "accuracy_greedy.csv"
    np.savetxt(acc_path, np.array(hist), delimiter=",",
               header="round,test_accuracy", comments="", fmt=["%d", "%.6f"])

    print(f"Saved accuracy to {acc_path}")

    rounds = [x[0] for x in hist]
    accs = [x[1] for x in hist]

    plt.plot(rounds, accs, label="Greedy (Max Loss)", color="blue")
    plt.grid()
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Greedy Selection Strategy")
    plt.savefig(ACC_OUT_DIR / "accuracy_plot_greedy.png")
    print("Accuracy plot saved.")