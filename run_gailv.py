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

# ----------------------- 2. 系统参数 -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
DEVICE = torch.device("cpu")

WARMUP_ROUNDS = 20
NUM_ROUNDS = 150
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10

TRAIN_EPOCHS_PER_ROUND = 30
BATCH_SIZE = 64
LR = 0.005
REWARD_SCALE = 2000.0
ALPHA_INIT = 0.5
ALPHA_DECAY = 0.98

# ----------------------- 3. 输出目录 -----------------------
DATA_OUT_DIR = Path("dataset") / "da_fcb_logs_GAILV"
if DATA_OUT_DIR.exists():
    shutil.rmtree(DATA_OUT_DIR)
DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

ACC_OUT_DIR = Path("accuary")
if ACC_OUT_DIR.exists():
    shutil.rmtree(ACC_OUT_DIR)
ACC_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------- 4. NeuralBandit（MLP） -----------------------
class NeuralBandit(nn.Module):
    def __init__(self, input_dim=9):
        super(NeuralBandit, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------- 5. 数据集 Non-IID 划分 -----------------------
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

# ----------------------- 6. PCDI -----------------------
def calculate_client_pcdi(dataloader, num_classes=10):
    label_counts = np.zeros(num_classes)
    for _, y in dataloader:
        y_np = y.numpy()
        label_counts += np.bincount(y_np, minlength=num_classes)
    avg_lambda = np.mean(label_counts)
    return np.sum((label_counts - avg_lambda)**2)

# ----------------------- 7. 构造 9 维特征 -----------------------
def build_features_tensor(
    rnd, last_selected, participation, loss, acc, data, grad, cosine, epochs, pcdi
):
    staleness = rnd - np.where(last_selected >= 0, last_selected, 0)

    feats = np.column_stack([
        loss, acc, staleness, data, grad, cosine, epochs, participation, pcdi
    ]).astype(np.float32)

    feats = np.nan_to_num(feats, nan=0.0)

    feats[:, 0] = np.clip(feats[:, 0] / 5.0, 0, 1)
    feats[:, 1] = np.clip(feats[:, 1], 0, 1)
    feats[:, 2] = np.clip(feats[:, 2] / 50.0, 0, 1)
    feats[:, 3] = np.clip(feats[:, 3] / 2000.0, 0, 1)
    feats[:, 4] = np.clip(feats[:, 4] / 20.0, 0, 1)
    feats[:, 5] = (feats[:, 5] + 1.0) / 2.0
    feats[:, 6] = np.clip(feats[:, 6] / 10.0, 0, 1)
    feats[:, 7] = np.clip(feats[:, 7] / 50.0, 0, 1)
    feats[:, 8] = np.clip(feats[:, 8] / 100000.0, 0, 1)
    
    return torch.tensor(feats)

# ----------------------- 8. 主流程 -----------------------
def run_online_learning_fl():

    print("Preparing Data...")
    trainloaders, valloaders, testloader = prepare_fashion_dataset_non_iid(NUM_CLIENTS, alpha=0.1)
    clients = [FlowerClient(tl, vl) for tl, vl in zip(trainloaders, valloaders)]
    global_params = clients[0].get_parameters({})

    print("Calculating PCDI...")
    client_pcdi_arr = np.zeros(NUM_CLIENTS)
    for i, loader in enumerate(trainloaders):
        client_pcdi_arr[i] = calculate_client_pcdi(loader)

    predictor = NeuralBandit(input_dim=9).to(DEVICE)
    predictor_optim = optim.Adam(predictor.parameters(), lr=LR)
    predictor_criterion = nn.MSELoss()

    replay_buffer_X = []
    replay_buffer_y = []

    # 状态
    last_selected_round = np.full(NUM_CLIENTS, -1, dtype=int)
    last_loss = np.zeros(NUM_CLIENTS)
    last_acc = np.zeros(NUM_CLIENTS)
    last_data = np.zeros(NUM_CLIENTS)
    last_grad = np.zeros(NUM_CLIENTS)
    last_cosine = np.zeros(NUM_CLIENTS)
    last_epochs = np.full(NUM_CLIENTS, 1.0)
    participation_count = np.zeros(NUM_CLIENTS, dtype=int)

    warmup_perm = np.random.permutation(NUM_CLIENTS)
    history_acc = []

    print(f"=== DA-FCB (Reward = Local Accuracy Gain) ===")

    for rnd in range(1, NUM_ROUNDS + 1):

        # ---------- 保存上一轮精度，用来算 reward ----------
        prev_client_acc = last_acc.copy()

        # ---------- 构造特征 ----------
        current_features_tensor = build_features_tensor(
            rnd, last_selected_round, participation_count,
            last_loss, last_acc, last_data, last_grad, last_cosine, last_epochs,
            client_pcdi_arr
        ).to(DEVICE)

        # ---------- Step A: 选人 ----------
        if rnd <= WARMUP_ROUNDS:
            start = (rnd - 1) * CLIENTS_PER_ROUND
            sampled_indices = [warmup_perm[i % NUM_CLIENTS] for i in range(start, start + CLIENTS_PER_ROUND)]
            print(f"Round {rnd} [Warm-up]: Random")
        else:
            predictor.eval()
            with torch.no_grad():
                pred_rewards = predictor(current_features_tensor).squeeze(-1).cpu().numpy()

            current_alpha = max(0.05, ALPHA_INIT * (ALPHA_DECAY ** (rnd - WARMUP_ROUNDS)))
            staleness_norm = current_features_tensor[:, 2].cpu().numpy()

            recently_selected = (rnd - last_selected_round) < 5
            penalty = np.where(recently_selected, 0.5, 0.0)

            # ---------- Step A: 选人（Softmax 概率抽样） ----------
            if rnd <= WARMUP_ROUNDS:
                start = (rnd - 1) * CLIENTS_PER_ROUND
                sampled_indices = [warmup_perm[i % NUM_CLIENTS] for i in range(start, start + CLIENTS_PER_ROUND)]
                print(f"Round {rnd} [Warm-up]: Random")
            else:
                predictor.eval()
                with torch.no_grad():
                    pred_rewards = predictor(current_features_tensor).squeeze(-1).cpu().numpy()

                current_alpha = max(0.05, ALPHA_INIT * (ALPHA_DECAY ** (rnd - WARMUP_ROUNDS)))
                staleness_norm = current_features_tensor[:, 2].cpu().numpy()

                recently_selected = (rnd - last_selected_round) < 5
                penalty = np.where(recently_selected, 0.5, 0.0)

                # ---------------------
                # ★ 计算最终分数（score）
                # ---------------------
                ucb_scores = pred_rewards + current_alpha * staleness_norm - penalty

                # ---------------------
                # ★ Softmax → 概率分布
                # ---------------------
                temperature = 0.7  # 可调，越小越偏向高分
                scores_scaled = ucb_scores / temperature

                # 数值稳定
                scores_scaled -= scores_scaled.max()

                probs = np.exp(scores_scaled)
                probs = probs / probs.sum()  # 归一化成概率

                # ---------------------
                # ★ 按概率抽样客户端
                # ---------------------
                sampled_indices = np.random.choice(
                    NUM_CLIENTS,
                    size=CLIENTS_PER_ROUND,
                    replace=False,
                    p=probs
                ).tolist()

                print(f"Round {rnd} [Prob-Select]: Temperature={temperature}")

            print(f"Round {rnd} [Bandit]: Alpha={current_alpha:.3f}")

        # ---------- Step B: 本地训练 ----------
        fit_results = []
        updates = {}
        flag = np.zeros(NUM_CLIENTS, dtype=int)

        selected_X_snapshot = current_features_tensor[sampled_indices].cpu().numpy()

        # 新 reward
        rewards_map = {}

        for idx, cid in enumerate(sampled_indices):

            params, n_examples, info = clients[cid].fit(global_params, {})
            fit_results.append((params, n_examples, info))

            curr_local_acc = info.get("accuracy", 0.0)

            # ★★★ 新 reward：局部精度提升 ★★★
            if participation_count[cid] == 0:
                reward = curr_local_acc
            else:
                reward = curr_local_acc - prev_client_acc[cid]

            rewards_map[cid] = reward

            # 更新状态
            last_loss[cid] = info.get("loss", 0.0)
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

        # ---------- Step C: cosine 更新 ----------
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

        # ---------- Step E: 写入 CSV ----------
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

        # ---------- Step F: 训练 Neural Bandit ----------
        if rnd > WARMUP_ROUNDS:
            for i, cid in enumerate(sampled_indices):
                replay_buffer_X.append(selected_X_snapshot[i])
                replay_buffer_y.append(rewards_map[cid] * REWARD_SCALE)

            if len(replay_buffer_X) > 2000:
                replay_buffer_X = replay_buffer_X[-2000:]
                replay_buffer_y = replay_buffer_y[-2000:]

            predictor.train()
            ds = TensorDataset(
                torch.tensor(np.array(replay_buffer_X), dtype=torch.float32),
                torch.tensor(np.array(replay_buffer_y), dtype=torch.float32).view(-1, 1)
            )
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

            for _ in range(TRAIN_EPOCHS_PER_ROUND):
                for bx, by in dl:
                    predictor_optim.zero_grad()
                    out = predictor(bx)
                    loss = predictor_criterion(out, by)
                    loss.backward()
                    predictor_optim.step()

    print("Done.")
    return history_acc


# ----------------------- 9. 主入口 -----------------------
if __name__ == "__main__":
    hist = run_online_learning_fl()

    acc_path = ACC_OUT_DIR / "accuracy_GAILV.csv"
    np.savetxt(acc_path, np.array(hist), delimiter=",",
               header="round,test_accuracy", comments="", fmt=["%d", "%.6f"])

    print(f"Saved accuracy to {acc_path}")

    rounds = [x[0] for x in hist]
    accs = [x[1] for x in hist]

    plt.plot(rounds, accs, label="DA-FCB(New Reward)", color="red")
    plt.grid()
    plt.legend()
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("DA-FCB with Local Accuracy Gain Reward")
    plt.savefig(ACC_OUT_DIR / "accuracy_plot.png")
    print("Accuracy plot saved.")
