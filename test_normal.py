# ===================================================
# 基础版联邦模拟：随机选择客户端，生成训练数据矩阵
# 用途：为 selector 训练准备数据（不使用 selector 模型）
# 这里改用 Fashion-MNIST，并将数据矩阵保存到 dataset/fsmnist
# ===================================================
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from flwr.server.strategy.aggregate import aggregate
from matplotlib import pyplot as plt

from cnn import Net, test
from client_selector import select_clients
from cust import FlowerClient

# 全程 CPU
DEVICE = torch.device("cpu")


def prepare_fashion_dataset(
    num_partitions: int = 100,
    batch_size: int = 32,
    val_ratio: float = 0.1,
    data_path: str = "./data_fashion",
):
    """Partition Fashion-MNIST into num_partitions clients, each with train/val, plus test loader."""
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

trainloaders, valloaders, testloader = prepare_fashion_dataset_non_iid()
NUM_CLIENTS = len(trainloaders)
CLIENTS_PER_ROUND = max(1, int(0.1 * NUM_CLIENTS))
NUM_ROUNDS = 120


def set_model_parameters(model: Net, parameters) -> Net:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate_global(parameters) -> Tuple[float, float]:
    model = Net(num_classes=10)
    set_model_parameters(model, parameters)
    return test(model, testloader)


def run_fedavg(num_rounds: int = NUM_ROUNDS) -> List[Tuple[int, float]]:
    """随机选择客户端（前 10 轮覆盖全部），生成指标矩阵并保存。"""
    clients = [FlowerClient(trainloaders[i], valloaders[i]) for i in range(NUM_CLIENTS)]
    global_params = clients[0].get_parameters({})

    # trackers
    last_selected_round = np.full(NUM_CLIENTS, -1, dtype=int)
    last_loss = np.full(NUM_CLIENTS, np.nan)
    last_acc = np.full(NUM_CLIENTS, np.nan)
    last_data = np.full(NUM_CLIENTS, np.nan)
    last_grad = np.full(NUM_CLIENTS, np.nan)
    last_cosine = np.full(NUM_CLIENTS, np.nan)
    last_epochs = np.full(NUM_CLIENTS, np.nan)
    last_deposit = np.full(NUM_CLIENTS, np.nan)
    participation_count = np.zeros(NUM_CLIENTS, dtype=int)

    history: List[Tuple[int, float]] = []

    out_dir = Path("dataset") / "fsmnist"
    out_dir.mkdir(parents=True, exist_ok=True)

    for rnd in range(1, num_rounds + 1):
        sampled = select_clients(
            round_idx=rnd,
            num_clients=NUM_CLIENTS,
            clients_per_round=CLIENTS_PER_ROUND,
            warmup_rounds=10,
        )
        print(f"Round {rnd}: {sampled}")
# === 【关键步骤 1】在一切开始前，备份“状态快照”用于存 CSV ===
        # 这些变量专门用于保存到 "Pre-Training" 状态
        snapshot_selected_round = last_selected_round.copy() 
        snapshot_loss = last_loss.copy()
        snapshot_accuracy = last_acc.copy()
        snapshot_data_size = last_data.copy()
        snapshot_grad_norm = last_grad.copy()
        snapshot_participation = participation_count.copy() # 如果你想预测基于历史参与次数

        # start from last round values
        loss_arr = last_loss.copy()
        acc_arr = last_acc.copy()
        data_arr = last_data.copy()
        grad_arr = last_grad.copy()
        cos_arr = last_cosine.copy()
        epochs_arr = last_epochs.copy()
        deposit_arr = last_deposit.copy()
        flag = np.zeros(NUM_CLIENTS, dtype=int)

        # snapshot pre-training states for selected clients to avoid leakage
        pre_training_states = {}
        updates = {}
        base = global_params
        fit_results: list[tuple[list[np.ndarray], int, dict]] = []

        for cid in sampled:
            pre_training_states[cid] = clients[cid].get_parameters({})
            params, num_examples, info = clients[cid].fit(global_params, {})
            fit_results.append((params, num_examples, info))

            loss_arr[cid] = info.get("loss", np.nan)
            acc_arr[cid] = info.get("accuracy", np.nan)
            data_arr[cid] = info.get("data_size", np.nan)
            grad_arr[cid] = info.get("grad_norm", np.nan)
            epochs_arr[cid] = info.get("local_epochs", np.nan)
            deposit_arr[cid] = np.random.uniform(0, 100)

            flag[cid] = 1
            last_selected_round[cid] = rnd
            updates[cid] = np.concatenate(
                [(params[i] - base[i]).ravel() for i in range(len(params))]
            )

        # cosine similarity vs mean update of selected clients
        if updates:
            U = np.stack(list(updates.values()))
            g = U.mean(axis=0)
            ng = np.linalg.norm(g)
            for cid, upd in updates.items():
                nl = np.linalg.norm(upd)
                cos_arr[cid] = np.dot(upd, g) / (nl * ng) if nl > 0 and ng > 0 else np.nan

        participation_count[flag == 1] += 1

        # update trackers
        last_loss, last_acc = loss_arr, acc_arr
        last_data, last_grad = data_arr, grad_arr
        last_cosine, last_epochs = cos_arr, epochs_arr
        last_deposit = deposit_arr

        # FedAvg aggregation（使用新训练的参数）
        global_params = aggregate([(p, n) for (p, n, _) in fit_results])

        # global eval
        loss, acc = evaluate_global(global_params)
        history.append((rnd, acc))
        print(f"[Round {rnd}] loss={loss:.4f}, acc={acc:.4f}")

        # leave-one-out: compute contribution for sampled clients
        contribution = np.full(NUM_CLIENTS, np.nan)
        for idx, cid in enumerate(sampled):
            others = [(p, n) for j, (p, n, _) in enumerate(fit_results) if j != idx]
            if others:
                w_minus = aggregate(others)
            else:
                w_minus = global_params
            _, acc_minus = evaluate_global(w_minus)
            contribution[cid] = acc - acc_minus

        # save per-round metrics (NaN -> 0 for readability), using pre-training snapshot for selected clients
        loss_out = np.nan_to_num(loss_arr, nan=0.0)
        acc_out = np.nan_to_num(acc_arr, nan=0.0)
        data_out = np.nan_to_num(data_arr, nan=0.0)
        grad_out = np.nan_to_num(grad_arr, nan=0.0)
        cos_out = np.nan_to_num(cos_arr, nan=0.0)
        epochs_out = np.nan_to_num(epochs_arr, nan=0.0)
        contrib_out = np.nan_to_num(contribution, nan=0.0)

        # === 【关键步骤 2】保存时，使用上面的快照变量 ===
        metrics_matrix = np.column_stack(
            [
                np.arange(NUM_CLIENTS),
                np.full(NUM_CLIENTS, rnd, dtype=int),
                snapshot_selected_round,  # <--- 改这里！用快照，不要用 last_selected_round
                flag,
                snapshot_participation,   # <--- 改这里！用快照
                deposit_arr,              # Deposit 如果是当前环境属性，可以用新的
                np.nan_to_num(snapshot_loss, nan=0.0),      # <--- 改这里！
                np.nan_to_num(snapshot_accuracy, nan=0.0),  # <--- 改这里！
                np.nan_to_num(snapshot_data_size, nan=0.0), # <--- 改这里！
                np.nan_to_num(snapshot_grad_norm, nan=0.0), # <--- 改这里！
                cos_out,                  # Cosine 是本轮计算结果，可以用新的
                epochs_out,               # Epochs 是本轮行为，可以用新的
                contrib_out,              # Contribution 是 Label，必须是新的
            ]
        )
        out_path = out_dir / f"round_{rnd:02d}_metrics.csv"
        header = (
            "client_id,current_round,last_selected_round,selected,participation_count,"
            "deposit,loss,accuracy,data_size,grad_norm,cosine_to_global_update,local_epochs,contribution"
        )
        fmt = [
            "%d",
            "%d",
            "%d",
            "%d",
            "%d",
            "%.3f",
            "%.3f",
            "%.3f",
            "%.0f",
            "%.3f",
            "%.4f",
            "%.0f",
            "%.6f",
        ]
        np.savetxt(out_path, metrics_matrix, delimiter=",", header=header, comments="", fmt=fmt)

    return history


def plot_history(history: List[Tuple[int, float]]) -> None:
    rounds = [r for r, _ in history]
    acc = [100.0 * a for _, a in history]
    plt.plot(rounds, acc)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("FedAvg - random selection (Fashion-MNIST)")
    plt.tight_layout()
    plt.savefig(Path("accuary") / "accuracy_curve_fsmnist.png")
    plt.close()


if __name__ == "__main__":
    print("Flower version:", fl.__version__)
    print("模式：随机选择客户端，生成训练数据矩阵")
    hist = run_fedavg()
    # save accuracy curve
    acc_dir = Path("accuary")
    acc_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(acc_dir / "accuracy_fsmnist.csv", np.array(hist), delimiter=",", header="round,accuracy", comments="")
    plot_history(hist)
