# ===================================================
# 强制使用 CPU，并避免任何 CUDA 加载（像 test_eval 一样）
# ===================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from collections import OrderedDict
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import numpy as np
import flwr as fl
import torch
from matplotlib import pyplot as plt
from flwr.server.strategy.aggregate import aggregate
import matplotlib

from cnn import Net, test
from client_selector import select_clients
from cust import FlowerClient
from fenqu import testloader, trainloaders, valloaders

# --- 让行为与 test_eval 完全一致 ---
EVAL_DIR = Path(__file__).parent / "eval"
if str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from scoring_cnn import CNNScoringNet  # 被动加载，不调用 .to(cuda)


# 默认 CPU
DEVICE = torch.device("cpu")

# 缓存 selector
_selector_model: Optional[torch.nn.Module] = None


# ===================================================
# 像 test_eval 一样加载模型（关键！！！）
# ===================================================
def load_selector_cpu_only() -> torch.nn.Module:
    global _selector_model
    if _selector_model is not None:
        return _selector_model

    weight_path = EVAL_DIR / "saved_models" / "scoring_cnn.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"模型不存在: {weight_path}")

    # ⭐ 关键代码：完全模仿 test_eval 的加载方式 ⭐
    model = CNNScoringNet(in_features=11)
    state = torch.load(weight_path, map_location="cpu")  # CPU ONLY
    model.load_state_dict(state)
    model.eval()

    _selector_model = model
    return model


# ===================================================
# FedAvg 相关函数
# ===================================================
def set_model_parameters(model: Net, parameters):
    keys = model.state_dict().keys()
    state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate_global(parameters):
    model = Net(num_classes=10)
    set_model_parameters(model, parameters)
    return test(model, testloader)


# ===================================================
# 特征构造方式保持不变
# ===================================================
def build_feature_matrix(
    rnd, last_selected_round, last_selected_flag, participation_count,
    deposit_arr, loss_arr, acc_arr, data_arr, grad_arr, cosine_arr, epochs_arr
):
    feats = np.column_stack([
        np.full(len(loss_arr), rnd, dtype=np.float32),
        last_selected_round.astype(np.float32),
        last_selected_flag.astype(np.float32),
        participation_count.astype(np.float32),
        np.nan_to_num(deposit_arr, nan=0.0).astype(np.float32),
        np.nan_to_num(loss_arr, nan=0.0).astype(np.float32),
        np.nan_to_num(acc_arr, nan=0.0).astype(np.float32),
        np.nan_to_num(data_arr, nan=0.0).astype(np.float32),
        np.nan_to_num(grad_arr, nan=0.0).astype(np.float32),
        np.nan_to_num(cosine_arr, nan=0.0).astype(np.float32),
        np.nan_to_num(epochs_arr, nan=0.0).astype(np.float32),
    ])

    # min-max normalize
    for i in range(feats.shape[1]):
        col = feats[:, i]
        mn, mx = col.min(), col.max()
        feats[:, i] = (col - mn) / (mx - mn) if mx > mn else 0.0
    return feats


# ===================================================
# 使用 selector 模型选择客户端（CPU-only）
# ===================================================
def sample_clients_by_model(
    rnd, k,
    last_selected_round, last_selected_flag, participation_count,
    deposit_arr, loss_arr, acc_arr, data_arr, grad_arr, cosine_arr, epochs_arr
) -> tuple[list[int], np.ndarray]:
    try:
        model = load_selector_cpu_only()
    except FileNotFoundError as e:
        print(f"{e}，改为随机采样")
        probs = np.full(len(loss_arr), 1.0 / len(loss_arr), dtype=np.float32)
        chosen = np.random.choice(len(loss_arr), size=k, replace=False).tolist()
        return chosen, probs

    feats = build_feature_matrix(
        rnd, last_selected_round, last_selected_flag, participation_count,
        deposit_arr, loss_arr, acc_arr, data_arr, grad_arr, cosine_arr, epochs_arr
    )

    X = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)  # CPU tensor

    with torch.no_grad():
        scores = model(X).squeeze(0)

    probs = torch.softmax(scores, dim=0)
    probs /= probs.sum()

    # 选分最高的 k 个客户端
    topk_idx = torch.topk(scores, k=k, largest=True).indices
    return topk_idx.cpu().numpy().tolist(), probs.cpu().numpy()


def plot_selection_probs(rnd: int, probs: np.ndarray, sampled: list[int], out_dir: Path) -> None:
    """画出本轮选择概率，选中客户端高亮。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    colors = ["tab:red" if i in sampled else "tab:gray" for i in range(len(probs))]
    plt.bar(np.arange(len(probs)), probs, color=colors)
    plt.xlabel("Client ID")
    plt.ylabel("Selection Prob.")
    plt.title(f"Round {rnd} selection probabilities")
    plt.tight_layout()
    plt.savefig(out_dir / f"probs_round_{rnd:03d}.png")
    plt.close()


# ===================================================
# FedAvg 主流程（完全 CPU，与 test_eval 相同）
# ===================================================
def run_fedavg(num_rounds=110):
    N = len(trainloaders)
    K = max(1, min(10, N))  # 每轮固定选分最高的 10 个（不足则全选）

    clients = [FlowerClient(trainloaders[i], valloaders[i]) for i in range(N)]
    global_params = clients[0].get_parameters({})

    # state trackers
    last_selected_round = np.full(N, -1)
    last_loss = np.full(N, np.nan)
    last_acc = np.full(N, np.nan)
    last_data = np.full(N, np.nan)
    last_grad = np.full(N, np.nan)
    last_cosine = np.full(N, np.nan)
    last_epochs = np.full(N, np.nan)
    last_deposit = np.full(N, np.nan)
    last_flag = np.zeros(N)
    count = np.zeros(N)

    history = []
    acc_csv: list[tuple[int, float]] = []
    prob_dir = Path("accuary")
    prob_dir.mkdir(parents=True, exist_ok=True)

    for rnd in range(1, num_rounds + 1):

        if rnd <= 10:
            sampled = select_clients(
                round_idx=rnd,
                num_clients=N,
                clients_per_round=K,
                warmup_rounds=10
            )
            probs = np.full(N, 1.0 / N, dtype=np.float32)
        else:
            sampled, probs = sample_clients_by_model(
                rnd, K,
                last_selected_round, last_flag, count,
                last_deposit, last_loss, last_acc,
                last_data, last_grad, last_cosine, last_epochs
            )

        print(f"Round {rnd}: {sampled}")

        # prepare arrays for this round
        loss_arr = last_loss.copy()
        acc_arr = last_acc.copy()
        data_arr = last_data.copy()
        grad_arr = last_grad.copy()
        cos_arr = last_cosine.copy()
        epochs_arr = last_epochs.copy()
        deposit_arr = last_deposit.copy()
        flag = np.zeros(N)

        updates = {}
        base = global_params

        for cid in sampled:
            params, num_examples, info = clients[cid].fit(global_params, {})
            loss_arr[cid] = info.get("loss", np.nan)
            acc_arr[cid] = info.get("accuracy", np.nan)
            data_arr[cid] = info.get("data_size", np.nan)
            grad_arr[cid] = info.get("grad_norm", np.nan)
            epochs_arr[cid] = info.get("local_epochs", np.nan)
            deposit_arr[cid] = np.random.uniform(0, 100)

            flag[cid] = 1
            last_selected_round[cid] = rnd
            updates[cid] = np.concatenate([(params[i] - base[i]).ravel() for i in range(len(params))])

        # cosine similarity
        if updates:
            U = np.stack(list(updates.values()))
            g = U.mean(axis=0)
            ng = np.linalg.norm(g)
            for cid, upd in updates.items():
                nl = np.linalg.norm(upd)
                cos_arr[cid] = np.dot(upd, g) / (nl * ng) if nl > 0 and ng > 0 else np.nan

        count[flag == 1] += 1
        last_flag = flag.copy()
        last_loss, last_acc = loss_arr, acc_arr
        last_data, last_grad = data_arr, grad_arr
        last_cosine, last_epochs = cos_arr, epochs_arr
        last_deposit = deposit_arr

        global_params = aggregate([(clients[cid].get_parameters({}), 1) for cid in sampled])

        loss, acc = evaluate_global(global_params)
        history.append((rnd, acc))
        acc_csv.append((rnd, acc))
        print(f"[Round {rnd}] loss={loss:.4f}, acc={acc:.4f}")

        # 每 10 轮保存一次概率图
        if rnd % 10 == 0:
            plot_selection_probs(rnd, probs, sampled, prob_dir)

    return history


def plot_history(hist):
    rounds = [r for r, _ in hist]
    acc = [a * 100 for _, a in hist]
    plt.plot(rounds, acc)
    plt.grid()
    plt.xlabel("Round")
    plt.ylabel("Accuracy (%)")
    plt.title("FedAvg (CPU only, same behavior as test_eval)")
    plt.tight_layout()
    plt.savefig(Path("accuary") / "accuracy_curve.png")
    plt.close()


# ===================================================
# main
# ===================================================
if __name__ == "__main__":
    print("Flower version:", fl.__version__)
    print("运行模式：CPU-only（完全和 test_eval 一样，不会触发任何 CUDA DLL）")

    hist = run_fedavg()
    # 保存准确率 CSV
    acc_dir = Path("accuary")
    acc_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(acc_dir / "accuracy.csv", np.array(hist), delimiter=",", header="round,accuracy", comments="")
    plot_history(hist)
