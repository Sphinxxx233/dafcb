import torch
import torch.nn.functional as F
from pathlib import Path

from dataloader_selector import build_dataloader
from scoring_cnn import CNNScoringNet


MODEL_PATH = Path("saved_models/scoring_cnn.pth")


def evaluate(batch_size: int = 1) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"首选设备: {device}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到已训练模型权重: {MODEL_PATH}")

    dataloader = build_dataloader(batch_size=batch_size, shuffle=False)

    model = CNNScoringNet(in_features=11).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for idx, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            try:
                pred = model(X)
            except RuntimeError as exc:
                if device.type == "cuda":
                    print(f"CUDA 运行失败，自动切换 CPU 重新评估: {exc}")
                    device = torch.device("cpu")
                    model = model.cpu()
                    X = X.cpu()
                    y = y.cpu()
                    pred = model(X)
                else:
                    raise

            loss = F.mse_loss(pred, y, reduction="mean")

            total_loss += loss.item()
            total_batches += 1

            if idx == 0:
                print(f"X shape: {X.shape}")  # torch.Size([B, num_clients, 11])
                print(f"y shape: {y.shape}")  # torch.Size([B, num_clients])
                print("前几个预测/教师评分对比（round 0）：")
                print("pred:", pred[0, :100].cpu().numpy())
                print("true:", y[0, :100].cpu().numpy())

    avg_loss = total_loss / max(total_batches, 1)
    print(f"全量轮次的平均 MSE: {avg_loss:.6f}")


if __name__ == "__main__":
    evaluate(batch_size=1)
