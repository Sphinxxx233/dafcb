import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader_selector import build_dataloader
from scoring_cnn import CNNScoringNet


# ========================
# Ranking Loss（可选）
# ========================
def ranking_loss(pred_scores):
    top10 = torch.topk(pred_scores, k=10, dim=1).values.mean(dim=1)
    tail90 = torch.topk(pred_scores, k=90, dim=1, largest=False).values.mean(dim=1)
    margin = 0.1
    loss = torch.clamp(margin - (top10 - tail90), min=0.0)
    return loss.mean()


# ========================
# 训练配置
# ========================
EPOCHS = 50
LR = 1e-3
SAVE_PATH = "saved_models/scoring_cnn.pth"
os.makedirs("saved_models", exist_ok=True)


def train():
    # 判断 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("⚠ 检测到 GPU 不被当前 PyTorch 支持，将使用 CPU 训练（安全模式）。")
    else:
        print("使用 GPU:", torch.cuda.get_device_name(0))

    # -------------------------------
    # 数据加载器
    # -------------------------------
    dataloader = build_dataloader(batch_size=1, shuffle=True)

    # -------------------------------
    # 模型、loss、优化器
    # -------------------------------
    model = CNNScoringNet(in_features=11).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -------------------------------
    # 训练循环
    # -------------------------------
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            loss_mse = criterion(pred, y)
            loss_rank = ranking_loss(pred) * 0.1

            loss = loss_mse + loss_rank

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {epoch_loss:.4f}")

    # -------------------------------
    # 训练完保存模型
    # -------------------------------
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n模型已保存到: {SAVE_PATH}")


if __name__ == "__main__":
    train()
