import torch
import torch.nn as nn


class CNNScoringNet(nn.Module):
    """
    基于 1D CNN 的客户端选择评分网络
    输入:  (B, 100, 11)
    输出:  (B, 100)
    """

    def __init__(self, in_features=11, hidden_channels=32):
        super().__init__()

        # 1D 卷积输入需要变成 (B, C_in, L)
        # C_in = in_features (11), L = 100
        # 所以 forward 里要 x.permute(0, 2, 1)

        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(in_channels=hidden_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # 最终映射到每个client一个分数
        self.to_score = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        """
        输入 x: (B, 100, 11)
        """

        # 转换到 CNN 输入格式：(B, C_in=11, L=100)
        x = x.permute(0, 2, 1)

        # 卷积
        x = self.conv_layers(x)  # (B, 16, 100)

        # 得到评分：(B, 1, 100)
        x = self.to_score(x)

        # 去掉通道维度 (B, 100)
        x = x.squeeze(1)

        return x
