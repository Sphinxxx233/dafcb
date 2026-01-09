import os
import numpy as np
import pandas as pd

# 你的保存目录（DA-FCB 的输出）
DATA_OUT_DIR = "../dataset/da_fcb_logs"

# 初始化计数器（假设有 100 个客户端）
NUM_CLIENTS = 100
selection_count = np.zeros(NUM_CLIENTS, dtype=int)

# 遍历目录中所有 CSV（按轮次保存）
for filename in os.listdir(DATA_OUT_DIR):
    if filename.endswith(".csv"):
        filepath = os.path.join(DATA_OUT_DIR, filename)
        df = pd.read_csv(filepath)

        # “selected” 列：1 表示被选中，0 表示未选
        selected = df["selected"].values
        client_ids = df["client_id"].values

        # 更新统计
        for cid, sel in zip(client_ids, selected):
            if sel == 1:
                selection_count[cid] += 1

# 输出：每个客户端被选次数
for cid, count in enumerate(selection_count):
    print(f"Client {cid}: selected {count} times")

# 保存为 CSV
out_df = pd.DataFrame({
    "client_id": np.arange(NUM_CLIENTS),
    "selection_count": selection_count
})
out_df.to_csv("client_selection_count.csv", index=False)

print("\n统计完成！结果已保存到 client_selection_count.csv")



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取统计结果
df = pd.read_csv("client_selection_count.csv")

client_ids = df["client_id"].values
counts = df["selection_count"].values

plt.figure(figsize=(14, 6))
plt.bar(client_ids, counts)

plt.xlabel("Client ID", fontsize=12)
plt.ylabel("Selection Count", fontsize=12)
plt.title("Client Selection Frequency", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.xticks(np.arange(0, max(client_ids)+1, 5))
plt.tight_layout()

# 保存图片
plt.savefig("client_selection_count_plot.png", dpi=300)
plt.show()

print("已生成图像：client_selection_count_plot.png")
