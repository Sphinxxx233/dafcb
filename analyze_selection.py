import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# ================= 配置区域 =================
# 请确保这两个路径是正确的文件夹路径
BASELINE_DIR = "dataset/fsmnist"          # 基准策略数据 (如 Random)
AI_STRATEGY_DIR = "dataset/fsmnist_valuepred"  # 你的 AI 策略数据

# ================= 1. 数据加载与打分系统 =================

def load_all_rounds(folder_path, tag="experiment"):
    """读取文件夹下所有 CSV，并打上实验标签"""
    all_files = glob.glob(os.path.join(folder_path, "round_*_metrics.csv"))
    if not all_files:
        print(f"警告: 在 {folder_path} 未找到 CSV 文件")
        return pd.DataFrame()
    
    print(f"[{tag}] 正在加载 {len(all_files)} 个文件...")
    df_list = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df['experiment'] = tag
            # 只有 selected=1 的行才有真实的 contribution 数据
            df_selected = df[df['selected'] == 1].copy()
            df_list.append(df_selected)
        except Exception as e:
            pass
    
    if not df_list:
        return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def calculate_client_quality_scores(df_combined):
    """
    打分系统核心：
    计算每个客户端的 '平均真实贡献' 作为其质量分。
    """
    # 按 client_id 分组，计算 contribution 的均值
    # 你也可以用 'sum' (总贡献) 或 'median' (中位数)
    client_stats = df_combined.groupby('client_id')['contribution'].agg(['mean', 'count', 'std'])
    
    # 重命名列
    client_stats.columns = ['quality_score', 'total_selected', 'stability']
    
    # 排序：分数高的在前面
    client_stats = client_stats.sort_values(by='quality_score', ascending=False)
    
    return client_stats

# ================= 主程序 =================

# 1. 加载两个实验的数据
df_baseline = load_all_rounds(BASELINE_DIR, tag="Baseline")
df_ai = load_all_rounds(AI_STRATEGY_DIR, tag="AI_Strategy")

if df_baseline.empty or df_ai.empty:
    print("错误: 数据加载失败，请检查路径。")
    exit()

# 2. 【上帝视角打分】合并所有数据来评估客户端的真实质量
# 为什么要合并？因为样本越多，对客户端质量的评估越准确
df_all = pd.concat([df_baseline, df_ai], ignore_index=True)
client_scores = calculate_client_quality_scores(df_all)

# 定义 "优质客户端" (Top 20) 和 "劣质客户端" (Bottom 20)
top_20_clients = client_scores.head(20).index.tolist()
bottom_20_clients = client_scores.tail(20).index.tolist()

print("\n=== 🌟 客户端质量打分完成 (Top 5) ===")
print(client_scores[['quality_score', 'total_selected']].head())

# 3. 【对比统计】计算每个客户端在不同策略下的被选中次数
# 注意：这里我们要回过头去统计原始数据里的 selected=1 次数
def count_selections(folder_path):
    # 这里我们要读取所有行(包括selected=0)来统计轮次，或者直接统计selected=1的行数
    # 为了简单，直接用我们刚才加载的 selected=1 的 DataFrame
    if folder_path == BASELINE_DIR:
        return df_baseline['client_id'].value_counts()
    else:
        return df_ai['client_id'].value_counts()

counts_baseline = df_baseline['client_id'].value_counts()
counts_ai = df_ai['client_id'].value_counts()

# 合并成一张对比表
comparison = pd.DataFrame({
    'Quality_Score': client_scores['quality_score'],
    'Baseline_Count': counts_baseline,
    'AI_Count': counts_ai
}).fillna(0) # 没被选中的补0

# 4. 【核心指标】优质客户端选中率对比
avg_sel_top_base = comparison.loc[top_20_clients, 'Baseline_Count'].mean()
avg_sel_top_ai = comparison.loc[top_20_clients, 'AI_Count'].mean()

avg_sel_bot_base = comparison.loc[bottom_20_clients, 'Baseline_Count'].mean()
avg_sel_bot_ai = comparison.loc[bottom_20_clients, 'AI_Count'].mean()

print(f"\n=== ⚔️ 终极对决结果 ===")
print(f"【优质客户端 (Top 20)】平均被选中次数:")
print(f"  - Baseline (Random): {avg_sel_top_base:.2f} 次")
print(f"  - AI Strategy      : {avg_sel_top_ai:.2f} 次  <-- 期望这里更高")

print(f"\n【劣质客户端 (Bottom 20)】平均被选中次数:")
print(f"  - Baseline (Random): {avg_sel_bot_base:.2f} 次")
print(f"  - AI Strategy      : {avg_sel_bot_ai:.2f} 次  <-- 期望这里更低")

# ================= 5. 可视化绘图 =================

plt.figure(figsize=(12, 6))

# 图1: 散点图 - 质量 vs 选中次数
plt.subplot(1, 2, 1)
plt.scatter(comparison['Quality_Score'], comparison['Baseline_Count'], 
            alpha=0.6, label='Baseline', color='gray')
plt.scatter(comparison['Quality_Score'], comparison['AI_Count'], 
            alpha=0.7, label='AI Strategy', color='red')
plt.xlabel('Client True Quality (Avg Contribution)')
plt.ylabel('Selected Count')
plt.title('Selection Frequency vs. Client Quality')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 图2: Top-10 优质客户端的对比柱状图
plt.subplot(1, 2, 2)
top_10_ids = client_scores.head(10).index
x = np.arange(10)
width = 0.35

plt.bar(x - width/2, comparison.loc[top_10_ids, 'Baseline_Count'], width, label='Baseline', color='gray')
plt.bar(x + width/2, comparison.loc[top_10_ids, 'AI_Count'], width, label='AI Strategy', color='#ff4757')

plt.xlabel('Client ID (Ranked by Quality)')
plt.ylabel('Selected Count')
plt.title('Selection Count of Top-10 Best Clients')
plt.xticks(x, top_10_ids)
plt.legend()

plt.tight_layout()
plt.savefig("selection_quality_analysis.png")
print("\n图表已保存为 selection_quality_analysis.png")
plt.show()