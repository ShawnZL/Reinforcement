import matplotlib.pyplot as plt
import json

# 读取数据
with open("1.json", 'r') as json_file:
    expanded_data = json.load(json_file)

with open("4.json", 'r') as json_file:
    MAPPO_data = json.load(json_file)

# 提取 x 轴数据
x_data = list(range(len(expanded_data)))

# 数据归一化函数
def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

# 提取 y 轴数据
NAMAPPO = expanded_data
MAPPO = MAPPO_data

# 对每个数据集进行归一化
NAMAPPO_normalized = normalize_data(NAMAPPO)
MAPPO_normalized = normalize_data(MAPPO)

# 创建两个子图
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 在第一个子图中绘制 NAMAPPO
axs[0].plot(x_data, NAMAPPO_normalized, label='NAMAPPO', color='blue')
axs[0].legend()
axs[0].set_ylabel('Normalized Returns')

# 在第二个子图中绘制 MAPPO
axs[1].plot(x_data, MAPPO_normalized, label='MAPPO', color='red')
axs[1].legend()
axs[1].set_xlabel('Episodes')
axs[1].set_ylabel('Normalized Returns')

# 调整布局，防止重叠
plt.tight_layout()

# 显示图形
plt.show()
