# 要在 Matplotlib 中画带有标准差阴影的曲线图

import numpy as np
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.pyplot as plt
# 读取数据
with open("1.json", 'r') as json_file:
    expanded_data = json.load(json_file)

with open("2.json", 'r') as json_file:
    new1_data = json.load(json_file)

with open("3.json", 'r') as json_file:
    IPPO_data = json.load(json_file)

with open("4.json", 'r') as json_file:
    MAPPO_data = json.load(json_file)

with open("5.json", 'r') as json_file:
    NVPPO_data = json.load(json_file)

# 提取 x 轴数据
# x_data = list(range(len(expanded_data)))
# 数据归一化函数
def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

# 提取 y 轴数据
MADDPG = np.array(new1_data)
NAMAPPO = np.array(expanded_data)
IPPO = np.array(IPPO_data)
MAPPO = np.array(MAPPO_data)
LCCPSRL = np.array(NVPPO_data)

# 对每个数据集进行归一化
MADDPG_normalized = normalize_data(MADDPG)
NAMAPPO_normalized = normalize_data(NAMAPPO)
IPPO_normalized = normalize_data(IPPO)
MAPPO_normalized = normalize_data(MAPPO)
LCCPSRL_normalized = normalize_data(LCCPSRL)

x_data = np.arange(len(MADDPG_normalized))  # 使用 MADDPG_normalized 的长度作为 x_data

# 创建 Matplotlib 折线图
plt.figure(figsize=(10, 6))

# 绘制曲线图
plt.plot(x_data, MADDPG_normalized, label='MADDPG')

# 计算标准差
std_MADDPG = np.std(MADDPG_normalized)

# 使用 fill_between 添加标准差阴影
plt.fill_between(x_data, MADDPG_normalized - std_MADDPG, MADDPG_normalized + std_MADDPG, alpha=0.2, label='Standard Deviation')

# 添加标签和图例
plt.xlabel('Timestep')
plt.ylabel('Normalized Reward')
plt.title('Normalized Reward with Standard Deviation Shadow')
plt.legend()

# 显示图形
plt.show()

