import numpy as np
import json

def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def calculate_metrics(episode_rewards):
    # 收敛速度：计算前期回报的变化
    convergence_speed = np.diff(episode_rewards[:100]).mean()

    # 训练稳定性：计算回报的标准差
    training_stability = np.std(episode_rewards)

    episode_rewards = normalize_data(episode_rewards)
    # 计算平均回报
    average_return = np.mean(episode_rewards)

    # 计算滚动平均回报
    rolling_average = np.convolve(episode_rewards, np.ones(100) / 100, mode='valid')

    # 计算标准差
    std_deviation = np.std(episode_rewards)

    return average_return, rolling_average, std_deviation, convergence_speed, training_stability

# 存储每个实验的指标结果
metrics_results = {}

# 实验名称与对应的json文件
experiments = {
    "NAMAPPO": "1.json",
    "MADDPG": "2.json",
    "IPPO": "3.json",
    "MAPPO": "4.json",
    "GRIDRL": "5.json"
}

# 读取每个实验的回报数据并计算指标
for experiment_name, json_file in experiments.items():
    with open(json_file, 'r') as json_file:
        episode_rewards = json.load(json_file)

    average_return, rolling_average, std_deviation, test1, test2  = calculate_metrics(episode_rewards)

    metrics_results[experiment_name] = {
        "Average Return": average_return,
        "Rolling Average": rolling_average,
        "Standard Deviation": std_deviation,
        "test1": test1,
        "test2": test2,
    }

# 打印指标结果
for experiment_name, metrics in metrics_results.items():
    print(f"{experiment_name} Metrics:")
    print(f"Average Return: {metrics['Average Return']}")
    print(f"Rolling Average Return: {metrics['Rolling Average']}")
    print(f"Standard Deviation: {metrics['Standard Deviation']}")
    print(f"convergence_speed: {metrics['test1']}")
    print(f"training_stability: {metrics['test2']}")
    print("\n")

# 存储每个算法的回报数据
episode_rewards = {}

for experiment_name, json_file in experiments.items():
    with open(json_file, 'r') as json_file:
        episode_rewards[experiment_name] = json.load(json_file)

# 计算每个算法的样本效率
final_performance = {}
number_of_training_samples = {}

for experiment_name, rewards in episode_rewards.items():
    final_performance[experiment_name] = np.mean(rewards[-100:])  # 使用最后100个回合的平均回报作为最终性能
    number_of_training_samples[experiment_name] = len(rewards)

# 计算样本效率
sample_efficiency = {}

for experiment_name in experiments:
    sample_efficiency[experiment_name] = final_performance[experiment_name] / number_of_training_samples[experiment_name]

# 打印每个算法的样本效率
for experiment_name, efficiency in sample_efficiency.items():
    print(f"{experiment_name} Sample Efficiency: {efficiency}")
