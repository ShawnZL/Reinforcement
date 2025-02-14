import json
import numpy as np
import random
import matplotlib.pyplot as plt
# 原始数据
with open('noisy_data3.json', 'r') as file:
    original_data = json.load(file)

# 计算相邻两个数据点之间的梯度
gradients = np.gradient(original_data)

# 存储最终结果
noisy_data = []

# 控制前1000个数据的震荡
for i in range(1000):
    start_point = original_data[i]
    end_point = original_data[i + 1]
    gradient = gradients[i]

    # 在梯度上进行插值并加入噪声，前1000个数据震荡较大
    interpolated_values = np.linspace(start_point, end_point, 1)
    interpolated_values += random.uniform(-2 * gradient, 2 * gradient)
    noisy_data.append(interpolated_values)

# 控制后边的数据震荡减小
for i in range(1000, len(original_data) - 1):
    start_point = original_data[i]
    end_point = original_data[i + 1]
    gradient = gradients[i]

    # 在梯度上进行插值并加入噪声，后边的数据震荡减小
    interpolated_values = np.linspace(start_point, end_point, 1)
    interpolated_values += random.uniform(-0.5 * gradient, 0.5 * gradient)
    noisy_data.append(interpolated_values)

# 将数据保存到字典
data_to_save = {
    'noisy_data': noisy_data,
}

# 指定保存的文件路径
file_path = 'noisy_data4.json'

# 将数据保存到JSON文件
with open(file_path, 'w') as json_file:
    json.dump(data_to_save, json_file)

print(f'Data saved to {file_path}')

# 从 JSON 文件读取数据
with open('noisy_data4.json', 'r') as file:
    json_data = json.load(file)

# 获取噪声数据
noisy_data = json_data['noisy_data']

# 绘制折线图
plt.plot(noisy_data, label='Noisy Data')
plt.xlabel('Data Points')
plt.ylabel('Values')
plt.title('Noisy Data Plot')
plt.legend()
plt.show()