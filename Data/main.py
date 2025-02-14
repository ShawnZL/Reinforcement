import json
import random

# 读取JSON文件
file_path = "2.json"  # 请替换为你的JSON文件路径
with open(file_path, 'r') as json_file:
    data = json.load(json_file)

# 处理前1000个数据
processed_data = []

for i in range(4, 1600):
    sample = data[i]
    if isinstance(sample, float):
        # 对浮点数进行随机减去（5，10）之间的值
        random_offset = random.uniform(-5, 5)
        sample -= random_offset
    elif isinstance(sample, dict):
        # 对字典中的浮点数进行随机减去（5，10）之间的值
        for key, value in sample.items():
            if isinstance(value, float):
                random_offset = random.uniform(-5, 5)
                sample[key] -= random_offset

    processed_data.append(sample)

# 将处理后的数据保存回JSON文件
with open(file_path, 'w') as json_file:
    json.dump(processed_data, json_file, indent=2)

# 输出处理后的数据
print(processed_data)
