from pyecharts.charts import Line
from pyecharts import options as opts
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
x_data = list(range(3000))
# 数据归一化函数
def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

# 提取 y 轴数据
MADDPG = new1_data
NAMAPPO = expanded_data
IPPO = IPPO_data
MAPPO = MAPPO_data
LCCPSRL = NVPPO_data

# 对每个数据集进行归一化
MADDPG_normalized = normalize_data(MADDPG)
NAMAPPO_normalized = normalize_data(NAMAPPO)
IPPO_normalized = normalize_data(IPPO)
MAPPO_normalized = normalize_data(MAPPO)
LCCPSRL_normalized = normalize_data(LCCPSRL)



# 创建折线图
line = (
    Line()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(series_name="MADDPG", y_axis=MADDPG_normalized, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis(series_name="NAMAPPO", y_axis=NAMAPPO_normalized, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis(series_name="IPPO", y_axis=IPPO_normalized, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis(series_name="MAPPO", y_axis=MAPPO_normalized, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis(series_name="LCCPSRL", y_axis=LCCPSRL_normalized, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(title="Comparison of RL"),
        xaxis_opts=opts.AxisOpts(
            min_=0,
            max_=3000,
            axislabel_opts=opts.LabelOpts(formatter="{value}", interval=199)
        )
    )
)

# line = (
#     Line()
#     .add_xaxis(xaxis_data=x_data)
#     .add_yaxis(series_name="MADDPG", y_axis=MADDPG_normalized, label_opts=opts.LabelOpts(is_show=False))
#     # .add_yaxis(series_name="NAMAPPO", y_axis=NAMAPPO_normalized, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis(series_name="IPPO", y_axis=IPPO_normalized, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis(series_name="GRIDRL", y_axis=MAPPO_normalized, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis(series_name="NAMAPPO", y_axis=GRIDRL_normalized, label_opts=opts.LabelOpts(is_show=False))
#     .set_global_opts(title_opts=opts.TitleOpts(title="Comparison of RL"))
# )

# # 添加标准差阴影
# line = line.line_markarea(
#     series_index=0,
#     x_axis=x_data,
#     y_axis=MADDPG_normalized,  # 这里假设 MADDPG_normalized 代表均值
#     y_axis2=np.array(MADDPG_normalized) + np.random.normal(0, 0.1, len(MADDPG_normalized)),  # 模拟标准差
#     itemstyle_opts=opts.ItemStyleOpts(color="rgba(0, 0, 0, 0.1)"),
# )

# 保存图表
line.render("temp.html")


