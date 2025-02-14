from pyecharts.charts import Line
from pyecharts import options as opts
import json
import numpy as np
import matplotlib.pyplot as plt
# 读取数据
# with open("chp3MADDPG.json", 'r') as json_file:
#     MADDPG = json.load(json_file)
#
# with open("chp3Ippo.json", 'r') as json_file:
#     IPPO = json.load(json_file)
#
# with open("chp3MAPPO.json", 'r') as json_file:
#     MAPPO = json.load(json_file)
#
with open("chp3best.json", 'r') as json_file:
    best = json.load(json_file)

with open("chp3second.json", 'r') as json_file:
    MADDPG = json.load(json_file)

x_data = list(range(len(MADDPG)))


# 创建折线图
line = (
    Line()
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(series_name="best", y_axis=best, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis(series_name="MADDPG", y_axis=MADDPG, label_opts=opts.LabelOpts(is_show=False))
    # .add_yaxis(series_name="IPPO", y_axis=IPPO, label_opts=opts.LabelOpts(is_show=False))
    # .add_yaxis(series_name="MAPPO", y_axis=MAPPO, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(title_opts=opts.TitleOpts(title="Comparison of RL"))
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

# 保存图表
line.render("temp.html")


