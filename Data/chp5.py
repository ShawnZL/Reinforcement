# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取数据
# data = pd.read_csv('test11.csv')
#
# # 提取时间点作为 x 轴数据
# time_points = data['0']
#
# # 提取特征数据作为 y 轴数据
# features = data.drop(columns=['0'])
# # features2 = data.drop(columns=['2'])
# # features3 = data.drop(columns=['3'])
# # features4 = data.drop(columns=['4'])
#
# # 绘制折线图
# plt.figure(figsize=(10, 6))  # 设置画布大小
# for feature in features.columns:
#     plt.plot(time_points, features[feature], label=feature)
#
# # 添加图例、标题和轴标签
# plt.legend()
# plt.title('Yearly Load and Consumption')
# plt.xlabel('Time')
# plt.ylabel('Value')
#
# # 设置 x 轴刻度
# plt.xticks(rotation=45)
#
# # 显示图形
# plt.tight_layout()  # 调整布局以防重叠
# plt.show()

from pyecharts.charts import Line
from pyecharts import options as opts
import pandas as pd

# 读取数据
data = pd.read_csv('test11.csv')

# 提取时间点作为 x 轴数据
time_points = data['0']

# 提取特征数据作为 y 轴数据
features = data.drop(columns=['0'])

# 创建 Line 实例
line = Line()

# 添加 x 轴数据
line.add_xaxis(time_points.tolist())

# 添加 y 轴数据
for column in features.columns:
    line.add_yaxis(column, features[column].tolist(), label_opts=None)

# 设置全局配置项
line.set_global_opts(
    title_opts=opts.TitleOpts(title="1月1日运行结果对比"),
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-45, interval=3, font_size=20)),  # 设置 x 轴标签旋转角度
    yaxis_opts=opts.AxisOpts(name="单位：MW",
                             axislabel_opts=opts.LabelOpts(font_size=20),  # 调整y轴标签的字体大小
                             ),  # 设置 y 轴名称
    legend_opts=opts.LegendOpts(orient="vertical",
                                pos_right="right",
                                ),  # 设置图例位置和方向
)

# 渲染图表
line.render("line_chart.html")
