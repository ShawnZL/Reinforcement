# from pyecharts.charts import Bar
# from pyecharts import options as opts
#
# # 您提供的数据
# months = ['2021年6月', '2021年7月', '2021年8月', '2021年9月', '2021年10月', '2021年11月', '2021年12月', '2022年1月', '2022年2月', '2022年3月', '2022年4月', '2022年5月', '2022年6月', '2022年7月', '2022年8月', '2022年9月', '2022年10月', '2022年11月', '2022年12月', '2023年1月', '2023年2月', '2023年3月', '2023年4月', '2023年5月', '2023年6月', '2023年7月', '2023年8月', '2023年9月', '2023年10月', '2023年11月', '2023年12月']
# pump1_runtime = [389.33, 40.18, 9.2, 23.33, 46.92, 0, 0.02, 2.85, 10.38, 0, 0, 0, 89.68, 160.62, 228.63, 161.75, 139.93, 150.25, 10.35, 0, 0, 0, 22.13, 162.03, 153.48, 162.08, 211.92, 144.07, 108.3, 45.47, 50.6]
# pump2_runtime = [449.63, 269.72, 228.93, 290.05, 189.05, 77.65, 10.6, 0, 0, 0, 0, 0, 143.98, 206.95, 172.37, 162.32, 189.32, 139.2, 44.62, 56.27, 0, 0, 39.3, 245.55, 280.97, 273.82, 235, 192.63, 139.85, 80.38, 48.88]
# pump3_runtime = [451.88, 255.7, 175.67, 129.87, 168.07, 14.83, 11.17, 16.87, 0, 0, 0, 0, 184.25, 230.4, 247.45, 126.45, 112.6, 112.63, 9.78, 19.8, 0, 0, 111.4, 25.18, 0, 0, 0, 0, 0, 0, 0]
# pump1_energy = [0, 2774.3, 536.92, 1870.58, 3902, 24.4, 25.1, 218.8, 732.1, 25.6, 25.7, 26.5, 7021.9, 13486.8, 16119.1, 8602.62, 9304.58, 11632, 537.2, 25.3, 23.3, 25.2, 25.7, 2044.2, 13381.38, 11934.92, 12142.5, 15900.5, 8906, 8014.4, 3033.1, 4982.6]
# pump2_energy = [13160.8, 19991.6, 19691.96, 20243.84, 12588.8, 4672.2, 786, 21.2, 19.5, 21.6, 21.6, 22.4, 11582, 18565.9, 13228.3, 9223.3, 14127, 11036.2, 3120.7, 3007.4, 23.9, 25.7, 21.6, 3399.1, 20350.7, 25217.9, 22028.5, 20955.1, 12208.4, 10917.5, 7229.4, 4791.2]
# pump3_energy = [0, 19894, 12725.87, 9661.03, 9732.5, 895.4, 715.6, 1205.4, 21.2, 29.2, 21.9, 23.5, 14803.1, 19977.2, 18251, 7732, 7251.2, 8520.3, 672.3, 1410, 29.2, 31.7, 9339.9, 1953.1, 7.7, 0, 0, 0, 0, 0, 0, 0]
#
# # bar1 = (
# #     Bar()
# #     .add_xaxis(months)
# #     .add_yaxis("冷冻水泵1运行时长(小时)", pump1_runtime, label_opts=opts.LabelOpts(is_show=False))
# #     .add_yaxis("冷冻水泵2运行时长(小时)", pump2_runtime, label_opts=opts.LabelOpts(is_show=False))
# #     .add_yaxis("冷冻水泵3运行时长(小时)", pump3_runtime, label_opts=opts.LabelOpts(is_show=False))
# #     .set_global_opts(title_opts=opts.TitleOpts(title="冷冻水泵运行时长统计"))
# # )
# # bar1.render("pump1_runtimes_energy.html")
# bar = (
#     Bar()
#     .add_xaxis(months)
#     .add_yaxis("冷冻水泵1电量(kW·h)", pump1_energy, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis("冷冻水泵2电量(kW·h)", pump2_energy, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis("冷冻水泵3电量(kW·h)", pump3_energy, label_opts=opts.LabelOpts(is_show=False))
#     .set_global_opts(title_opts=opts.TitleOpts(title="冷冻水泵运行电量统计"))
# )
#
# bar.render("test.html")

#
# from pyecharts.charts import Bar
# from pyecharts import options as opts
#
# # 您提供的数据
# months = ['2021年6月', '2021年7月', '2021年8月', '2021年9月', '2021年10月', '2021年11月', '2021年12月', '2022年1月', '2022年2月', '2022年3月', '2022年4月', '2022年5月', '2022年6月', '2022年7月', '2022年8月', '2022年9月', '2022年10月', '2022年11月', '2022年12月', '2023年1月', '2023年2月', '2023年3月', '2023年4月', '2023年5月', '2023年6月', '2023年7月', '2023年8月', '2023年9月', '2023年10月', '2023年11月', '2023年12月']
# pump1_runtime = [389.57, 3.88, 11.17, 16.32, 21.17, 16.37, 0, 0, 0, 0, 0, 0, 52.1, 109.58, 154.43, 31.97, 41.37, 57.65, 0, 0, 0, 27.02, 66.92, 95, 235.57, 224.52, 82.93, 95.43, 131.53, 0]
# pump3_runtime = [428.82, 195.57, 148.82, 44.35, 77.45, 11.15, 0, 0, 0, 0, 0, 0, 93.03, 187.58, 183.03, 109.3, 78.1, 69.07, 0, 0, 0, 64.3, 81.3, 216.83, 299.32, 232.3, 184.65, 103.62, 111.78, 0]
# pump5_runtime = [389.35, 58.25, 11.08, 16.25, 0, 0, 0, 0, 0, 0, 0, 0, 13.68, 7.47, 31.9, 40.52, 65.62, 20.82, 0, 0, 0, 14.27, 23.25, 60.6, 55.98, 44.37, 91.77, 41.05, 2.78, 0]
#
# bar = (
#     Bar()
#     .add_xaxis(months)
#     .add_yaxis("冷却水泵1运行时长(小时)", pump1_runtime, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis("冷却水泵2运行时长(小时)", pump3_runtime, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis("冷却水泵3运行时长(小时)", pump5_runtime, label_opts=opts.LabelOpts(is_show=False))
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="冷却水泵运行时长"),))
#
# bar.render("pump_runtime1.html")
# from pyecharts.charts import Bar
# from pyecharts import options as opts
#
# # 您提供的数据
# months = ['2021年6月', '2021年7月', '2021年8月', '2021年9月', '2021年10月', '2021年11月', '2021年12月', '2022年1月', '2022年2月', '2022年3月', '2022年4月', '2022年5月', '2022年6月', '2022年7月', '2022年8月', '2022年9月', '2022年10月', '2022年11月', '2022年12月', '2023年1月', '2023年2月', '2023年3月', '2023年4月', '2023年5月', '2023年6月', '2023年7月', '2023年8月', '2023年9月', '2023年10月', '2023年11月', '2023年12月']
# gen1_runtime = [0, 273.43, 277.45, 336.67, 282.07, 89.22, 122.73, 185.02, 299.02, 64.93, 0, 0, 0.68, 201.78, 338.97, 253.93, 219.65, 12.15, 344.73, 368.88, 336.3, 48.32, 3.07, 304.63, 359.17, 373.77, 358.88, 364.3, 283.45, 48.68, 267.28]
# gen2_runtime = [720, 72, 0.15, 2.08, 1.62, 24.8, 0.02, 55.82, 0, 0, 0, 0, 0, 0, 0, 192, 240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 0, 0, 0, 0]
#
# bar = (
#     Bar()
#     .add_xaxis(months)
#     .add_yaxis("燃气发电机1运行时长(小时)", gen1_runtime, label_opts=opts.LabelOpts(is_show=False))
#     .add_yaxis("燃气发电机2运行时长(小时)", gen2_runtime, label_opts=opts.LabelOpts(is_show=False))
#     .set_global_opts(
#         title_opts=opts.TitleOpts(title="燃气发电机运行时长"))
# )
#
# bar.render("generator_runtime.html")

from pyecharts.charts import Bar, Line
from pyecharts import options as opts

# 数据
dates = ['2023年12月21日', '2024年1月11日', '2024年1月11日']
baseline_load = [7.72, 6.95, 7.80]
response_load = [7.21, 6.51, 6.39]
response_rate = [0.946, 0.895, 1]

# 创建Bar图表
bar = (
    Bar()
    .add_xaxis(dates)
    .add_yaxis("基线负荷", baseline_load, label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("响应时段平均负荷", response_load,label_opts=opts.LabelOpts(is_show=False))
    .add_yaxis("实际调控比率", response_rate, label_opts=opts.LabelOpts(is_show=True))
    .set_global_opts(title_opts=opts.TitleOpts(title="需求响应事件对比"))
)

# 创建Line图表
line = (
    Line()
    .add_xaxis(dates)
    .add_yaxis("负荷响应率", response_rate, yaxis_index=1)
    .set_series_opts(label_opts=opts.LabelOpts(position="top", is_show=False))
)

# # 合并图表
# bar.overlap(line)

# 渲染图表
bar.render("demand.html")

