import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.font_manager import FontProperties
try:
    # 创建 FontProperties 对象，引入自定义字体
    custom_font_path = 'D:/font/Bookman Old Style.ttf'
    custom_font = FontProperties(fname=custom_font_path)
    plt.figure(figsize=(12,5))
except:
    print('导入失败!')
# Data provided
# metrics = ['MAE', 'MSE', 'RMSE']
# real_data = [0.073095291, 0.01277117, 0.113009602]
# noise_005 = [0.073040597, 0.012481659, 0.111721344]
# noise_01 = [0.073181035, 0.012597985, 0.112240744]

# # Number of groups
# n_groups = len(metrics)

# # Create the figure and the axes
# fig, ax = plt.subplots(figsize=(10, 6))

# # Index for the groups
# index = np.arange(n_groups)
# bar_width = 0.2

# # Opacity for the bars
# opacity = 0.8

# # Bar plots
# rects1 = ax.bar(index, real_data, bar_width,
#                 alpha=opacity, color=sns.color_palette('Blues',4)[3], label='Real Data')

# rects2 = ax.bar(index + bar_width, noise_005, bar_width,
#                 alpha=opacity, color=sns.color_palette('Oranges',4)[3], label='0.05 Noise')

# rects3 = ax.bar(index + bar_width*2, noise_01, bar_width,
#                 alpha=opacity, color=sns.color_palette('BuGn',4)[3], label='0.1 Noise')
# sns.set_style('dark')
# # Labeling
# ax.set_xlabel('Metrics',fontproperties=custom_font,fontsize=24)
# ax.set_ylabel('Values',fontproperties=custom_font,fontsize=24)
# ax.set_title('Comparison of Real Data and Noise Added Data',fontproperties=custom_font,fontsize=24,pad=10)
# ax.set_xticks(index + bar_width)
# ax.set_xticklabels(metrics)
# plt.legend( fontsize=24, prop={'size': 18, 'family': custom_font.get_name()})
# plt.tick_params(axis='both',which='major',labelsize=20)
# # Show the plot
# plt.tight_layout()
# plt.show()






import matplotlib.pyplot as plt

# Data provided
metrics = ['MAE', 'MSE', 'RMSE']
real_data = [0.073095291, 0.01277117, 0.113009602]
noise_005 = [0.073040597, 0.012481659, 0.111721344]
noise_01 = [0.073181035, 0.012597985, 0.112240744]

# Create the figure and the axes for the line plot
fig, ax = plt.subplots(figsize=(10, 6))

# Line plots
ax.plot(metrics, real_data, marker='o', label='Real Data')
ax.plot(metrics, noise_005, marker='s', label='0.05 Noise')
ax.plot(metrics, noise_01, marker='^', label='0.1 Noise')

# Labeling
ax.set_xlabel('Metrics')
ax.set_ylabel('Values')
ax.set_title('Comparison of Metrics for Real Data and Noise Added Data')
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
