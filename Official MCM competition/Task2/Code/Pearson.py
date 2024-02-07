import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import ScalarFormatter




data = pd.read_excel('D:/abmathdata/task2/All_play_for1.xlsx')

data = data[['momentum','Victor','p_streaks']]


pearson_correlation = data.corr(method='spearman')


def ShowGRAHeatMap(data):
    # 色彩集
    colormap = plt.cm.BuPu
    plt.figure(figsize=(18,16))
    
    ax = sns.heatmap(data.astype(float),linewidths=0.1,vmax=1.0,square=True,\
               cmap=colormap,linecolor='white',annot=True)
    # 改变刻度字体大小
    ax.tick_params(axis='x', labelsize=18)  # 改变x轴刻度字体大小
    ax.tick_params(axis='y', labelsize=18)  # 改变y轴刻度字体大小

    plt.show()
    
ShowGRAHeatMap(pearson_correlation)