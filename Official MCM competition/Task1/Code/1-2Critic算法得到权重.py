'''
template writing by Yu Liu

'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib  # 用于保存模型
#按指定路径导入数据，以“地区”为索引（文件路径需按实际情况更换）
data = pd.read_excel('D:/111111/momentum111.xlsx',sheet_name='Sheet1')
data_m =  data.drop(columns=['player','time','groups','momentum'])

# #定义正向指标min-max标准化函数
# def minmax_p(x): 
#     return (x - x.min()) / (x.max() - x.min())

# #定义负向指标min-max标准化函数
# def minmax_n(x): 
#     return (x.max() - x) / (x.max() - x.min())

# #使用正向指标min-max标准化函数标准化数据
# data_m = minmax_p(data_m)

# vi = data_m.std()/ data_m.mean()

# ri = data_m.corr().abs()
# fi = (1 - ri).sum()

# pi = vi * fi
# # 创建MinMaxScaler实例
scaler = MinMaxScaler(feature_range=(0, 1))

# #归一化评价指标的综合信息载荷量
# w = pi / pi.sum()
w = [0.084594,0.044375,0.176083,0.102338,0.592609]
data['momentum'] = data_m.dot(w)
data['momentum'] = scaler.fit_transform(data[['momentum']])
data.to_excel('D:/111111/momentum1.xlsx',index=False)

# 保存拟合好的缩放器
# joblib.dump(scaler, 'scaler.save')


#data = pd.read_excel('D:/abmathdata/task1PLUS/结果/All_play_Data.xlsx')

# player1 = data.iloc[:1188,:]
# player2 = data.iloc[1188:,:]
# print(player1)
# print(player2)

# with pd.ExcelWriter('D:/abmathdata/task1PLUS/结果/Player.xlsx') as writer:
#     player1.to_excel(writer, sheet_name='player1', index=False)
#     player2.to_excel(writer, sheet_name='player2', index=False)
