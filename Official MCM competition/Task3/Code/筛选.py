import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from minepy import MINE
import pandas as pd

# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/All_play_Data-用于预测.xlsx'
df = pd.read_excel(path)
df_x = df.drop(columns=['groups','player','time','momentum','Victor'])
df_y = df[['momentum']]


x_train = df_x
y_train =df_y

p = []
mine = MINE()
for feature in x_train.columns:
    mine.compute_score(x_train[feature].values.ravel(),y_train['momentum'].values.ravel())
    mic = mine.mic()
    p.append(mic)
    print(11)


df_p = pd.DataFrame({
    'feature':x_train.columns,
    'importances':p
})
df_p_sorted = df_p.sort_values(by='importances', ascending=False)
df_p_sorted.to_excel('D:/abmathdata/task3PLUS/结果/筛选1.xlsx',index=False)