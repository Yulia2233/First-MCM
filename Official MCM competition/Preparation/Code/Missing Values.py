'''
template writing by Yu Liu

'''
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd 


path = 'D:/abmathdata/task4/前提/美国网球公开赛.xlsx'
df = pd.read_excel(path)
df1 = df.drop(columns=['match_id','player1','player2','elapsed_time'])
X = df1.values
# 初始化IterativeImputer
imp = IterativeImputer(estimator=RandomForestRegressor(), max_iter=4, random_state=0)

# 拟合模型并转换数据集以填充缺失值
X_imputed = imp.fit_transform(X)
path = 'D:/abmathdata/task4/前提/美国网球公开赛.xlsx'
df2 = pd.DataFrame(X_imputed, columns=df1.columns)
df2.insert(0,'elapsed_time',df['elapsed_time'].values)
df2.insert(0,'player2',df['player2'].values)
df2.insert(0,'player1',df['player1'].values)
df2.insert(0,'match_id',df['match_id'].values)

df2.to_excel(path,index=False)