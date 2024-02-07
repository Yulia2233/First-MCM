from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/All_play_Data-用于预测.xlsx'
df = pd.read_excel(path)
df_x = df.drop(columns=['groups','player','time','Victor'])

df_sc = StandardScaler().fit_transform(df_x)
df_sc = pd.DataFrame(df_sc, columns=df_x.columns)

df_x = df_sc.drop(columns=['momentum'])


df_y = df_sc[['momentum']]



x_train = df_x
y_train =df_y
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

alpha_lasso = 10**np.linspace(-3,1,100)
lasso = Lasso()
coefs_lasso = []

for i in alpha_lasso:
    lasso.set_params(alpha = i)
    lasso.fit(X_train, y_train)
    coefs_lasso.append(lasso.coef_)
    
lasso = Lasso(alpha=10**(-3))
model_lasso = lasso.fit(X_train, y_train)
coef = pd.Series(model_lasso.coef_,index=X_train.columns)
print(coef[coef != 0].abs().sort_values(ascending = False))



