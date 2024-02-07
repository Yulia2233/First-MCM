import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/All_play_Data-用于预测.xlsx'
df = pd.read_excel(path)
df_all = df.drop(columns=['groups','player','time','Victor','V14','V11','V8','V10'])
df_x = df_all.drop(columns=['momentum'])
df_y = df_all[['momentum']]


# 分离测试集和训练集
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.3,random_state=42)


x_train1 = df_x.iloc[43:1188,:]
x_test1 = df_x.iloc[:43,:]
y_train1 = df_y.iloc[43:1188,:]
y_test1 = df_y.iloc[:43,:]

param_grid = {
    'n_estimators':range(100,701,100),  # 尝试不同的邻居数
    'learning_rate':[0.5,0.1,0.01]    # 深度
    }

model = AdaBoostRegressor()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(x_train1,y_train1)
model = grid_search.best_estimator_
pre1 = model.predict(x_test1)



x_train2 = df_x.iloc[1188+43:,:]
x_test2 = df_x.iloc[1188:43+1188,:]
y_train2 = df_y.iloc[1188+43:,:]
y_test2 = df_y.iloc[1188:43+1188,:]


model = AdaBoostRegressor()
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(x_train2,y_train2)
model = grid_search.best_estimator_
pre2 = model.predict(x_test2)

y2 = (y_test1.values - y_test2.values).ravel()
y = pre1 - pre2

# Y1 = np.concatenate([y1,y2])
# Y2 = np.concatenate([y1,y])
import math
mae = mean_absolute_error(y2,y)
mse = mean_squared_error(y2,y)
rmse = math.sqrt(mse)
r2 = r2_score(y2,y)

print(mae)
print(mse)
print(rmse)
print(r2)
import matplotlib.pyplot as plt
df = pd.DataFrame({
    'T':y2,
    'P':y
})
df.to_excel('D:/abmathdata/task3PLUS/结果/ADA_.xlsx',index=False)
plt.plot(range(43),y,color='g')
plt.plot(range(43),y2,color='b')
# plt.xticks(range(1,1200,100))
plt.xticks(range(0,43,3))
plt.show()
