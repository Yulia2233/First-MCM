import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVR,SVR

# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/All_play_Data-用于预测.xlsx'
df = pd.read_excel(path)
df_all = df.drop(columns=['groups','player','time','Victor','V14','V11','V8','V10'])
df_x = df_all.drop(columns=['momentum'])
df_y = df_all[['momentum']]




x_train1 = df_x.iloc[43:1188,:]
x_test1 = df_x.iloc[:43,:]
y_train1 = df_y.iloc[43:1188,:]
y_test1 = df_y.iloc[:43,:]

model = LinearSVR(epsilon=0.0, C=1.0, max_iter=10000, random_state=42)
model.fit(x_train1,y_train1)


pre1 = model.predict(x_test1)



x_train2 = df_x.iloc[1188+43:,:]
x_test2 = df_x.iloc[1188:43+1188,:]
y_train2 = df_y.iloc[1188+43:,:]
y_test2 = df_y.iloc[1188:43+1188,:]

model = LinearSVR(epsilon=0.0, C=1.0, max_iter=10000, random_state=42)
model.fit(x_train2,y_train2)


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
df.to_excel('D:/abmathdata/task3PLUS/结果/LSVM_.xlsx',index=False)
plt.plot(range(43),y,color='g')
plt.plot(range(43),y2,color='b')
# plt.xticks(range(1,1200,100))
plt.xticks(range(0,43,3))
plt.show()


