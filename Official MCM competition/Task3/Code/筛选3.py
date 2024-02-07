from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import xgboost as xgb
import pandas as pd

# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/All_play_Data-用于预测.xlsx'
df = pd.read_excel(path)
df_x = df.drop(columns=['groups','player','time','momentum','Victor'])
df_y = df[['momentum']]


x_train = df_x
y_train =df_y

model = xgb.XGBRegressor(objective='reg:squarederror',n_estimators=200,max_depth=20,min_child_weight=2,learning_rate=0.1)
model.fit(x_train, y_train)
feature_importances = model.feature_importances_
features = x_train.columns


df_p = pd.DataFrame({
        'feature':features,
        'importances':feature_importances
})
df_p_sorted = df_p.sort_values(by='importances', ascending=False)

df_p_sorted.to_excel('D:/abmathdata/task3PLUS/结果/筛选3.xlsx')