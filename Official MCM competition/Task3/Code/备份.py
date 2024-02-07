import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/All_play_Data-用于预测.xlsx'
df = pd.read_excel(path)
df_all = df.drop(columns=['groups','player','time','Victor','V8','V10','V11','V14'])
df_x = df_all.drop(columns=['momentum'])
df_y = df_all['momentum']
df_x.insert(0,'momentum',df_y)
df = df_x
# 标准化
scaler = MinMaxScaler(feature_range=(0,1))
numeric_columns = df.columns
df_scale = scaler.fit_transform(df,df.columns)




n_past = 3  # 记录次数
p = 0.2     # 比例
featurenum =  7    # 特征数量


np.set_printoptions(suppress=True, precision=8)



# 拆分训练集和测试集
#测试集比例
test_split = round(len(df)*p)

df_to_train = df[:-test_split]

df_to_test = df[-test_split:]



scaler = MinMaxScaler(feature_range=(0,1))
numeric_columns = df_to_train.columns
df_to_train_scale = scaler.fit_transform(df_to_train[numeric_columns],df_to_train.columns)
df_to_test_scale = scaler.fit_transform(df_to_test[numeric_columns],df_to_train.columns)
print(df_to_train_scale)





# 拆分数据的X和Y
def createXY(dataset,n_past = 30):
    datax = []
    datay = []
    for i in range(n_past,len(dataset)):
        datax.append(dataset[i-n_past:i,1:])
        datay.append(dataset[i,0])
    return np.array(datax),np.array(datay)


trainX,trainY = createXY(df_to_train_scale,n_past)
testX,testY = createXY(df_to_test_scale,n_past)


# 建立模型
def build_model(optimizer):
    gridModel = Sequential()
    gridModel.add(LSTM(200,return_sequences=True,input_shape=(n_past,featurenum)))
    gridModel.add(LSTM(200))
    gridModel.add(Dropout(0.2))
    gridModel.add(Dense(1))
    gridModel.compile(loss='mse',optimizer=optimizer)
    return gridModel


# gridModel = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
# parameters = {
#     'batch_size':[16],
#     'epochs':[50],
#     'optimizer':['adam']
# }
# grid_search = GridSearchCV(estimator=gridModel,param_grid=parameters,cv=2)

# 修改部分
my_model = build_model('adam')
my_model.fit(trainX,trainY,batch_size=32,epochs=100)

# 上面是修改部分


# grid_search = grid_search.fit(trainX,trainY)
# print(grid_search.best_params_)
# my_model = grid_search.best_estimator_

predict = my_model.predict(testX)

predict_cop = np.repeat(predict,df.shape[1],axis=-1)
pre = scaler.inverse_transform(np.reshape(predict_cop,(len(predict),df.shape[1])))[:,0]
testY_cop = np.repeat(testY,df.shape[1],axis=-1)
tY = scaler.inverse_transform(np.reshape(testY_cop,(len(predict),df.shape[1])))[:,0]

mae_lgb = mean_absolute_error(tY, pre)
mse_lgb = mean_squared_error(tY, pre)
rmse_lgb = np.sqrt(mse_lgb)
r2_lgb = r2_score(tY, pre)

# 打印评估结果
print("MAE:", mae_lgb)
print("MSE:", mse_lgb)
print("RMSE:", rmse_lgb)
print("R^2:", r2_lgb)