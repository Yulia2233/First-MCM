import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.python.keras.layers import GRU, Dense,Dropout
np.set_printoptions(suppress=True, precision=8)

n_past = 3  # 记录次数
p = 0.8     # 比例
featurenum =  7    # 特征数量
# 拆分数据的X和Y
def createXY(dataset,n_past = 30):
    datax = []
    datay = []
    for i in range(n_past,len(dataset)):
        datax.append(dataset[i-n_past:i,1:])
        datay.append(dataset[i,0])
    return datax,datay


# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/每场比赛数据(2).xlsx'


# 建立模型
def build_model(optimizer):
    gridModel = Sequential()
    gridModel.add(LSTM(500,return_sequences=True,input_shape=(n_past,featurenum)))
    gridModel.add(LSTM(500))
    gridModel.add(Dropout(0.2))
    gridModel.add(Dense(1))
    gridModel.compile(loss='mse',optimizer=optimizer)
    return gridModel
'''
def build_model(optimizer):
    model = Sequential()
    model.add(GRU(500, return_sequences=False ,input_shape=(n_past,7)))
    model.add(Dense(units=500))
    model.add(Dense(units=500, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model
'''
scaler = MinMaxScaler(feature_range=(0, 1))

def Page(pagestart,pageend):
    DataX,DataY = [],[]
    for i in range(pagestart,pageend+1):
        df = pd.read_excel(path,sheet_name=f'Sheet{i}')
        df_all = df.drop(columns=['groups','player','time','Victor','V8','V10','V11','V14'])
        df_x = df_all.drop(columns=['momentum'])
        df_y = df_all['momentum']
        df_x.insert(0,'momentum',df_y)
        df = df_x
        DX,DY = createXY(df.values,n_past)
        DataX.extend(DX)
        DataY.extend(DY)
    
    print(len(DataY))

    # 标准化数据

    scaled_DataX = np.array(DataX).reshape(-1, featurenum)
    scaled_DataX = scaler.fit_transform(scaled_DataX)
    scaled_DataX = scaled_DataX.reshape(-1, n_past, featurenum)

    DataY = np.array(DataY).reshape(-1, 1)
    DataY = scaler.fit_transform(DataY)
    DataY = DataY.ravel()
    return scaled_DataX,DataY
    

trainX1, trainY1 = Page(2,31)
testX1, testY1 = Page(1,1)

my_model = build_model('adam')
my_model.fit(trainX1,trainY1,batch_size=32,epochs=100)
pre1 = my_model.predict(testX1)


trainX2,trainY2 = Page(33,62)
testX2, testY2 = Page(32,32)

my_model = build_model('adam')
my_model.fit(trainX2,trainY2,batch_size=32,epochs=100)
pre2 = my_model.predict(testX2)



pre1 = scaler.inverse_transform(pre1.reshape(-1, 1))
pre2 = scaler.inverse_transform(pre2.reshape(-1, 1))
test1 = scaler.inverse_transform(testY1.reshape(-1, 1))
test2 = scaler.inverse_transform(testY2.reshape(-1, 1))


y2 = (test1 - test2).ravel()
y = (pre1 - pre2).ravel()

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
df.to_excel('D:/abmathdata/task3PLUS/结果/.xlsx',index=False)
plt.plot(range(38),y,color='g')
plt.plot(range(38),y2,color='b')
# plt.xticks(range(1,1200,100))
plt.xticks(range(0,43,3))
plt.show()

# trainX,trainY = createXY(df_to_train_scale,n_past)
# testX,testY = createXY(df_to_test_scale,n_past)



# gridModel = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
# parameters = {
#     'batch_size':[16],
#     'epochs':[50],
#     'optimizer':['adam']
# }
# grid_search = GridSearchCV(estimator=gridModel,param_grid=parameters,cv=2)

# 修改部分


# 上面是修改部分


# grid_search = grid_search.fit(trainX,trainY)
# print(grid_search.best_params_)
# my_model = grid_search.best_estimator_



# # 逆变换预测和实际值
# predict_inversed = scaler.inverse_transform(predict.reshape(-1, 1))

# # 注意：假设testY已经是标准化后的数据，我们需要对其进行逆变换
# testY_inversed = scaler.inverse_transform(testY.reshape(-1, 1))

# # 计算评估指标
# mae_lgb = mean_absolute_error(testY_inversed, predict_inversed)
# mse_lgb = mean_squared_error(testY_inversed, predict_inversed)
# rmse_lgb = np.sqrt(mse_lgb)
# r2_lgb = r2_score(testY_inversed, predict_inversed)

# # 打印评估结果
# print("MAE:", mae_lgb)
# print("MSE:", mse_lgb)
# print("RMSE:", rmse_lgb)
# print("R^2:", r2_lgb)

# plt.plot(range(len(predict_inversed)),predict_inversed,color='g')
# plt.plot(range(len(testY_inversed)),testY_inversed,color='b')
# # plt.xticks(range(1,1200,100))
# plt.show()