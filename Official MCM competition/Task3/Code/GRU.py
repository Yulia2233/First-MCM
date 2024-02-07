import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import GRU, Dense,Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import math


n_past = 30  # 记录次数
p = 0.3     # 比例
featurenum = 20     # 特征数量


np.set_printoptions(suppress=True, precision=8)


# 读取X的数据
path = 'D:/abmathdata/task3PLUS/前提/All_play_Data-用于预测.xlsx'
df = pd.read_excel(path)
df_all = df.drop(columns=['groups','player','time','Victor','V14','V13','V8'])
df_x = df_all.drop(columns=['momentum'])
df_y = df_all['momentum']
df_x.insert(0,'momentum',df_y)
df = df_x


# 拆分训练集和测试集
#测试集比例
test_split = round(len(df)*p)
# print(test_split)
df_to_train = df[:-test_split]
# print(len(df_to_train))
df_to_test = df[-test_split:]
# print(len(df_to_test))

df_read_test = pd.read_excel('D:/data/Q1_test.xlsx')
df_to_test = df_read_test.drop(columns=['SMILES'])
df_mark = pd.DataFrame({
    'pIC50':[1]*50
})
df_to_test.insert(0,'pIC50',df_mark)

scaler = MinMaxScaler(feature_range=(0,1))
numeric_columns = df_to_train.columns
df_to_train_scale = scaler.fit_transform(df_to_train[numeric_columns],df_to_train.columns)
df_to_test_scale = scaler.fit_transform(df_to_test[numeric_columns],df_to_train.columns)
# print(df_to_train_scale)
 
print(df_to_test_scale.shape)
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



def build_model(optimizer):
    model = Sequential()
    model.add(GRU(500, return_sequences=False ,input_shape=(None,20)))
    model.add(Dense(units=500))
    model.add(Dense(units=500, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer=optimizer, loss='mse')
    model.summary()
    return model




# gridModel = KerasRegressor(build_fn=build_model,verbose=1,validation_data=(x_train,y_train))
# parameters = {
#     'batch_size':[32],
#     'epochs':[10],
#     'optimizer':['adam']
# }
# grid_search = GridSearchCV(estimator=gridModel,param_grid=parameters,cv=2)
# grid_search = grid_search.fit(x_train,y_train)
# print(grid_search.best_params_)

# model = grid_search.best_estimator_

model = build_model('adam')
model.fit(trainX,trainY,batch_size=32,epochs=100)
predict = model.predict(testX)

predict_cop = np.repeat(predict,df.shape[1],axis=-1)
pre = scaler.inverse_transform(np.reshape(predict_cop,(len(predict),df.shape[1])))[:,0]

testY_cop = np.repeat(testY,df.shape[1],axis=-1)
tY = scaler.inverse_transform(np.reshape(testY_cop,(len(predict),df.shape[1])))[:,0]

IC = []
for i in pre:
    IC.append(-math.log10(i*(10**-9)))
print(pre)
print(tY)
print(pre.shape)
prin = pd.DataFrame({
    'SMILES':df_read_test['SMILES'],
    'IC50_nM':IC,
    'pIC50':pre
})

mae_lgb = mean_absolute_error(tY, pre)
mse_lgb = mean_squared_error(tY, pre)
rmse_lgb = np.sqrt(mse_lgb)
r2_lgb = r2_score(tY, pre)

# 打印评估结果
print("MAE:", mae_lgb)
print("MSE:", mse_lgb)
print("RMSE:", rmse_lgb)
print("R^2:", r2_lgb)

plt.plot(tY,color='red',label='Real')
plt.plot(pre,color='blue',label='Prediction')
plt.legend()
plt.show()
