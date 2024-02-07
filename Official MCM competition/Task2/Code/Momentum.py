import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

path = 'D:/abmathdata/task2/All_play_for1.xlsx'
data = pd.read_excel(path)

df = data.iloc[:,3:-3]
df_y = data[['momentum']]


new_data = pd.DataFrame()
cnt = 0
for i in df.columns:
    cnt +=1
    if cnt== 5:
        break
    new_data[i] = np.random.randint(df[i].min(),df[i].max(), size=2375)
    
cnt = 0
for i in df.columns:
    cnt+=1
    if cnt<5:
       continue
    new_data[i] = np.random.uniform(low=df[i].min(), high=df[i].max(), size=2375)
    
w = [0.015424,0.029805,0.015783,0.064550,0.032772,0.122131,0.062999,0.004656,0.010041,0.177961, 0.205174,0.160125,0.068978,0.023963,0.005637]
print(len(w))
new_data['momentum'] = new_data.dot(w)
new_data['momentum'] =  MinMaxScaler().fit_transform(new_data[['momentum']])

print(new_data)