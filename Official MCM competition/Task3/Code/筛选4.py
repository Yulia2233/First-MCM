from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
path = 'D:/abmathdata/task3PLUS/结果/All_play_Data-分开.xlsx'
df = pd.read_excel(path)
df_x = df.drop(columns=['groups','player','time','momentum','Victor','translate','momentum-','Victor-','player2'])
df_y = df[['IS']]

# 实例化基模型
model = RandomForestClassifier()
X_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state=42)

# RFE 实例化和运行
selector = RFE(model, n_features_to_select=15, step=20)
selector = selector.fit(X_train, y_train.values)

# 获取所选特征的掩码
selected_features = selector.support_

# 获取选择的特征
X_selected = X_train.columns[selected_features]

print(X_selected)