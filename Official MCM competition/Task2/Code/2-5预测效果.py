import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score,accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import shap

# path = 'D:/abmathdata/task2PLUS/前提/All_play_Data-用于预测.xlsx'
# data = pd.read_excel(path)
# data_For_shaixuan = data.drop(columns=['groups','player','time','Victor','momentum'] )
# data_For_y = data[['Victor']]
# x_train,x_test,y_train,y_test = train_test_split(data_For_shaixuan,data_For_y,test_size=0.3,random_state=42)
# model = RandomForestClassifier(n_estimators=100,random_state=42,max_depth=12,min_samples_leaf=10,min_samples_split=10)

# model.fit(x_train,y_train)

# 重要性 = model.feature_importances_
# 特征 = data_For_shaixuan.columns

# dp = pd.DataFrame({
#     '特征':特征,
#     '重要性':重要性
# })

# dp.to_excel('D:/abmathdata/task2PLUS/结果/重要度.xlsx')

def knn_cls(x_train,x_test,y_train,y_test):
    knn = KNeighborsClassifier()
    param_grid = {
    'n_neighbors': [3,6,9,12,15,18],  # 尝试不同的邻居数
    'metric': [ 'manhattan']  # 尝试不同的距离度量
    }

    print(111)
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
    grid_search.fit(x_train,y_train)
    return grid_search

if __name__ == '__main__':
    path = 'D:/abmathdata/task2PLUS/前提/All_play_Data-用于预测.xlsx'
    data = pd.read_excel(path)
    data_For_shaixuan = data.drop(columns=['groups','player','time','Victor','V16','V8','V10','V11','momentum'] )
    data_For_y = data[['Victor']]
    x_train,x_test,y_train,y_test = train_test_split(data_For_shaixuan,data_For_y,test_size=0.3,random_state=42)
    # gridsearch = knn_cls(x_train,x_test,y_train,y_test)
    # model = gridsearch.best_estimator_
    # pre_y = model.predict(x_test)
    # print('对于测试集：')
    # acc_for_test = accuracy_score(y_test,pre_y)
    # auc_for_test = roc_auc_score(y_test,pre_y)
    # confusion_matrix_for_test = confusion_matrix(y_test,pre_y)
    # classification_report_for_test = classification_report(y_test,pre_y)
    # print('acc:',acc_for_test,'\nauc:',auc_for_test,'\nconfusion:',confusion_matrix_for_test,'\nreport:',classification_report_for_test)   
    # 使用KernelExplainer计算SHAP值 - 注意KNN不支持TreeExplainer
# 使用一小部分训练数据作为背景数据集
    # background = x_train.sample(100, random_state=42)

    # explainer = shap.KernelExplainer(model.predict_proba, background)
    # shap_values = explainer.shap_values(x_test.sample(50, random_state=42), n_jobs=-1)  # 并行计算，并只对测试集的一个子集计算SHAP值

    # # 绘制SHAP值
    # shap.summary_plot(shap_values, x_test.sample(50, random_state=42), feature_names=data_For_shaixuan.columns.tolist())

    accu1 = []
    data_For_shaixuan = data.drop(columns=['groups','player','time','Victor','momentum','V16','V8','V10','V11'] )
    data_For_y = data[['Victor']]
    for i in np.arange(0.1, 0.41, 0.01):
        x_train,x_test,y_train,y_test = train_test_split(data_For_shaixuan,data_For_y,test_size=i,random_state=42)
        model1 = KNeighborsClassifier(n_neighbors=12,metric='manhattan')
        model1.fit(x_train,y_train)
        pre_y = model1.predict(x_test)
        accu1.append(accuracy_score(y_test,pre_y))
        
        
    data_For_shaixuan = data.drop(columns=['groups','player','time','Victor','V16','V8','V10','V11'] )
    data_For_y = data[['Victor']]      
    accu2 = []  
    for i in np.arange(0.1, 0.41, 0.01):
        x_train,x_test,y_train,y_test = train_test_split(data_For_shaixuan,data_For_y,test_size=i,random_state=42) 
        model2 = KNeighborsClassifier(n_neighbors=6,metric='manhattan') 
        model2.fit(x_train,y_train)
        pre_y = model2.predict(x_test)
        accu2.append(accuracy_score(y_test,pre_y))
    

print(accu1)
print(accu2)

from scipy import stats


# 进行Shapiro-Wilk 测试
stat, p = stats.shapiro(accu1)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# 解释
alpha = 0.05
if p > alpha:
    print('样本看起来是正态分布的 (接受 H0)')
else:
    print('样本看起来不是正态分布的 (拒绝 H0)')
    
stat, p = stats.shapiro(accu2)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# 解释
alpha = 0.05
if p > alpha:
    print('样本看起来是正态分布的 (接受 H0)')
else:
    print('样本看起来不是正态分布的 (拒绝 H0)')
    
    
from scipy.stats import wilcoxon

# 假设x和y是两组配对的观测值
x = accu1
y = accu2

# x = [0,1,2,3,5,9]
# y = [3,5,7,5,9,10]

# 执行Wilcoxon符号秩检验
stat, p = wilcoxon(x, y)

print(f'统计量={stat}, p值={p}')

# 根据p值判断结果（以0.05为显著性水平）
alpha = 0.05
if p > alpha:
    print('差异不显著，不能拒绝零假设（两个配对样本来自相同的分布）')
else:
    print('差异显著，拒绝零假设（两个配对样本不来自相同的分布）')