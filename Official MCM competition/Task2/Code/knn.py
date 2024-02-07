from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Model
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import xgboost as xgb

# 返回 X ，总Y ，单列Y
def read_xy(pathx,pathy,name):
    df_x = pd.read_excel(pathx)
    df_y = pd.read_excel(pathy)
    # 处理df_x
    # 处理df_y
    df_y_1 = df_y[[name]]
    print('下面是x的数据:')
    print(df_x.head(5))
    print('下面是y的数据:')
    print(df_y.head(5))
    print(df_y_1.head(5))
    return df_x,df_y,df_y_1



def calculate_specificity(y_true, y_pred):
    """
    计算特异性
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 特异性值
    """
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # 计算特异性
    specificity = tn / (tn + fp)
    return specificity




def knn_cls(x_train,y_train):
    knn = KNeighborsClassifier()
    param_grid = {
    'n_neighbors': [3, 5, 7, 10,12,16,18],  # 尝试不同的邻居数
    'metric': ['euclidean', 'manhattan']  # 尝试不同的距离度量
}

    print(111)
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)
    grid_search.fit(x_train,y_train)
    return grid_search

def svc_cls(x_train,y_train):
    svc = SVC()
    param_grid = {'C': [ 1], 
     'kernel': ['rbf','sigmoid'], 
     'gamma': [ 'auto', 0.1, 1]
     }
    print(111)
    grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5)
    grid_search.fit(x_train,y_train)
    return grid_search

def xgb_cls(x_train,y_train):
    model = xgb.XGBClassifier()
    param_grid = {'learning_rate':[0.01,0.05,0.1],
     'n_estimators': [100,200,300,400,500], 
     'max_depth': [2,4,8,16]
     }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid_search.fit(x_train,y_train)
    return grid_search

if __name__ == '__main__':
    path = 'D:/abmathdata/task3PLUS/结果/All_play_Data-分开11.xlsx'
    data = pd.read_excel(path)
    data = data.drop(columns=['groups','player','time','momentum','Victor','player2','momentum-','Victor-','translate'])
    X = data.drop(columns=['IS'])
    Y = data[['IS']]
    X = X.iloc[155:181,:]
    Y = Y.iloc[155:181,:]
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4)
    gridsearch = xgb_cls(x_train,y_train)
    model = gridsearch.best_estimator_
    pre_y = model.predict(x_test)
    print('最好的模型:',gridsearch.best_params_)
    print('对于测试集：')
    acc_for_test = accuracy_score(y_test,pre_y)
    auc_for_test = roc_auc_score(y_test,pre_y)
    confusion_matrix_for_test = confusion_matrix(y_test,pre_y)
    classification_report_for_test = classification_report(y_test,pre_y)
    print(acc_for_test,auc_for_test,confusion_matrix_for_test,classification_report_for_test)    
    print('特异性:',calculate_specificity(y_test,pre_y))
    # name = 'MN'
    # pathx = f'D:/data/task3/x-train/x-{name}.xlsx'
    # pathy = f'D:/data/task3/y-train/y-{name}.xlsx'
    # df_x,df_y,df_y_1 = read_xy(pathx=pathx,pathy=pathy,name=name)
    # x_train,x_test,y_train,y_test = train_test_split(df_x,df_y_1,test_size=0.3,random_state=42)
    # gridsearch = knn_cls(x_train,x_test,y_train,y_test)
    # print('最好的模型:',gridsearch.best_params_)
    # model = gridsearch.best_estimator_
    
    # pre_y = model.predict(x_test)
    # print(pre_y)