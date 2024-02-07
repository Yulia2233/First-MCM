import pandas as pd
from scipy.stats import pearsonr



path = 'D:/abmathdata/task2PLUS/前提/All_play_Data.xlsx'

data = pd.read_excel(path)


X = data['Victor']
X1 = data['p_streaks']

Y = data['momentum']


c1,p1 = pearsonr(X,Y)
c2,p2 = pearsonr(X,Y)

print(c1,p1)
print(c2,p2)