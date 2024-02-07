import pandas as pd


path = 'D:/abmathdata/task1/结果/CRITIC法综合评价结果player1.xlsx'
df1 = pd.read_excel(path)

path = 'D:/abmathdata/task1/结果/CRITIC法综合评价结果player2.xlsx'
df2 = pd.read_excel(path)


df = pd.DataFrame()
df['groups'] = df1['groups']
df['time'] = df1['time']
df['差'] = df1['momentum'] - df2['momentum']
df.to_excel('差.xlsx')


