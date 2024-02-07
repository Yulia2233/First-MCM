import pandas as pd


path = 'D:/abmathdata/task4/前提/美国网球公开赛.xlsx'

df = pd.read_excel(path)

df['speed_mph'] = df['speed_mph'].round()

df['serve_width'] = df['serve_width'].round()

df['serve_depth'] = df['serve_depth'].round()

df['return_depth'] = df['return_depth'].round()

'''
template writing by Yu Liu

'''
df.to_excel('D:/abmathdata/task4/前提/美国网球公开赛.xlsx',index=False)