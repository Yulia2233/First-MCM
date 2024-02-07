import pandas as pd

'''
template writing by Yu Liu

'''

path =  'D:/abmathdata/prepo/Data.xlsx'
df = pd.read_excel(path)

c = [0,0,0,0,0]
for i in df['serve_width发球方向']:
    if i == 0:
        c[0]+=1
    elif i == 1:
        c[1]+=1
    elif i == 2:
        c[2]+=1
    elif i == 3:
        c[3]+=1
    elif i == 4:
        c[4]+=1
    else:
        print('有异常值!!')
        
print(c)