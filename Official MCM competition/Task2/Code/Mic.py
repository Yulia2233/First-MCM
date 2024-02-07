from minepy import MINE
import pandas as pd



#读取y的数据
path = 'D:/abmathdata/task1/结果/All_play_for1.xlsx'
x = pd.read_excel(path)


p = []
mine = MINE()
mine.compute_score(x['momentum'].values.ravel(),x['p_breaks'].values.ravel())
mic = mine.mic()
p.append(mic)
# mine.compute_score(x['momentum'].values.ravel(),x['p_streaks'].values.ravel())
# mic = mine.mic()
# p.append(mic)
print(p)

