import pandas as pd

player1 = pd.read_excel('D:/abmathdata/task1/结果/player1_for1.xlsx')
player2 = pd.read_excel('D:/abmathdata/task1/结果/player2_for1.xlsx')






# 按行合并DataFrame，只保留第一个DataFrame的列名
df_combined = pd.concat([player1, player2], ignore_index=True)

df_combined.to_excel('D:/abmathdata/task1/结果/All_play_for1.xlsx',index=False)