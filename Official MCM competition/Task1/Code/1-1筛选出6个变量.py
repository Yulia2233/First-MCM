'''
template writing by Yu Liu

'''
import pandas as pd
import numpy as np

path1 = 'D:/abmathdata/task1PLUS/前提/player1.xlsx'
player1 = pd.read_excel(path1)


path2 = 'D:/abmathdata/task1PLUS/前提/player2.xlsx'
player2 = pd.read_excel(path2)

# player1 = player1[['groups','player','time','p_sub_scores分数差','p_sub_sets','p_sub_games胜局数差','serve_players','p_scores胜利分数差','p_breaks破球率']]
# player2 = player2[['groups','player','time','p_sub_scores分数差','p_sub_sets','p_sub_games胜局数差','serve_players','p_scores胜利分数差','p_breaks破球率']]

# 创建一个ExcelWriter实例
# with pd.ExcelWriter('D:/abmathdata/task1PLUS/过程/Player.xlsx') as writer:
#     player1.to_excel(writer, sheet_name='player1', index=False)
#     player2.to_excel(writer, sheet_name='player2', index=False)
    
player1 = player1.drop(columns=['p_sub_scores分数差','p_sub_sets','p_sub_games胜局数差','serve_players','p_scores胜利分数差','p_breaks破球率','momentum'])
player2 = player2.drop(columns=['p_sub_scores分数差','p_sub_sets','p_sub_games胜局数差','serve_players','p_scores胜利分数差','p_breaks破球率','momentum'])
    

with pd.ExcelWriter('D:/abmathdata/task1PLUS/过程/Player-用于预测.xlsx') as writer:
    player1.to_excel(writer, sheet_name='player1', index=False)
    player2.to_excel(writer, sheet_name='player2', index=False)

# 存放到过程中