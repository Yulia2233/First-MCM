'''
template writing by Yu Liu

'''
import pandas as pd

path = 'D:/abmathdata/task4/前提/美国网球公开赛.xlsx'
df = pd.read_excel(path)
print(df.head(10))



allGroup = 0    #总局数
start = 0
times = []
p1s = []
p2s = []
mats = []
groups = []
p1_sub_scores = []      # p1-p2的差值
p2_sub_scores = []      # p2-p1的差值
p1_sub_sets = []        # p1-p2的差值
p2_sub_sets = []        # p2-p1的差值
p1_sub_games = []       # p1-p2的差值
p2_sub_games = []       # p2-p1的差值
serve_players = []       # 发球选手
p1_scores = []          # p1的分个数
p2_scores = []          # p2得分个数
p1_aces = []            # p1的ace球概率
p2_aces = []            # p2的ace球概率
p1_shots = []           # 击球得分
p2_shots = []   
p1_doubles = []         # 双误概率
p2_doubles = []
p1_1sts = []            # 单错误
p2_1sts = []
p1_nets = []            #截击次数
p2_nets = []
p1_breaks = []          # 破发成功率
p2_breaks = []
p1_faces = []           # 面临破点率
p2_faces = []
p1_wids = []            # 发球方向
p2_wids = []
p1_deps = []            # 深度
p2_deps = []     
p1_diss = []            # 举例
p2_diss = []  
p1_speed = []
p2_speed = []
p1_ret = []
p2_ret = []

for i in range(len(df['match_id'])-1): # i 是行
    if df.loc[i,'game_no 局数'] == df.loc[i+1,'game_no 局数']:
        continue
    else:
        allGroup += 1
        # print(df.loc[start:i,'game_no 局数'])    # 输出所有组            i是末尾，start开头
        mats.append(df.loc[start,'match_id'])
        p1s.append(df.loc[start,'player1'])
        p2s.append(df.loc[start,'player2'])
        times.append(df.loc[start,'elapsed_time'])
        groups.append(allGroup)
        
        
        # 总得分求差值
        p1_sub_score = df.loc[i,'p1_points_won总得分数。Q R计算差值作为新变量；差值为正则为正相关，负值为负相关'] - df.loc[i,'p2_points_won总得分数。Q R计算差值作为新变量；差值为正则为正相关，负值为负相关']
        p2_sub_score = -p1_sub_score
        
        p1_sub_scores.append(p1_sub_score)
        p2_sub_scores.append(p2_sub_score)
        
        
        # 赢得局数差值
        p1_sub_set = df.loc[i,'p1_sets赢得盘数。HI做差，差值为正则为正相关，负值为负相关'] - df.loc[i,'p2_sets赢得盘数。HI做差，差值为正则为正相关，负值为负相关']
        p2_sub_set = -p1_sub_set
        
        p1_sub_sets.append(p1_sub_set)
        p2_sub_sets.append(p2_sub_set)
        
        
        # 赢得的盘数差值
        p1_sub_game = df.loc[i,'p1_games赢得局数。JK做差，差值为正则为正相关，负值为负相关'] - df.loc[i,'p2_games赢得局数。JK做差，差值为正则为正相关，负值为负相关']
        p2_sub_game = -p1_sub_game
        
        p1_sub_games.append(p1_sub_game)
        p2_sub_games.append(p2_sub_game)

        
        # 发球选手
        serve_player = df.loc[i,'server 发球选手']
        
        
        serve_players.append(serve_player)
        
        
        # 得分个数
        # 计算1的个数
        count_1 = (df.loc[start:i, 'point_victor得分选手  （分成两个表）'] == 1).sum()
        # 计算2的个数
        count_2 = (df.loc[start:i, 'point_victor得分选手  （分成两个表）'] == 2).sum()
        
        p1_scores.append(count_1)
        p2_scores.append(count_2)
        
        # ace得分率
        p1_ace = (df.loc[start:i,'p1_ace 发球直接得分   正相关'].sum())/(i-start+1)
        p2_ace = (df.loc[start:i,'p2_ace发球直接得分     正相关'].sum())/(i-start+1)
        
        p1_aces.append(p1_ace)
        p2_aces.append(p2_ace)
        
        
        # 击球得分
        p1_shot = (df.loc[start:i,'p1_winner击球得分      正相关'].sum())/(i-start+1)
        p2_shot = (df.loc[start:i,'p2_winner击球得分    正相关'].sum())/(i-start+1)
        
        p1_shots.append(p1_shot)
        p2_shots.append(p2_shot)
        
        
        # 双误概率
        p1_double = (df.loc[start:i,'p1_double_fault两次发球失误     负相关'].sum())/(i-start+1)
        p2_double = (df.loc[start:i,'p2_double_fault两次发球失误。   负相关'].sum())/(i-start+1)
        
        p1_doubles.append(p1_double)
        p2_doubles.append(p2_double)
        
        
        # 单
        p1_1st = (df.loc[start:i,'p1_unf_err失误。负相关'].sum())/(i-start+1)
        p2_1st = (df.loc[start:i,'p2_unf_err失误。负相关'].sum())/(i-start+1)
        
        p1_1sts.append(p1_1st)
        p2_1sts.append(p2_1st)


        # 截击成功率
        if (df.loc[start:i,'p1_net_pt积极进攻。   正相关'].sum()) != 0 and (df.loc[start:i,'p2_net_pt积极进攻。   正相关'].sum())  != 0:
            p1_net = (df.loc[start:i,'p1_net_pt_won积极进攻得分     正相关'].sum())/(df.loc[start:i,'p1_net_pt积极进攻。   正相关'].sum())
            p2_net = (df.loc[start:i,'p2_net_pt_won积极进攻得分     正相关'].sum())/(df.loc[start:i,'p2_net_pt积极进攻。   正相关'].sum())  
        
            p1_nets.append(p1_net)
            p2_nets.append(p2_net)
        else:
            p1_nets.append(0)
            p2_nets.append(0)
        
        # 破发成功率
        p1_break = (df.loc[start:i,'p1_break_pt_won在p2发球局获胜'].sum())/(i-start+1)
        p2_break = (df.loc[start:i,'p2_break_pt_won在p1发球局获胜'].sum())/(i-start+1)
        
        p1_breaks.append(p1_break)
        p2_breaks.append(p2_break)
        
        #  面临破点率
        if serve_player == 1:
            p1_face = 0
            p2_face = (df.loc[start:i,'p2_break_pt有机会在p1发球局获胜；AI,AJ取交集，同时为1则正相关，反之负相关'].sum())/(i-start+1)
            
        else:
            p1_face = (df.loc[start:i,'p1_break_pt有机会在p2发球局获胜；AH，AJ取交集；同时为1则正相关，反之负相关'].sum())/(i-start+1)
            p2_face = 0
            
        p1_faces.append(p1_face)
        p2_faces.append(p2_face)
        
        # 发球方向
        if serve_player == 1:
            p1_wid = (df.loc[start:i,'serve_width发球方向；求方差 '].var())
            p2_wid = 0
            
        else:
            p2_wid = (df.loc[start:i,'serve_width发球方向；求方差 '].var())
            p1_wid = 0
            
            
        p1_wids.append(p1_wid)
        p2_wids.append(p2_wid)
        
        if serve_player == 1:
            p1_dep = (df.loc[start:i,'serve_depth发球深度；求方差'].var())
            p2_dep = 0
            
        else:
            p2_dep = (df.loc[start:i,'serve_depth发球深度；求方差'].var())
            p1_dep = 0
        
        p1_deps.append(p1_dep)
        p2_deps.append(p2_dep)
        
        if serve_player == 1:
            p2_rets = (df.loc[start:i,'return_depth回击发球深度：求方差'].var())
            p1_rets = 0
            
        else:
            p1_rets = (df.loc[start:i,'return_depth回击发球深度：求方差'].var())
            p2_rets = 0
        
        p1_ret.append(p1_rets)
        p2_ret.append(p2_rets)        
        
        
        
        p1_dis = df.loc[start:i,'p1_distance_run跑动距离 负相关；求和'].sum()
        p2_dis = df.loc[start:i,'p2_distance_run跑动距离 负相关；求和'].sum()
        
        p1_diss.append(p1_dis)
        p2_diss.append(p2_dis)
        
        # 发球方向
        if serve_player == 1:
            p1_speeds = (df.loc[start:i,'speed_mph发球速度。 '].mean())
            p2_speeds = 0
            
        else:
            p2_speeds = (df.loc[start:i,'speed_mph发球速度。 '].mean())
            p1_speeds = 0
            
            
        p1_speed.append(p1_speeds)
        p2_speed.append(p2_speeds)
        
        start = i+1
        
data = pd.DataFrame({
    'groups':groups,
    'player1':p1s,
    'player2':p2s,
    'time':times,
    'p1_sub_scores' :p1_sub_scores,     # p1-p2的差值
    'p2_sub_scores':p2_sub_scores,     # p2-p1的差值
    'p1_sub_sets' :p1_sub_sets,          # p1-p2的差值
    'p2_sub_sets' :p2_sub_sets,         # p2-p1的差值
    'p1_sub_games' :p1_sub_games,        # p1-p2的差值
    'p2_sub_games':p2_sub_games,        # p2-p1的差值
    'serve_players':serve_players,         # 发球选手
    'p1_scores' :p1_scores,            # p1的分个数
    'p2_scores':p2_scores,           # p2得分个数
    'p1_aces' :p1_aces,             # p1的ace球概率
    'p2_aces' :p2_aces,             # p2的ace球概率
    'p1_shots' :p1_shots,             # 击球得分
    'p2_shots' :p2_shots,     
    'p1_doubles' :p1_doubles,           # 双误概率
    'p2_doubles' :p2_doubles,  
    'p1_1sts' :p1_1sts,              # 单错误
    'p2_1sts' :p2_1sts,  
    'p1_nets':p1_nets,              #截击次数
    'p2_nets' :p2_nets,  
    'p1_breaks' :p1_breaks,            # 破发成功率
    'p2_breaks' :p2_breaks,  
    'p1_faces':p1_faces,            # 面临破点率
    'p2_faces' :p2_faces,  
    'p1_wids':p1_wids,             # 发球方向
    'p2_wids' :p2_wids,  
    
    'p1_diss' :p1_diss,              # 举例
    'p2_diss' :p2_diss,  
    'p1_ret':p1_ret,
    'p2_ret':p2_ret,
    'p1_speed':p1_speed,
    'p2_speed':p2_speed,
    'p1_deps' :p1_deps,             # 深度9*
    'p2_deps' :p2_deps,   
    
})        
# print(allGroup)

data.to_excel('D:/111111/data.xlsx',index=False)

