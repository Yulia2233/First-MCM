'''
template writing by Yu Liu

'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
try:
    # 创建 FontProperties 对象，引入自定义字体
    custom_font_path = 'D:/font/Bookman Old Style.ttf'
    custom_font = FontProperties(fname=custom_font_path)
except:
    print('导入失败!')
sns.set_style('dark')  
path = 'D:/abmathdata/task1PLUS/结果/Player.xlsx'
p1 = pd.read_excel(path,sheet_name='player1')
p2 = pd.read_excel(path,sheet_name='player2')

plt.plot(range(1,47),p1.loc[1142:,'momentum'], color=sns.color_palette("YlGnBu",8)[7],marker='o',linewidth=3,label='Carlos Alcaraz')
plt.plot(range(1,47),p2.loc[1142:,'momentum'],color=sns.color_palette("YlGnBu",8)[3],marker='o',linewidth=3,label='Novak Djokovic')
plt.title('', fontproperties=custom_font, fontsize=30)
plt.xlabel('Number Of Game', fontproperties=custom_font, fontsize=24)
plt.ylabel('Momentum', fontproperties=custom_font, fontsize=24)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xticks(range(1,45,3))
plt.legend( fontsize=24, prop={'size': 18, 'family': custom_font.get_name()})
plt.show()