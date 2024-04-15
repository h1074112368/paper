import  netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pylab import*
from  matplotlib.font_manager import FontProperties
# plt.rcParams["figure.autolayout"] = True
mpl.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['SimHei']
dataset = np.array(nc.Dataset(r"F:\bsPaper\wdept1016.nc")["wdept"])

data_min=np.min(dataset)
data_max=np.max(dataset)

plt.style.use('seaborn')
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False
# 设置font字典为 SimSun（宋体），大小为12（默认为10）
font = {'family': 'SimSun',
        'size': '12'}
# 设置 字体
plt.rc('font', **font)
# 解决中文字体下坐标轴负数的负号显示问题
plt.rc('axes', unicode_minus=False)

fig,ax = plt.subplots(6, 4,figsize = (6,8),sharey=True)

vmin = data_min
vmax = data_max
#Normalize()跟归一化没有任何关系，函数的作用是将颜色映射到vmin-vmax上，
#颜色表/颜色柱的起始和终止分别取值vmin和vmax
norm = Normalize(vmin = vmin,vmax = vmax)
extent = (0,1,0,1)
timei=0
for i in range(6):
    for j in range(4):
        im1 = ax[i][j].imshow(dataset[int(i*4+j)],extent = extent,norm = norm,cmap = 'Blues')
        ax[i][j].set_axis_off()
        ax[i][j].set_title(f"t{timei}")
        timei+=1
# im2 = ax[1].imshow(dataset[1],extent = extent,norm = norm,cmap = 'jet')
# ax[1].set_axis_off()
# im3 = ax[2].imshow(dataset[2],extent = extent,norm = norm,cmap = 'jet')
# ax[2].set_axis_off()
# ax[2].text(.8,-.02,'\nVisualization by DataCharm',transform = ax[2].transAxes,
#         ha='center', va='center',fontsize = 10,color='black')

fig.subplots_adjust(right=0.85)
fig.tight_layout()
#前面三个子图的总宽度为全部宽度的 0.9；剩下的0.1用来放置colorbar
fig.subplots_adjust(right=0.9)
position = fig.add_axes([0.9, 0.02, 0.02, 0.96 ])#位置[左,下,右,上]
cb = fig.colorbar(im1, cax=position)

my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)

#设置colorbar标签字体等
colorbarfontdict = {"size":15,"color":"k",'family':'Times New Roman'}
# cb.ax.set_title('积水深度(m)',pad=8)
cb.ax.set_ylabel('积水深度(m)')
cb.ax.tick_params(labelsize=11,direction='in')
#cb.ax.set_yticklabels(['0','10','20','30','40','50','>60'],family='Times New Roman')
# fig.suptitle('场景设计积水深度变化 ',size=22,
#              x=.55,y=.95)
# plt.savefig(r'F:\DataCharm\Python-matplotlib 空间数据可视化\map_colorbar.png',dpi = 600,
#             bbox_inches='tight',width = 12,height=4)
plt.show()