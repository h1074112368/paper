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
yuce = np.load(r"F:\bsPaper\yuce.npy")
shiji = np.load(r"F:\bsPaper\shiji.npy")
yuce_rf = np.load(r"F:\bsPaper\yuce_rf.npy")
shiji_rf = np.load(r"F:\bsPaper\shiji_rf.npy")
yuce_cnn = np.load(r"F:\bsPaper\yuce_cnn.npy")
shiji_cnn = np.load(r"F:\bsPaper\shiji_cnn.npy")
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

fig,ax = plt.subplots(3, 3,figsize = (7,8),sharey=True)

vmin = data_min
vmax = data_max
#Normalize()跟归一化没有任何关系，函数的作用是将颜色映射到vmin-vmax上，
#颜色表/颜色柱的起始和终止分别取值vmin和vmax
norm = Normalize(vmin = vmin,vmax = vmax)
extent = (0,1,0,1)
point=[
    (1,1),
    (1,32),
    (1,59),
(31,1),
    (31,32),
    (31,59),
(63,6),
    (63,32),
    (63,59),

]
ii=0
import string
point_name=string.ascii_lowercase
for i in range(3):
    for j in range(3):
        # im1 = ax[i][j].plot(np.array(list(range(5,5*len(dataset)+5,5))),dataset[:,point[ii][0],point[ii][1]])
        im1 = ax[i][j].plot(np.array(list(range(5, 5 * len(yuce) + 5, 5))), yuce[:, point[ii][0], point[ii][1]],linewidth=1,label='ST-FLOOD')
        im1 = ax[i][j].plot(np.array(list(range(5, 5 * len(shiji) + 5, 5))), shiji[:, point[ii][0], point[ii][1]],linewidth=1,label='LISFLOOD')
        # im1 = ax[i][j].plot(np.array(list(range(5, 5 * len(yuce_rf) + 5, 5))), yuce_rf[:, point[ii][0], point[ii][1]],
        #                     linewidth=1, label='RF')
        # im1 = ax[i][j].plot(np.array(list(range(5, 5 * len(shiji_rf) + 5, 5))), shiji_rf[:, point[ii][0], point[ii][1]],
        #                     linewidth=1, label='LISFLOOD')
        # im1 = ax[i][j].plot(np.array(list(range(5, 5 * len(yuce_cnn) + 5, 5))), yuce_cnn[:, point[ii][0], point[ii][1]],
                            # linewidth=1, label='CNN')
        # im1 = ax[i][j].plot(np.array(list(range(5, 5 * len(shiji_cnn) + 5, 5))), shiji_cnn[:, point[ii][0], point[ii][1]],
        #                     linewidth=1, label='LISFLOOD')
        ax[i][j]#.set_axis_off()
        ax[i][j].set_title(f"({point_name[ii]})")
        if j==0:
            ax[i][j].set_ylabel('积水深度(m)')
        if i == 2:
            ax[i][j].set_xlabel('时间(min)')
        ii+=1

# im2 = ax[1].imshow(dataset[1],extent = extent,norm = norm,cmap = 'jet')
# ax[1].set_axis_off()
# im3 = ax[2].imshow(dataset[2],extent = extent,norm = norm,cmap = 'jet')
# ax[2].set_axis_off()
# ax[2].text(.8,-.02,'\nVisualization by DataCharm',transform = ax[2].transAxes,
#         ha='center', va='center',fontsize = 10,color='black')

fig.subplots_adjust(bottom=0.1,top=0.90)
fig.subplots_adjust(hspace =0.3)
# fig.tight_layout()
lines, labels = fig.axes[-1].get_legend_handles_labels()

fig.legend(lines, labels, loc='upper center',ncol=2)

#前面三个子图的总宽度为全部宽度的 0.9；剩下的0.1用来放置colorbar
# fig.subplots_adjust(right=0.9)
# position = fig.add_axes([0.9, 0.02, 0.02, 0.96 ])#位置[左,下,右,上]
# cb = fig.colorbar(im1, cax=position)

my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)

#设置colorbar标签字体等
# colorbarfontdict = {"size":15,"color":"k",'family':'Times New Roman'}
# cb.ax.set_title('积水深度(m)',pad=8)
# cb.ax.set_ylabel('积水深度(m)')
# cb.ax.tick_params(labelsize=11,direction='in')
#cb.ax.set_yticklabels(['0','10','20','30','40','50','>60'],family='Times New Roman')
# fig.suptitle('场景设计积水深度变化 ',size=22,
#              x=.55,y=.95)
# plt.savefig(r'F:\DataCharm\Python-matplotlib 空间数据可视化\map_colorbar.png',dpi = 600,
#             bbox_inches='tight',width = 12,height=4)
plt.show()