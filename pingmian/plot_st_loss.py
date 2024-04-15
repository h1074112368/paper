import  netCDF4 as nc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from pylab import*
from  matplotlib.font_manager import FontProperties
# plt.rcParams["figure.autolayout"] = True
mpl.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['font.sans-serif']=['SimHei']

data_val=pd.read_csv(r"F:\bsPaper\validation.csv")
data_train=pd.read_csv(r"F:\bsPaper\train.csv")

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
# fig=plt.figure()
# ax = plt.subplot(111)


# extent = (0,1,0,1)
ii=0
import string
point_name=string.ascii_lowercase

im1, = plt.plot(data_train.iloc[:, 1], data_train.iloc[:, 2], linewidth=1, label='train')
# im1 = ax[i][j].plot(np.array(list(range(5,5*len(dataset)+5,5))),dataset[:,point[ii][0],point[ii][1]])
im2, = plt.plot(data_val.iloc[:,1], data_val.iloc[:,2],linewidth=1,label='validation')


plt.ylabel('Loss')

plt.xlabel('Epoch')


# im2 = ax[1].imshow(dataset[1],extent = extent,norm = norm,cmap = 'jet')
# ax[1].set_axis_off()
# im3 = ax[2].imshow(dataset[2],extent = extent,norm = norm,cmap = 'jet')
# ax[2].set_axis_off()
# ax[2].text(.8,-.02,'\nVisualization by DataCharm',transform = ax[2].transAxes,
#         ha='center', va='center',fontsize = 10,color='black')

# fig.subplots_adjust(bottom=0.1,top=0.90)
# fig.subplots_adjust(hspace =0.3)
# fig.tight_layout()
# lines, labels = fig.axes[-1].get_legend_handles_labels()

plt.legend(handles=[im1,im2],loc="upper right")
# fig.legend()
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