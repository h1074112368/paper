import  netCDF4 as nc
import numpy as np
import cv2
import xarray as xr
import matplotlib.pyplot as plt
path=r'D:\tensorflow\lisflood\shanghai\maps'
# day_nc.to_netcdf(path+'/area.nc')
mask=np.array(nc.Dataset(path+'/area.nc')['value'])
chan=np.array(nc.Dataset(path+'/chan.nc')['value'])
# ncSet=np.array(nc.Dataset(r"F:\not_wuding_out\srun.nc")['srun'])

# ncSet[np.where((mask==1)&(chan==1))]=0
ncSet=np.array(nc.Dataset(r"D:\tensorflow\lisflood\shanghai\out\srun.nc")['srun'])
for i in range(ncSet.shape[0]):
# for i in range(len(data)):
#      print([i,np.max(data[i])])

     # cv2.imshow('a',dataSet[i])
     # cv2.waitKey(0)
     ncSet[i][np.where((mask==1)&(chan==1))]=0
#"D:\lisflod\lisflood-usecases-master\LF_lat_lon_UseCase\maps\elvstd.nc"
# nc=xr.open_dataset(r"D:\lisflod\lisflood-usecases-master\LF_lat_lon_UseCase\maps\elvstd.nc")
# nc1=xr.open_dataset(r"D:\tensorflow\lisflood\shanghai\maps\elvstd.nc")
# data=np.array(ncSet.variables['srun'])
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#MP4格式
# #完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
# videowrite = cv2.VideoWriter('Yanhua_wdept.mp4',fourcc,1,(ncSet.shape[2],ncSet.shape[1]),isColor=True)#2是每秒的帧数，size是图片尺寸
set=[]
# for i in range(len(dataSet)-1):
#     temp=(dataSet[i+1]-dataSet[i])/10
#     for j in range(10):
#         set.append(dataSet[i]+(j)*temp)
# set.append(dataSet[len(dataSet)-1])
ncSet=ncSet.astype('float32')
max=np.max(ncSet)
ncSet[ncSet<0]=0
ncSet*=20
wr=xr.open_dataset(r"D:\tensorflow\lisflood\shanghai\out\srun.nc")
wr['srun']=(['time', 'lat', 'lon'],ncSet)
wr.to_netcdf('feite_srun_4_27.nc')
# min=np.min(data)
# data[data==-9999]=0
# import matplotlib.pyplot as plt
# for i in range(ncSet.shape[0]):
#      import matplotlib
#      plt.imshow(ncSet[i],cmap='rainbow')#,vmax=20,vmin=0)
#      plt.colorbar()
#      plt.show()
# for i in range(len(data)):
#      print([i,np.max(data[i])])

     # cv2.imshow('a',dataSet[i])
     # cv2.waitKey(0)
     # ncSet[i][np.where((mask==1)&(chan==1))]=0
#      tmp= cv2.resize((ncSet[i]/max*255), (ncSet.shape[2],ncSet.shape[1]))
#      # dataSet[i] = cv2.resize(dataSet[i]/max*255,data.shape)
#
#      # img2 = cv2.cvtColor(dataSet[i], cv2.COLOR_GRAY2BGR)
#      tmp=tmp.astype(np.uint8)
#      # tmp[tmp<0]=0
#      img2 = cv2.applyColorMap(tmp, 20)#16
#      videowrite.write(img2)
#      # print('第{}张图片合成成功'.format(i))
# videowrite.release()
# # ncpre=nc.Dataset(r"D:\lisflod\lisflood-usecases-master\LF_ETRS89_UseCase\meteo\pr_daily.nc")
# print()