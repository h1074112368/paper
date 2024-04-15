import os.path

import  netCDF4 as nc
import numpy as np
import cv2
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shutil
path=r'D:\tensorflow\lisflood\shanghai\maps'
# day_nc.to_netcdf(path+'/area.nc')
mask=np.array(nc.Dataset(path+'/area.nc')['value'])
chan=np.array(nc.Dataset(path+'/chan.nc')['value'])
# ncSet=np.array(nc.Dataset(r"D:\tensorflow\lisflood\shanghai\out\srun.nc")['srun'][0,:,:])
# ncSet=np.array(nc.Dataset(r"D:\tensorflow\lisflood\shanghai\out\srun.nc")['srun'][0,:,:])
#"D:\tensorflow\lisflood\feite_wdept.nc"
# ncSet=np.array(nc.Dataset(r"D:\tensorflow\lisflood\feite_wdept.nc")['wdept'])
# ncSet[np.where((mask==1)&(chan==1))]=0
# ncSet=np.array(nc.Dataset(r"F:\bsPaper\out_hour\wdept.nc")['wdept'])
ncDataset=nc.Dataset(r"F:\bsPaper\out_hour\wdept.nc")
ncSet=np.array(ncDataset['wdept'])
def timeSerial(ncSet):
     timeunits=ncDataset['time'].units.split(' ')
     if timeunits[0]=='days':
          jiange=60*60*24
     starttime=timeunits[2]+' '+timeunits[3].split("\r")[0]

     import datetime
     from dateutil.relativedelta import relativedelta
     timeArray=datetime.datetime.strptime(starttime,"%Y-%m-%d %H:%M:%S.%f")
     timeSet=[]
     for i in range(ncSet.shape[0]):
          if timeunits[0]=='days':
               temp=timeArray+relativedelta(days=i)
          if timeunits[0] == 'hours':
               temp = timeArray + relativedelta(hours=i)
               #timeSet.append(timeArray+relativedelta(days=i))
          timeSet.append(temp.strftime("%Y-%m-%d_%H_%M_%S"))
     return timeSet
timeSet=timeSerial(ncSet)
# ncSet=np.array(nc.Dataset(r"D:\tensorflow\lisflood\shanghai\out\srun.nc")['srun'])
for i in range(ncSet.shape[0]):
# for i in range(len(data)):
#      print([i,np.max(data[i])])

     # cv2.imshow('a',dataSet[i])
     # cv2.waitKey(0)
     ncSet[i][np.where((mask==0))]=0
#"D:\lisflod\lisflood-usecases-master\LF_lat_lon_UseCase\maps\elvstd.nc"
# nc=xr.open_dataset(r"D:\lisflod\lisflood-usecases-master\LF_lat_lon_UseCase\maps\elvstd.nc")
# nc1=xr.open_dataset(r"D:\tensorflow\lisflood\shanghai\maps\elvstd.nc")
# data=np.array(ncSet.variables['srun'])
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')#MP4格式
# #完成写入对象的创建，第一个参数是合成之后的视频的名称，第二个参数是可以使用的编码器，第三个参数是帧率即每秒钟展示多少张图片，第四个参数是图片大小信息
# videowrite = cv2.VideoWriter('feite_wdept_lai_es.mp4',fourcc,4,(ncSet.shape[2],ncSet.shape[1]),isColor=True)#2是每秒的帧数，size是图片尺寸
set=[]
# for i in range(len(dataSet)-1):
#     temp=(dataSet[i+1]-dataSet[i])/10
#     for j in range(10):
#         set.append(dataSet[i]+(j)*temp)
# set.append(dataSet[len(dataSet)-1])
#ncSet*=20
img=np.zeros([*ncSet.shape,4],dtype='uint8')
ncSet[ncSet<0]=0
# ncSet*=100
img[np.where((ncSet<0.15)&(ncSet>0.05))]=[240,237,182,255]#BGRA
img[np.where((ncSet<0.27)&(ncSet>=0.15))]=[232,180,116,255]
img[np.where((ncSet<0.4)&(ncSet>=0.27))]=[224,131,31,255]
img[np.where((ncSet<0.6)&(ncSet>=0.4))]=[184,68,29,255]
img[np.where(ncSet>=0.6)]=[145,9,9,255]
# img[np.where(ncSet>=0)]=[0,255,0]
# max=np.max(ncSet)
# min=np.min(data)
# data[data==-9999]=0
import matplotlib.pyplot as plt

if os.path.exists('../png/flood'):
     shutil.rmtree('../png/flood')
     os.mkdir('../png/flood')
else:
     os.mkdir('../png/flood')
for i in range(img.shape[0]):

     saveFile = "../png/flood/"+timeSet[i]+".png"  # 带有中文的保存文件路径
     # img3 = cv2.imread(imgFile, flags=1)
     cv2.imwrite(saveFile, img[i])  # imwrite 不支持中文路径和文件名，读取失败，但不会报错!
     # img_write = cv2.imencode(".png", img[i])[1].tofile(saveFile)
