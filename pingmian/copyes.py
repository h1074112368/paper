from timeArray import timeArray
import shutil,os
names=timeArray('2021-07-23 00','2021-07-29 00',24)
i=0
import  netCDF4 as nc
import numpy as np
import cv2
import xarray as xr
import matplotlib.pyplot as plt
from fangshanduozaizhong import craetTif
from osgeo import gdal, ogr, gdalconst
name='wdept'
# "F:\毕设论文\out_daily\wdept.nc"
path0=r'F:\bsPaper\out_daily\NotTw'
ncSet=np.array(nc.Dataset(r"G:\livap\output\efas\es.nc")['es'])
path=r"D:\tensorflow\lisflood\data_25m\landUseAddWudingAddwater.tif"
data = gdal.Open(path)
im_width = data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
im_height = data.RasterYSize  # 获取高度，数组第一维，上下方向元素长度，代表纬度范围
im_bands = data.RasterCount  # 波段数
"""
GeoTransform 的含义：
    影像左上角横坐标：im_geotrans[0]，对应经度
    影像左上角纵坐标：im_geotrans[3]，对应纬度

    遥感图像的水平空间分辨率(纬度间隔)：im_geotrans[5]
    遥感图像的垂直空间分辨率(经度间隔)：im_geotrans[1]
    通常水平和垂直分辨率相等

    如果遥感影像方向没有发生旋转，即上北下南，则 im_geotrans[2] 与 im_geotrans[4] 为 0

计算图像地理坐标：
    若图像中某一点的行数和列数分别为 row 和 column，则该点的地理坐标为：
        经度：xGeo = im_geotrans[0] + col * im_geotrans[1] + row * im_geotrans[2]
        纬度：yGeo = im_geotrans[3] + col * im_geotrans[4] + row * im_geotrans[5]
"""
im_geotrans = data.GetGeoTransform()  # 获取仿射矩阵，含有 6 个元素的元组

im_proj = data.GetProjection()  # 获取地理信息
for i in range(len(ncSet)):
    tifname=path0+'/'+name+'/'+name+"_"+"{:03d}".format(i)+".tif"
    craetTif(r'G:\data\es/' + names[i] + '.tif', im_width , im_height, 1, im_geotrans, im_proj, ncSet[i],-9999)
# for file in os.listdir(r'F:\yh_pr_hour\pre_re'):
#     if file.endswith('tif'):
#         # set.append([readTifArray(r'F:\yh_pr_hour\pre_re'+'/'+file)])
#         shutil.copyfile(r'F:\yh_pr_hour\pre_re'+'/'+file, r'G:\data\pr/' + name[i] + '.tif')
#         i+=1
# name=timeArray('2021-07-23 00','2021-07-28 23',1)
# for i in range(len(name)):
#     shutil.copyfile(r'F:\LAI\re_GLASS01D01.V60.A2021209.h28v05.2022151.tif',r'G:\data\lai/'+i+'.tif')