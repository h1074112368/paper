import  netCDF4 as nc
import numpy as np

import xarray as xr
import matplotlib.pyplot as plt
from fangshanduozaizhong import craetTif,readTifArray
from osgeo import gdal, ogr, gdalconst
cnbh=readTifArray(r"F:\bsPaper\re_CNBH10m\CNBH10M_hebing1.tif")
s1=readTifArray(r"F:\bsPaper\re_s1flood_shanghai.tif")
ul=readTifArray(r"D:\tensorflow\lisflood\dem_shanghai_25m\n51_30_2020lc030_land.tif")
err =  np.where(s1 == 20, 0, 1) * np.where(s1 == 10, 0, 1) * np.where(ul == 60, 0,1)*np.where(cnbh > 0, 0, 1)

name='wdept'
# "F:\毕设论文\out_daily\wdept.nc"
path0=r'G:\keshixu_wdept\421550_best'
ncSet=np.array(nc.Dataset(path0+'/'+name+'.nc')["wdept"])
# ncSet*=err
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
    err9999=np.where(ncSet[i]==-9999)
    output=ncSet[i]*err
    # output[s1==0]=output[s1==0]*0.5
    output[err9999]=-9999
    craetTif(tifname, im_width , im_height, 1, im_geotrans, im_proj, output,-9999)
