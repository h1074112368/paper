from osgeo import gdal, ogr, gdalconst
import xarray as xr
import pandas as pd
import numpy as np
import os
def AddRound(npgrid):
    nx, ny = npgrid.shape[0], npgrid.shape[1]  # ny:行数，nx:列数;此处注意顺序
    # np.zeros()返回来一个给定形状和类型的用0填充的数组；
    zbc = np.zeros((nx + 2, ny + 2))
    # 填充原数据数组
    zbc[1:-1, 1:-1] = npgrid

    # 四边填充数据
    zbc[0, 1:-1] = npgrid[0, :]  # 上边；0行，所有列；
    zbc[-1, 1:-1] = npgrid[-1, :]  # 下边；最后一行，所有列；
    zbc[1:-1, 0] = npgrid[:, 0]  # 左边；所有行，0列。
    zbc[1:-1, -1] = npgrid[:, -1]  # 右边；所有行，最后一列

    # 填充剩下四个角点值
    zbc[0, 0] = npgrid[0, 0]
    zbc[0, -1] = npgrid[0, -1]
    zbc[-1, 0] = npgrid[-1, 0]
    zbc[-1, -1] = npgrid[-1, 0]

    return zbc
def tiff2nc(path):
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

    """
    GetRasterBand(bandNum)，选择要读取的波段数，bandNum 从 1 开始
    ReadAsArray(xoff, yoff, xsize, ysize)，一般就按照下面这么写，偏移量都是 0 ，返回 ndarray 数组
    """
    # wuding_ele=np.zeros_like(wuding,dtype='float32')
    # wuding_ele[np.where(wuding>0)]=5
    im_data = data.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=im_width, win_ysize=im_height)#+wuding_ele
    # 根据im_proj得到图像的经纬度信息
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]

    t_attr = dict(
        long_name='long name of variable',
        units='m',
        esri_pe_string =im_proj )
    lat_attr = dict(
        long_name='y', units='degrees_north')
    lon_attr = dict(
        long_name='x', units='degrees_east')
    time_attr=dict(
        # calendar='proleptic_gregorian',
        standard_name="time",
        # units=""
    )

    # demgrid = AddRound(im_data)

    def flow_f(data):
        tmp=[]
        tmp.append(np.where((data<256)&(data>=128),9,0))
        # tmp.append(np.where((data < 128)&(data >= 64), 8, 0))
        tmp.append(np.where((data < 128) & (data >= 64), 8, 0))
        tmp.append(np.where((data < 64) & (data >= 32), 7, 0))
        tmp.append(np.where((data < 32) & (data >= 16), 4, 0))
        tmp.append(np.where((data < 16) & (data >= 8), 1, 0))
        tmp.append(np.where((data < 8) & (data >= 4), 2, 0))
        tmp.append(np.where((data < 4) & (data >= 2), 3, 0))
        tmp.append(np.where((data < 2) & (data >= 1), 6, 0))
        # data[data==255]=5
        # data[data >= 1] = 6
        # data[data >= 2] = 3
        # data[data >= 4] = 2
        # data[data >= 8] = 1
        # data[data >= 16] = 4
        # data[data >= 32] = 7
        # data[data >= 64] = 8
        # data[data >= 128] = 9
        # data[data == 255] =9
        # flow =np.zeros_like(data,dtype='int')
        # flow+=np.where(data==255,5,0)
        # for i in range(data.shape[0]):
        #   for j in range(data.shape[1]):
        #       if(data[i,j])

        return np.sum(tmp,axis=0)
    # laea_attr=dict()
    # time_data=pd.date_range('2015-01-01 00:00:00','2015-02-01 00:00:00',freq="H")

    inp=flow_f(im_data)
    inp[inp==0]=5
    # inp[0,0]=255
    # inp=10-inp
    # inp[np.where(im_data!=255)]=1
    # im_data=im_data.astype('float32')
    ds = xr.Dataset({
         'x': (['x'], im_lon, lon_attr),
        'y': (['y'], im_lat, lat_attr),

        'value': (['y', 'x'], inp, t_attr),},
       )
    return ds

import  netCDF4 as nc
from fangshanduozaizhong import readTifArray
# path_wd_re=r"D:\tensorflow\lisflood\wuding_re.tif"
# wuding=readTifArray(path_wd_re)
day_nc = tiff2nc(r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m_tianwa_ldd.tif")
path=r'D:\tensorflow\lisflood\shanghai\maps'
day_nc.to_netcdf(path+'/ldd_tw.nc')
# day_nc.variables["time"].calendar
# print()
# day_nc.to_netcdf('test.nc')
