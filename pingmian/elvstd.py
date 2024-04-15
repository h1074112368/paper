from osgeo import gdal, ogr, gdalconst
import xarray as xr
import pandas as pd
import numpy as np
import os


# 为便于后续坡度计算，需要在原图像的周围添加一圈数值
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
    # wuding_ele = np.zeros_like(wuding, dtype='float32')
    # wuding_ele[np.where(wuding > 0)] = 5
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

    # laea_attr=dict()
    # time_data=pd.date_range('2015-01-01 00:00:00','2015-02-01 00:00:00',freq="H")

    # inp=flow_f(im_data)
    # inp=10-inp
    # inp[np.where(im_data!=255)]=1
    # inp=im_data.astype('float32')-np.min(im_data)
    # inp=AddRound(im_data)
    def std(inp):
        out=np.zeros([inp.shape[0]-2,inp.shape[1]-2])
        for i in range(1,out.shape[0]+1):
            for j in range(1,out.shape[1]+1):
                out[i-1,j-1]=np.std(inp[i-1:i+2,j-1:j+2])
        return out
    err=np.where(im_data==-9999)
    im_data=AddRound(im_data)
    inp=im_data
    inp[err] = 0
    inp=std(inp)
    inp[err]=-9999
    ds = xr.Dataset({
        # 'x': (['x'], im_lon, lon_attr),
        # 'y': (['y'], im_lat, lat_attr),
        #
        # 'value': (['y', 'x'], inp, t_attr)},)
        'x': (['x'], im_lon, lon_attr),
        'y': (['y'], im_lat, lat_attr),

        'value': (['y', 'x'], inp, t_attr),},
    )
    return ds

import  netCDF4 as nc

from fangshanduozaizhong import readTifArray
# path_wd_re=r"D:\tensorflow\lisflood\wuding_re.tif"
# wuding=readTifArray(path_wd_re)
day_nc = tiff2nc(r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m.tif")
path=r'D:\tensorflow\lisflood\shanghai\maps'
day_nc.to_netcdf(path+'/elvstd.nc',encoding={
            'value':{
            '_FillValue':-9999}
        })
# day_nc.variables["time"].calendar
# print()
# day_nc.to_netcdf('test.nc')
