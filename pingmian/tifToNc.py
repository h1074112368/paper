from osgeo import gdal, ogr, gdalconst
import xarray as xr
import pandas as pd
import numpy as np
import os
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
    im_data = data.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=im_width, win_ysize=im_height)
    # 根据im_proj得到图像的经纬度信息
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]

    t_attr = dict(
        long_name='long name of variable',
        units='the units of var',
        esri_pe_string =im_proj )
    lat_attr = dict(
        long_name='latitude', units='degrees_north')
    lon_attr = dict(
        long_name='longitude', units='degrees_east')
    time_attr=dict(
        # calendar='proleptic_gregorian',
        standard_name="time",
        # units=""
    )

    laea_attr=dict()
    time_data=pd.date_range('2015-01-01 00:00:00','2015-02-01 00:00:00',freq="H")
    inp=np.zeros([time_data.shape[0],im_data.shape[0],im_data.shape[1]],dtype='float32')
    ds = xr.Dataset({
        'e0': (['time', 'lat', 'lon'], inp, t_attr)},
        coords={'time': (['time'],time_data,time_attr),
                'lat': (['lat'], im_lat, lat_attr),
                'lon': (['lon'],  im_lon, lon_attr),
                # 'laea':(['laea'],[],im_proj)

                })
    return ds

import  netCDF4 as nc
# end_file=r"D:\lisflod\lisflood-code-master\tests\data\LF_ETRS89_UseCase\maps\mask.map"
# basename = end_file.replace('.map', '')
# state_file = '{}.nc'.format(basename)
# # if not os.path.exists(state_file):
# #     continue
# state_nc = nc.Dataset(state_file)
# nc.variables["time"].calendar
# gdal.Translate('test.nc',r"D:\tensorflow\3dTest\jishui_00.tif",format='netCDF')
# test=nc.Dataset(r"D:\lisflod\lisflood-code-master\tests\data\LF_ETRS89_UseCase\maps\mask.map")
test=nc.Dataset('test.nc')
# # nc.variables["time"].calendar
# test1=nc.Dataset(r"D:\out\dirrun.nc")
# test1.variables["time"].calendar
day_nc = tiff2nc(r"D:\tensorflow\3dTest\jishui_01.tif")
# day_nc.variables["time"].calendar
print()
day_nc.to_netcdf('test.nc')
