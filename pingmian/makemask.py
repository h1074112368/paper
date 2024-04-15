from osgeo import gdal, ogr, gdalconst
import xarray as xr
import pandas as pd
import numpy as np
from fangshanduozaizhong import readTifArray
import os
def tiff2nc(path1,path2,path3):
    data = gdal.Open(path1)
    data2=gdal.Open(path2)
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
    im_data2=data2.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=im_width, win_ysize=im_height)
    # 根据im_proj得到图像的经纬度信息
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]
    proj4 = im_proj.split(',')[38].split('"')[1]
    t_attr = dict(
        long_name='long name of variable',
        units='celcius',
        grid_mapping='Pseudo-Mercator',
        esri_pe_string =im_proj,#'PROJCS["WGS_1984_Web_Mercator_Auxiliary_Sphere",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Mercator_Auxiliary_Sphere"],PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",0.0],PARAMETER["Auxiliary_Sphere_Type",0.0],UNIT["Meter",1.0]]'
        )
    lat_attr = dict(
        long_name='y', units='Meter')
    lon_attr = dict(
        long_name='x', units='Meter')
    time_attr=dict(
        # calendar='proleptic_gregorian',
        standard_name="time",
        # units=""
    )
    laea_attr=dict(
        grid_mapping_name= 'Pseudo-Mercator',
        false_easting= 0,
        false_northing= 0,
        # longitude_of_projection_origin= 10.0,
        # latitude_of_projection_origin= 52.0,
        # semi_major_axis= 6378137.0,
        # inverse_flattening= 298.257223563,
        proj4_params=proj4,
        EPSG_code='EPSG:3857',
    )
    im_data3=readTifArray(path3)
    inp=np.where((im_data!=-9999)& (im_data2!=255)&(im_data3!=-32768),1,0)
    # laea_attr=dict()
    # time_data=pd.date_range('2015-01-01 00:00:00','2015-02-01 00:00:00',freq="H")
    # inp=np.zeros([im_data.shape[0],im_data.shape[1]],dtype='int8')
    # inp[np.where(im_data!=255)]=1
    ds = xr.Dataset({
         'x': (['x'], im_lon, lon_attr),
        'y': (['y'], im_lat, lat_attr),

        'value': (['y', 'x'], inp, t_attr),
        # 'laea': (['laea'],[],laea_attr)
    }

       )
    return ds

import  netCDF4 as nc
path_land=r"D:\tensorflow\lisflood\data_25m\landUseAddWudingAddwater.tif"
path_dem=r"D:\tensorflow\lisflood\data_25m\demAddWuding.tif"
thickness=r"G:\data_FLOOD\soil\thickness.tif"
day_nc = tiff2nc(path_dem,path_land,thickness)
path=r'D:\tensorflow\lisflood\shanghai\maps/'
day_nc.to_netcdf(path+'/area.nc')
# day_nc.variables["time"].calendar
# print()
# day_nc.to_netcdf('test.nc')
