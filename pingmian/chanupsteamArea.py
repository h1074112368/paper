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


# 计算xy方向的梯度
def Cacdxdy(npgrid, sizex, sizey):
    nx, ny = npgrid.shape
    s_dx = np.zeros((nx, ny))
    s_dy = np.zeros((nx, ny))
    a_dx = np.zeros((nx, ny))
    a_dy = np.zeros((nx, ny))


    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            s_dx[i, j] = (8 * sizex) / ((npgrid[i - 1, j + 1] + 2 * npgrid[i, j + 1] + npgrid[i + 1, j + 1]) - (
                    npgrid[i - 1, j - 1] + 2 * npgrid[i, j - 1] + npgrid[i + 1, j - 1]) + np.finfo(np.float64).eps)
            s_dy[i, j] = (8 * sizex) / ((npgrid[i + 1, j - 1] + 2 * npgrid[i + 1, j] + npgrid[i + 1, j + 1]) - (
                    npgrid[i - 1, j - 1] + 2 * npgrid[i - 1, j] + npgrid[i - 1, j + 1]) + np.finfo(np.float64).eps)

    a_dx = s_dx * sizex
    a_dy = s_dy * sizey
    # 保留原数据区域的梯度值
    s_dx = s_dx[1:-1, 1:-1]
    s_dy = s_dy[1:-1, 1:-1]
    a_dx = a_dx[1:-1, 1:-1]
    a_dy = a_dy[1:-1, 1:-1]
    # np.savetxt(r"D:\ProfessionalProfile\DEMdata\slopeAspectPython0322\1dxdy.csv",dx,delimiter=",")

    return s_dx, s_dy, a_dx, a_dy


# 计算坡度/坡向
def CacSlopAsp(s_dx, s_dy, a_dx, a_dy):
    import math
    # 坡度
    slope = (np.arctan(np.sqrt(s_dx * s_dx + s_dy * s_dy))) * 180 / math.pi  # 转换成°

    # 坡向
    # #出错：TypeError: only size-1 arrays can be converted to Python scalars
    # a2 = math.atan2(a_dy,-a_dx)*180/math.pi
    a = np.zeros((a_dy.shape[0], a_dy.shape[1]))
    for i in range(0, a_dx.shape[0]):
        for j in range(0, a_dx.shape[1]):
            a[i, j] = math.atan2(a_dy[i, j], -a_dx[i, j]) * 180 / math.pi

    # 输出
    aspect = a
    # # 坡向值将根据以下规则转换为罗盘方向值（0 到 360 度）：
    # x, y = a.shape[0], a.shape[1]
    # for m in range(0, x):
    #     for n in range(0, y):
    #         if a[m, n] < 0:
    #             aspect[m, n] = 90 - a[m, n]
    #         elif a[m, n] > 90:
    #             aspect[m, n] = 360.0 - a[m, n] + 90.0
    #         else:
    #             aspect[m, n] = 90.0 - a[m, n]

    return slope, aspect


def tiff2nc(path, chan, ldd, area):
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

    # demgrid = AddRound(im_data)
    # def reflow_f(data):
    #     flow=np.zeros([data.shape[0]-2,data.shape[1]-2],dtype=int)
    #     # flow=np.zeros_like(data,dtype=int)
    #     for i in range(flow.shape[0]):
    #         for j in range(flow.shape[1]):
    #             c=data[i+1,j+1]
    #             # temp=np.zeros_like(data[i-1:i+2,j-1:j+2])
    #             temp=data[i:i+3,j:j+3]-c
    #             maxt=np.max(temp)
    #             if maxt==0:
    #                 flow[i,j]=5
    #             else:
    #                 maxidx=np.argmax(temp)+1
    #                 flow[i,j]=maxidx
    #
    #     return flow
    # reldd = reflow_f(demgrid)

    def upsteamArea(dem, chan, ldd, area):
        out = np.zeros_like(chan, dtype='float32')
        jinji = np.zeros_like(chan, dtype=int)
        for i in range(chan.shape[0]):
            for j in range(chan.shape[1]):
                if chan[i, j] == 1 and jinji[i, j] == 0:
                    x, y = i, j;
                    a = 0.0001
                    jinji1 = set()
                    while x>=0 and x<chan.shape[0] and y>=0 and y<chan.shape[1] and chan[x, y] == 1 and jinji[x, y] == 0:
                        # print(tuple((x, y)))
                        if (x, y) in jinji1:
                            break
                        # print(tuple((x, y)))
                        jinji1.add((x, y))
                        if chan[x, y] == 1:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            x += 1
                            y -= 1
                        elif chan[x, y] == 2:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            x += 1
                            # y -= 1
                        elif chan[x, y] == 3:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            x += 1
                            y += 1
                        elif chan[x, y] == 4:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            # x -= 1
                            y -= 1
                        elif chan[x, y] == 5:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            break
                            # x -= 1
                            # y -= 1
                        elif chan[x, y] == 6:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            # x -= 1
                            y += 1
                        elif chan[x, y] == 7:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            x -= 1
                            y -= 1
                        elif chan[x, y] == 8:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            x -= 1
                            # y -= 1
                        elif chan[x, y] == 9:
                            jinji[x, y] = 1
                            out[x, y] = a
                            a += area[x, y]
                            x -= 1
                            y += 1
                    if x>=0 and x<chan.shape[0] and y>=0 and y<chan.shape[1] and jinji[x, y] == 1 and chan[x,y]!=5:
                        jinji1 .clear()
                        while x>=0 and x<chan.shape[0] and y>=0 and y<chan.shape[1] and chan[x, y] == 1:
                            if (x, y) in jinji1 :
                                break
                            jinji1.add((x, y))
                            if chan[x, y] == 1:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                x += 1
                                y -= 1
                            elif chan[x, y] == 2:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                x += 1
                                # y -= 1
                            elif chan[x, y] == 3:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                x += 1
                                y += 1
                            elif chan[x, y] == 4:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                # x -= 1
                                y -= 1
                            elif chan[x, y] == 5:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                break
                                # x -= 1
                                # y -= 1
                            elif chan[x, y] == 6:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                # x -= 1
                                y += 1
                            elif chan[x, y] == 7:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                x -= 1
                                y -= 1
                            elif chan[x, y] == 8:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                x -= 1
                                # y -= 1
                            elif chan[x, y] == 9:
                                jinji[x, y] = 1
                                out[x, y] += a
                                # a += area[x, y]
                                x -= 1
                                y += 1

        return out

    # upA = upsteamArea(im_data, chan, ldd, area)
    # out = upA * 0.0032
    # for i in range(upA.shape[0],upA.shape[1]):
    #     if(chan[i,j]==1)

    # def slopeCal(dem,chan,ldd,area):
    #     out=np.zeros_like(chan,dtype='float32')
    #     for i in range(chan.shape[0]):
    #         for j in range(chan.shape[1]):
    #             if chan[i, j] == 1:
    #                 if chan[i,j]==1 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1-1,j+1-1])/length[i,j]
    #                 if chan[i,j]==2 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1-1,j+1])/length[i,j]
    #                 if chan[i,j]==3 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1-1,j+1+1])/length[i,j]
    #                 if chan[i,j]==4 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1,j+1-1])/length[i,j]
    #                 if chan[i,j]==5 :
    #                     out[i,j]=0
    #                 if chan[i,j]==6 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1,j+1+1])/length[i,j]
    #                 if chan[i,j]==7 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1+1,j+1-1])/length[i,j]
    #                 if chan[i,j]==8 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1+1,j+1])/length[i,j]
    #                 if chan[i,j]==9 :
    #                     out[i,j]=(dem[i+1,j+1]-dem[i+1+1,j+1+1])/length[i,j]
    #     return out
    # 梯度计算
    # dx1, dy1, dx2, dy2 = Cacdxdy(demgrid, 2, 2)
    # # 坡度、坡向计算
    # inp, aspect = CacSlopAsp(dx1, dy1, dx2, dy2)
    # inp = slopeCal(demgrid, chan, reldd, area)
    t_attr = dict(
        long_name='long name of variable',
        units='m',
        esri_pe_string=im_proj)
    lat_attr = dict(
        long_name='y', units='degrees_north')
    lon_attr = dict(
        long_name='x', units='degrees_east')
    time_attr = dict(
        # calendar='proleptic_gregorian',
        standard_name="time",
        # units=""
    )

    # laea_attr=dict()
    # time_data=pd.date_range('2015-01-01 00:00:00','2015-02-01 00:00:00',freq="H")

    # inp=flow_f(im_data)
    # inp=10-inp
    # inp[np.where(im_data!=255)]=1
    # def gra(data):

    # inp=im_data.astype('float32')-np.min(im_data)
    # inp=im_data*area[0,0]
    inp=im_data*1000*1000
    inp[np.where(inp < 0)] = -9999
    ds = xr.Dataset({
        'x': (['x'], im_lon, lon_attr),
        'y': (['y'], im_lat, lat_attr),

        'value': (['y', 'x'], inp, t_attr),},
    )
    return ds


import netCDF4 as nc

path = r'D:\tensorflow\lisflood\shanghai\maps'
chan = np.array(nc.Dataset(path + '/chan.nc')['value'])
ldd = np.array(nc.Dataset(path + '/ldd.nc')['value'])
area = np.array(nc.Dataset(path + '/pixel.nc')['value'])
day_nc = tiff2nc(r"F:\毕设论文\re_n30e120_upa.tif", chan, ldd, area)

day_nc.to_netcdf(path + '/upsteamArea.nc',encoding={
            'value':{
            '_FillValue':-9999}
        })
# day_nc.variables["time"].calendar
# print()
# day_nc.to_netcdf('test.nc')
