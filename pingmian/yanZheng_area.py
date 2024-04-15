import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
from osgeo import gdal, ogr, gdalconst
import time
dw=0.6
# for i in range(69,71):
for i in range(0, 6):
    time.sleep(1)
    cnbh=readTifArray(r"F:\bsPaper\re_CNBH10m\CNBH10M_hebing1.tif")
    s1=readTifArray(r"F:\bsPaper\re_s1flood_shanghai.tif")
    lf = readTifArray(fr"G:\keshixu_wdept\421550_best/wdept/wdept_00{i}.tif")
    lf1 = readTifArray(f"F:/bsPaper/out_hour/wdept/wdept_071.tif")
    # lf = readTifArray(f"G:/out/daily/wdept/wdept_00{i}.tif")
    # lf=readTifArray(f"F:/bsPaper/out_hour/wdept/wdept_0{i}.tif")
    ul=readTifArray(r"D:\tensorflow\lisflood\dem_shanghai_25m\n51_30_2020lc030_land.tif")
    data = gdal.Open(r"F:\bsPaper\re_s1flood_shanghai.tif")
    im_proj = data.GetProjection()  # 获取地理信息
    im_width = data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
    im_height = data.RasterYSize  # 获取高度，数组第一维，上下方向元素长度，代表纬度范围
    im_bands = data.RasterCount  # 波段数
    im_geotrans = data.GetGeoTransform()  # 获取仿射矩阵，含有 6 个元素的元组
    lf[np.where(ul==60)]=0
    craetTif('wdept.tif', im_width, im_height, im_bands, im_geotrans, im_proj, lf,-9999)
    a1=0
    a2=0
    err=np.where(lf==-9999,0,1)*np.where(s1==20,0,1)*np.where(s1==10,0,1)*np.where(ul==60,0,1)#*np.where(s1==1,0,1)*np.where(s1==2,0,1)*np.where(s1==3,0,1)
    s1=s1*err*np.where(s1==0,0,1)
    lf=lf*err
    lisflood=np.where(lf>dw,1,0)
    s1flood=np.where(s1>0,1,0)



    # craetTif('lisflood.tif', im_width, im_height, im_bands, im_geotrans, im_proj, np.where(s1>0,1,0)*np.where(lf>0.1,1,0),-9999)
    craetTif('lisflood.tif', im_width, im_height, im_bands, im_geotrans, im_proj,
             np.where(lf>dw,1,0), -9999)
    craetTif('s1flood.tif', im_width, im_height, im_bands, im_geotrans, im_proj, np.where(s1>0,1,0),-9999)
    yanzheng=np.zeros_like(s1flood)
    yanzheng[lisflood==1]=1
    # yanzheng[s1flood==1]=2
    # fanwei=40
    # for ii in range(lisflood.shape[0]):
    #     for j in range(lisflood.shape[1]):
    #         try:
    #             if(s1flood[ii,j]==1and np.max(lisflood[ii-fanwei:ii+fanwei,j-fanwei:j+fanwei])==1):
    #                 yanzheng[ii,j]=3
    #         except:
    #             print(ii,j)
    yanzheng[np.where(cnbh>0)]=0
    craetTif(fr'F:\bsPaper\yanzheng_area{i}.tif', im_width, im_height, im_bands, im_geotrans, im_proj, yanzheng, -9999)
    print(i,np.sum(yanzheng==3),np.sum(yanzheng==2)+np.sum(yanzheng==3),np.sum(yanzheng==1)+np.sum(yanzheng==3))
    # for i in range(s1.shape[0]):
    #     for j in range(s1.shape[1]):
    #         if s1[i,j]!=-9999 and lf[i,j]!=-9999:
    #             a1+=1;
    #             if s1[i,j]>0 and lf[i,j]>0:
    #                 a2+1
    # print(a1,a2)