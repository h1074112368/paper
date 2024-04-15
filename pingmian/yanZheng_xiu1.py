import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
from osgeo import gdal, ogr, gdalconst
import time
dw=0.6
# for i in range(69,71):
for i in range(2, 3):

    time.sleep(1)
    cnbh=readTifArray(r"F:\bsPaper\re_CNBH10m\CNBH10M_hebing1.tif")
    # s1=readTifArray(r"F:\bsPaper\re_zvvall105.tif")
    s1 = readTifArray(r"F:\bsPaper\re_s1flood_shanghai.tif")
    lf = readTifArray(rf"G:\keshixu_wdept\422102_default\wdept\wdept_00{i}.tif")
    # lf = readTifArray(f"F:/bsPaper/out_daily/wdept/wdept_00{i}.tif")
    # lf1 = readTifArray(f"F:/bsPaper/out_hour/wdept/wdept_071.tif")
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
    craetTif('shuiti.tif', im_width, im_height, im_bands, im_geotrans, im_proj, np.where(s1==20,0,1)*np.where(s1==10,0,1), -9999)
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
    yanzheng[s1flood==1]=2
    fanwei=10
    jiange=14
    s1lfi=0
    s1i=0
    lfi=0
    for ii in range(0,lisflood.shape[0],jiange):
        for j in range(0,lisflood.shape[1],jiange):

            if (1 in s1flood[ii:ii+jiange,j:j+jiange])and (1 in lisflood[ii:ii+jiange,j:j+jiange]):


                    yanzheng[ii:ii+jiange,j:j+jiange][(s1flood[ii:ii+jiange,j:j+jiange]==1)|(lisflood[ii:ii+jiange,j:j+jiange]==1)]=3
                    if 80 not in ul[ii:ii+jiange,j:j+jiange]:
                        s1lfi+=1
                    # yanzheng[ii:ii + jiange, j:j + jiange] = 3
                    # yanzheng[ii:ii + jiange, j:j + jiange] = 3

            else:
                if np.sum(cnbh[ii:ii + jiange, j:j + jiange]) > 0:
                    continue
                elif (np.sum(s1flood[ii:ii+jiange,j:j+jiange])>0):
                    if 80 not in ul[ii:ii + jiange, j:j + jiange]:
                        s1i+=1
                #     yanzheng[ii:ii + jiange, j:j + jiange] = 2
                elif(np.sum(lisflood[ii:ii+jiange,j:j+jiange])>0):
                    if 80 not in ul[ii:ii + jiange, j:j + jiange]:
                        lfi+=1
                #     yanzheng[ii:ii + jiange, j:j + jiange] = 1
            # except:
            #     print(ii,j)
    # yanzheng[np.where(cnbh>0)]=0
    # yanzheng[np.where(ul==80)]=0
    craetTif('yanzheng1_xiu1_default.tif', im_width, im_height, im_bands, im_geotrans, im_proj, yanzheng, -9999)
    path_w_re = r"F:\bsPaper\re_quhua1.tif"
    cankao = r"F:\bsPaper\yanzheng_area2.tif"
    quhuaa = readTifArray(path_w_re)
    print(i, np.sum(yanzheng== 3),
          (np.sum(yanzheng== 2) + np.sum(yanzheng== 3)),
          (np.sum(yanzheng == 1) + np.sum(yanzheng == 3)))
    # for k in range(16):
    #     print(i,np.sum(yanzheng[quhuaa==k]==3),(np.sum(yanzheng[quhuaa==k]==2)+np.sum(yanzheng[quhuaa==k]==3)),(np.sum(yanzheng[quhuaa==k]==1)+np.sum(yanzheng[quhuaa==k]==3)))
    # for i in range(s1.shape[0]):
    #     for j in range(s1.shape[1]):
    #         if s1[i,j]!=-9999 and lf[i,j]!=-9999:
    #             a1+=1;
    #             if s1[i,j]>0 and lf[i,j]>0:
    #                 a2+1
    # print(a1,a2)