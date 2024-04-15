from osgeo import gdal
import os
import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
import os
from osgeo import gdal
path="/mnt/data2/hyc/tifdata/soil_texture/re"

def get_tif_para(path):
    tif = gdal.Open(path)
    projection = tif.GetProjection()
    transform = tif.GetGeoTransform()
    maxcol = tif.RasterXSize
    maxrow = tif.RasterYSize
    return projection, transform, maxcol, maxrow
def cal_layer2(value):
    res=np.ones_like(value)*-9999
    for i in range(len(value)):
        
        for j in range(len(value[0])):
            if value[i][j]==-9999 :
                continue
            if value[i][j]<30 :
                res[i,j]=(value[i,j]-5)/2
            else:
                res[i,j]=max(5,min(45,value[i,j]-30-5))
    return res
def cal_layer3(value,value2):
    res=np.ones_like(value)*-9999
    for i in range(len(value)):        
        for j in range(len(value[0])):
            if value[i][j]==-9999 :
                continue
            res[i,j]=value[i,j]-value2[i,j]-5
    return res
def cal_layer1(value):
    res=np.ones_like(value)
    res*=5
    res[value==-9999]=-9999
    return res

projection, transform, maxcol, maxrow=get_tif_para("/mnt/data2/hyc/tifdata/soil_texture/re/bd05.tif")
value=readTifArray("/mnt/data2/hyc/tifdata/soil_texture/re/thickness.tif")
value[value<0]=-9999
path_out="/mnt/data2/hyc/tifdata/soil_texture"
craetTif(path_out+f"/thickness1.tif", maxcol, maxrow, 1, transform, projection, cal_layer1(value),-9999)
value2=cal_layer2(value)
craetTif(path_out+f"/thickness2.tif", maxcol, maxrow, 1, transform, projection, value2,-9999)

craetTif(path_out+f"/thickness3.tif", maxcol, maxrow, 1, transform, projection, cal_layer3(value,value2),-9999)



