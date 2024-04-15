import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
from osgeo import gdal, ogr, gdalconst
import time
data=np.zeros([5,6])
data1=np.zeros([5,6])
path_w_re=r"F:\bsPaper\re_quhua1.tif"
cankao=r"F:\bsPaper\yanzheng_area2.tif"
ul=readTifArray(r"D:\tensorflow\lisflood\dem_shanghai_25m\n51_30_2020lc030_land.tif")
quhuaa=readTifArray(path_w_re)
value=[10,20,30,50,80]
cnbh=readTifArray(r"F:\bsPaper\re_CNBH10m\CNBH10M_hebing1.tif")
for i in range(6):
    jishui=readTifArray(fr"F:\bsPaper\yanzheng_area{i}.tif")
    jj=0
    for j in value:
        data[jj,i]=np.sum((ul==j)&(jishui==1))*25*25
        data1[jj, i] = np.sum((ul == j) & (jishui == 1)) /np.sum((ul==j)&((cnbh<=0)|(np.isnan(cnbh))))
        jj+=1
import pandas as pd
index=[
"耕地",
"林地",
"草地",
"湿地",
"人造地表",
]
pd.DataFrame(data,index=index).to_excel(fr"F:\bsPaper\sum_lu.xlsx")
pd.DataFrame(data1,index=index).to_excel(fr"F:\bsPaper\sum_lu_per.xlsx")
