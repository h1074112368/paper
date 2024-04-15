import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
from osgeo import gdal, ogr, gdalconst
import time
import  netCDF4 as nc
data=np.zeros([16,6])
path_w_re=r"F:\bsPaper\re_quhua1.tif"
cankao=r"F:\bsPaper\yanzheng_area2.tif"
pr=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\meteo\daily_pr.nc")["value"])
quhuaa=readTifArray(path_w_re)
for i in range(6):
    jishui=readTifArray(fr"F:\bsPaper\yanzheng_area{i}.tif")
    for j in range(16):
        data[j,i]=np.max(pr[i][quhuaa==j])
import pandas as pd
index=[
"崇明区",
"奉贤区",
"虹口区",
"黄浦区",
"嘉定区",
"金山区",
"静安区",
"闵行区",
"浦东新区",
"青浦区",
"松江区",
"徐汇区",
"杨浦区",
"长宁区",
"普陀区",
"宝山区",
]
pd.DataFrame(data,index=index).to_excel(fr"F:\bsPaper\rain_max_quhua.xlsx")
