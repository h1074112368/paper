import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
from osgeo import gdal, ogr, gdalconst
cankao=r"G:\data_FLOOD\soil\resamle_soil\clay_0_5cm_mean.tif"
# ncT=nc.Dataset(path_w_re)
can_data=gdal.Open(cankao)
cankao_data=readTifArray(cankao)
im_geotrans = can_data.GetGeoTransform()  #
im_width = can_data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
im_height = can_data.RasterYSize
path=r"D:\tensorflow\lisflood\data_25m"
wuding=readTifArray(r"F:\bsPaper\re_CNBH10m\CNBH10M_hebing1.tif")
dem=readTifArray(r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m.tif")

wuding[np.isnan(wuding)]=0
res=np.ones_like(dem)*0
res=res+wuding+dem
# res[np.where( cankao_data==np.min(cankao_data))]=-9999
# res[np.where( wuding==0)]=-9999
craetTif( path+'/demAddcnbh.tif', im_width, im_height, 1, im_geotrans, can_data.GetProjection(), res,-9999)
# dem=r"D:\tensorflow\lisflood\data\上海市_DEM_高程.tif"
