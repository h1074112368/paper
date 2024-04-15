import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
from osgeo import gdal, ogr, gdalconst
cankao=r"D:\tensorflow\lisflood\data_25m\dem_shanghai_25m_mask_re.tif"
# ncT=nc.Dataset(path_w_re)
can_data=gdal.Open(cankao)
im_geotrans = can_data.GetGeoTransform()  #
im_width = can_data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
im_height = can_data.RasterYSize
path=r"D:\tensorflow\lisflood\data_25m"
wuding=readTifArray(path+'/jianzhu_25m_re.tif')
water=readTifArray(path+'/shuiti_25m_re.tif')
land=readTifArray(path+'/landuse_25m.tif')
# dem=readTifArray(path+'/上海市_DEM_高程.tif')
land[np.where(wuding>0)]=80
land[np.where( water>0)]=60
craetTif( path+'/landUseAddWudingAddwater.tif', im_width, im_height, 1, im_geotrans, can_data.GetProjection(), land,255)
# dem=r"D:\tensorflow\lisflood\data\上海市_DEM_高程.tif"
