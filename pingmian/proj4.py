import pycrs
from osgeo import gdal, ogr, gdalconst
import  netCDF4 as nc
import json
# path_w=r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m.tif"
# path_w_re=r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m_re.tif"
path_w=r"D:\tensorflow\lisflood\wuding_2m.tif"
path_w_re=r"D:\tensorflow\lisflood\data\wuding_2m_re.tif"
cankao=r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m.tif"
# ncT=nc.Dataset(path_w_re)
can_data=gdal.Open(cankao)
im_geotrans = can_data.GetGeoTransform()  #
im_proj=can_data.GetProjection()
im_width = can_data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
im_height = can_data.RasterYSize
src_data=gdal.Open(path_w)
# wkt text可以直接从ArcGIS导出
wkt_text = im_proj
srs_proj40=im_proj.split(',')
srs_proj4=srs_proj40[38].split('"')[1]
srs_proj4 = pycrs.parse.from_esri_wkt(wkt_text).to_proj4()
print(srs_proj4)