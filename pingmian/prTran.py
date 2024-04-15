from osgeo import gdal, ogr, gdalconst
import  netCDF4 as nc
import os
path_w=r"D:\tensorflow\lisflood\data_2m\上海市_DEM_高程.tif"
src_data = gdal.Open(path_w)
# path_w_re=r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m_re.tif"

cankao=r"D:\tensorflow\lisflood\data_25m\dem_shanghai_25m.tif"
# ncT=nc.Dataset(path_w_re)
can_data=gdal.Open(cankao)
im_geotrans = can_data.GetGeoTransform()  #
im_width = can_data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
im_height = can_data.RasterYSize



minX,minY, maxX, maxY=im_geotrans[0],im_geotrans[3]+im_height*im_geotrans[5],im_geotrans[0]+im_geotrans[1]*im_width,im_geotrans[3]
lists=os.listdir(r'F:\毕设论文\pr')
for file in lists:
    if file.endswith('tif'):
        path_w = r'F:\毕设论文\pr'+'/'+file
        path_w_re = r'D:\tensorflow\lisflood\PreSet'+'/re_'+file
        # src_data=gdal.Open(path_w)
        gdal.Warp(path_w_re,path_w,width=im_width,height=im_height,outputBounds=[minX,minY, maxX, maxY],resampleAlg=0,dstSRS=can_data.GetProjection(),srcSRS=src_data.GetProjection())
# gdal.Warp(path_w_re,path_w,format='netCDF',width=im_width,height=im_height,outputBounds=[minX,minY, maxX, maxY],resampleAlg=0)