from osgeo import gdal, ogr, gdalconst
import  netCDF4 as nc
path_w=r"D:\tensorflow\lisflood\n51_30_2010lc030_Pro.tif"
path_w_re=r"D:\tensorflow\lisflood\landuse_re_clip.tif"
cankao=r"D:\tensorflow\3dTest\landuse_re_clip.tif"
# ncT=nc.Dataset(path_w_re)
can_data=gdal.Open(cankao)
im_geotrans = can_data.GetGeoTransform()  #
im_width = can_data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
im_height = can_data.RasterYSize
src_data=gdal.Open(path_w)

minX,minY, maxX, maxY=im_geotrans[0],im_geotrans[3]+im_height*im_geotrans[5],im_geotrans[0]+im_geotrans[1]*im_width,im_geotrans[3]
gdal.Warp(path_w_re,path_w,width=im_width,height=im_height,outputBounds=[minX,minY, maxX, maxY],resampleAlg=0,dstSRS=can_data.GetProjection(),srcSRS=src_data.GetProjection())
# gdal.Warp(path_w_re,path_w,format='netCDF',width=im_width,height=im_height,outputBounds=[minX,minY, maxX, maxY],resampleAlg=0)