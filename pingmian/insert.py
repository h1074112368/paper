from osgeo import gdal, ogr, gdalconst
import  netCDF4 as nc
from fangshanduozaizhong import shpReadTable
def insertRaster(field,i,iminX,minY, maxX, maxY,im_proj,im_width,im_height):
    # startTime = ps.to_datetime(datetime.datetime.now())
    # print('开始插值：%s' % startTime)
    point_file = r"F:\bsPaper\yanhua_daily_pr1_mokatuo.shp"
    output_file =r"F:\bsPaper\yanhua_station_pr\Pre_"+"{:03d}".format(i)+".tif"

    opts = gdal.GridOptions(format="GTiff", outputType=gdal.GDT_Float32,
                            algorithm="invdist:power=3:smothing=0.0:radius=100000.0:max_points=12:min_points=1:nodata=-1",
                            zfield=field,
                            outputBounds=[minX,minY, maxX, maxY],
                            outputSRS=im_proj,
                            width=im_width,
                            height=im_height,
                            )

    gdal.Grid(destName=output_file, srcDS=point_file, options=opts)
    # endTime = ps.to_datetime(datetime.datetime.now())
    # useTime = endTime - startTime
    # print('插值完成！用时：%s s' % useTime.total_seconds())
table=shpReadTable(r"F:\bsPaper\yanhua_daily_pr1_mokatuo.shp")
from readShp import jiangshui
fields=table.columns[5:]
path_w=r"D:\tensorflow\lisflood\data_25m\demAddWuding.tif"
# path_w_re=r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m_re.tif"
cankao=r"D:\tensorflow\lisflood\data_25m\landUseAddWudingAddwater.tif"
# ncT=nc.Dataset(path_w_re)
can_data=gdal.Open(cankao)
im_geotrans = can_data.GetGeoTransform()  #
im_width = can_data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
im_height = can_data.RasterYSize
src_data=gdal.Open(path_w)
im_proj=can_data.GetProjection()

minX,minY, maxX, maxY=im_geotrans[0],im_geotrans[3]+im_height*im_geotrans[5],im_geotrans[0]+im_geotrans[1]*im_width,im_geotrans[3]
# gdal.Warp(path_w_re,path_w,width=im_width,height=im_height,outputBounds=[minX,minY, maxX, maxY],resampleAlg=0,dstSRS=can_data.GetProjection(),srcSRS=src_data.GetProjection())
for i in range(len(fields)):
    insertRaster(fields[i],i,minX,minY, maxX, maxY,im_proj,im_width,im_height)
