from osgeo import gdal
import os
def resample(path,name):
    repath="/mnt/data2/hyc/tifdata/soil_texture/re/"
    # path_w=r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m.tif"
    # path_w_re=r"D:\tensorflow\lisflood\dem_shanghai_25m\dem_shanghai_25m_re.tif"
    # path_w=r"F:\毕设论文\N51_30_2020LC030\n51_30_2020lc030.tif"
    # path_w_re=r"D:\tensorflow\lisflood\dem_shanghai_25m\n51_30_2020lc030_land.tif"
    path_w=os.path.join(path,name)
    path_w_re=repath+name
    cankao="/home/hyc/flaskFiles/floodBase/landUseAddWudingAddwater.tif"
    # ncT=nc.Dataset(path_w_re)
    can_data=gdal.Open(cankao)
    im_geotrans = can_data.GetGeoTransform()  #
    im_width = can_data.RasterXSize  # 获取宽度，数组第二维，左右方向元素长度，代表经度范围
    im_height = can_data.RasterYSize
    src_data=gdal.Open(path_w)

    minX,minY, maxX, maxY=im_geotrans[0],im_geotrans[3]+im_height*im_geotrans[5],im_geotrans[0]+im_geotrans[1]*im_width,im_geotrans[3]
    gdal.Warp(path_w_re,path_w,width=im_width,height=im_height,outputBounds=[minX,minY, maxX, maxY],resampleAlg=0,dstSRS=can_data.GetProjection(),srcSRS=src_data.GetProjection())
input_path = "/mnt/data2/hyc/tifdata/soil_texture/chinasoil_90m"
error=[]
for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith(".tif")and file=="CNBH10M_hebing1.tif":
                    try:
                        resample(root, file)
                    except:
                        error.append(os.path.join(root,file))
                        print("error",root, file)
                    print(root,file)
print(error)                    
