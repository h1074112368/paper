import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
import os
from osgeo import gdal
path="/mnt/data2/hyc/tifdata/soil_texture/re"
name="100200"
name2=["0_5","5_15","15_30","30_60","60_100","100_200"]
dept=[5,15,30,60,100]
chan=[5,10,15,30,40,100]
def get_tif_para(path):
    tif = gdal.Open(path)
    projection = tif.GetProjection()
    transform = tif.GetGeoTransform()
    maxcol = tif.RasterXSize
    maxrow = tif.RasterYSize
    return projection, transform, maxcol, maxrow

projection, transform, maxcol, maxrow=get_tif_para("/home/hyc/flaskFiles/floodBase/landUseAddWudingAddwater.tif")
path_out="/mnt/data2/hyc/tifdata/soil_texture"
thickness2=readTifArray(path_out+"/thickness2.tif")
thickness3=readTifArray(path_out+"/thickness3.tif")
Thetas=[]
Thetar=[]
Lamba=[]
Alpha=[]
Ksaturated=[]
ii=0
for i in name2:
    Thetas.append(readTifArray(path_out+"/Thetas"+i+".tif"))
    Thetar.append(readTifArray(path_out+"/Thetar"+i+".tif"))
    Lamba.append(readTifArray(path_out+"/Lamba"+i+".tif"))
    Alpha.append(readTifArray(path_out+"/Alpha"+i+".tif"))
    Ksaturated.append(readTifArray(path_out+"/Ksaturated"+i+".tif"))
    ii+=1
    



def layer3(thickness2):
    finsh=np.zeros_like(thickness2)
    leiji_sum=np.zeros_like(thickness2)
    Thetas_sum=np.zeros_like(thickness2)
    Thetar_sum=np.zeros_like(thickness2)
    Lamba_sum=np.zeros_like(thickness2)
    Alpha_sum=np.zeros_like(thickness2)
    Ksaturated_sum=np.zeros_like(thickness2)
    
    for ii in range(thickness2.shape[0]):
        for jj in range(thickness2.shape[1]):
            for kk in range(int(5+thickness2[ii,jj]),int(5+thickness2[ii,jj]+thickness3[ii,jj])):
                if kk<=dept[1]:
                    Thetas_sum[ii,jj]+=Thetas[1][ii,jj]
                    Thetar_sum[ii,jj]+=Thetar[1][ii,jj]
                    Lamba_sum[ii,jj]+=Lamba[1][ii,jj]
                    Alpha_sum[ii,jj]+=Alpha[1][ii,jj]
                    Ksaturated_sum[ii,jj]+=Ksaturated[1][ii,jj]
                elif kk<=dept[2]:
                    Thetas_sum[ii,jj]+=Thetas[2][ii,jj]
                    Thetar_sum[ii,jj]+=Thetar[2][ii,jj]
                    Lamba_sum[ii,jj]+=Lamba[2][ii,jj]
                    Alpha_sum[ii,jj]+=Alpha[2][ii,jj]
                    Ksaturated_sum[ii,jj]+=Ksaturated[2][ii,jj]
                elif kk<=dept[3]:
                    Thetas_sum[ii,jj]+=Thetas[3][ii,jj]
                    Thetar_sum[ii,jj]+=Thetar[3][ii,jj]
                    Lamba_sum[ii,jj]+=Lamba[3][ii,jj]
                    Alpha_sum[ii,jj]+=Alpha[3][ii,jj]
                    Ksaturated_sum[ii,jj]+=Ksaturated[3][ii,jj]
                elif kk<=dept[4]:
                    Thetas_sum[ii,jj]+=Thetas[4][ii,jj]
                    Thetar_sum[ii,jj]+=Thetar[4][ii,jj]
                    Lamba_sum[ii,jj]+=Lamba[4][ii,jj]
                    Alpha_sum[ii,jj]+=Alpha[4][ii,jj]
                    Ksaturated_sum[ii,jj]+=Ksaturated[4][ii,jj]
                else:
                    Thetas_sum[ii,jj]+=Thetas[5][ii,jj]
                    Thetar_sum[ii,jj]+=Thetar[5][ii,jj]
                    Lamba_sum[ii,jj]+=Lamba[5][ii,jj]
                    Alpha_sum[ii,jj]+=Alpha[5][ii,jj]
                    Ksaturated_sum[ii,jj]+=Ksaturated[5][ii,jj]
            Thetas_sum[ii,jj]/=thickness3[ii,jj]
            Thetar_sum[ii,jj]/=thickness3[ii,jj]
            Lamba_sum[ii,jj]/=thickness3[ii,jj]
            Alpha_sum[ii,jj]/=thickness3[ii,jj]
            Ksaturated_sum[ii,jj]/=thickness3[ii,jj]
    

                    
                
                
    return Thetas_sum,Thetar_sum,Lamba_sum,Alpha_sum,Ksaturated_sum
                        
Thetas_sum,Thetar_sum,Lamba_sum,Alpha_sum,Ksaturated_sum=layer3(thickness2)                   
craetTif(path_out+f"/Thetas_layer3.tif", maxcol, maxrow, 1, transform, projection, Thetas_sum,-9999)
craetTif(path_out+f"/Thetar_layer3.tif", maxcol, maxrow, 1, transform, projection, Thetar_sum,-9999)
craetTif(path_out+f"/Lamba_layer3.tif", maxcol, maxrow, 1, transform, projection, Lamba_sum,-9999)
craetTif(path_out+f"/Alpha_layer3.tif", maxcol, maxrow, 1, transform, projection, Alpha_sum,-9999)
craetTif(path_out+f"/Ksaturated_layer3.tif", maxcol, maxrow, 1, transform, projection, Ksaturated_sum,-9999)




