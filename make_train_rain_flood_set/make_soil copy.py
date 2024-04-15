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
    Thetas.append(readTifArray(path_out+"/Thetas"+i+".tif")*chan[ii])
    Thetar.append(readTifArray(path_out+"/Thetar"+i+".tif")*chan[ii])
    Lamba.append(readTifArray(path_out+"/Lamba"+i+".tif")*chan[ii])
    Alpha.append(readTifArray(path_out+"/Alpha"+i+".tif")*chan[ii])
    Ksaturated.append(readTifArray(path_out+"/Ksaturated"+i+".tif")*chan[ii])
    ii+=1
    
def layer2(thickness2):
    finsh=np.zeros_like(thickness2)
    leiji_sum=np.zeros_like(thickness2)
    Thetas_sum=np.zeros_like(thickness2)
    Thetar_sum=np.zeros_like(thickness2)
    Lamba_sum=np.zeros_like(thickness2)
    Alpha_sum=np.zeros_like(thickness2)
    Ksaturated_sum=np.zeros_like(thickness2)
    for i in range(1,len(dept)):
        for ii in range(thickness2.shape[0]):
            for jj in range(thickness2.shape[1]):
                if finsh[ii,jj]==0:
                    if(thickness2[ii,jj]+5>dept[i]):
                        Thetas_sum[ii,jj]+=Thetas[i][ii,jj]
                        Thetar_sum[ii,jj]+=Thetar[1][ii,jj]
                        Lamba_sum[ii,jj]+=Lamba[i][ii,jj]
                        Alpha_sum[ii,jj]+=Alpha[i][ii,jj]
                        Ksaturated_sum[ii,jj]+=Ksaturated[i][ii,jj]
                        leiji_sum[ii,jj]+=chan[i]
                    elif(thickness2[ii,jj]+5<=dept[i]):
                        Thetas_sum[ii,jj]+=Thetas[i+1][ii,jj]*(thickness2[ii,jj]+5-dept[i-1])/chan[i]
                        Thetar_sum[ii,jj]+=Thetar[i+1][ii,jj]*(thickness2[ii,jj]+5-dept[i-1])/chan[i]
                        Lamba_sum[ii,jj]+=Lamba[i+1][ii,jj]*(thickness2[ii,jj]+5-dept[i-1])/chan[i]
                        Alpha_sum[ii,jj]+=Alpha[i+1][ii,jj]*(thickness2[ii,jj]+5-dept[i-1])/chan[i]
                        Ksaturated_sum[ii,jj]+=Ksaturated[i+1][ii,jj]*(thickness2[ii,jj]+5-dept[i-1])/chan[i]
                        leiji_sum[ii,jj]+=thickness2[ii,jj]+5-dept[i-1]
                        Thetas_sum[ii,jj]/=leiji_sum[ii,jj]
                        Thetar_sum[ii,jj]/=leiji_sum[ii,jj]
                        Lamba_sum[ii,jj]/=leiji_sum[ii,jj]
                        Alpha_sum[ii,jj]/=leiji_sum[ii,jj]
                        Ksaturated_sum[ii,jj]/=leiji_sum[ii,jj]
                        finsh[ii,jj]=1
    return Thetas_sum,Thetar_sum,Lamba_sum,Alpha_sum,Ksaturated_sum
                        
Thetas_sum,Thetar_sum,Lamba_sum,Alpha_sum,Ksaturated_sum=layer2(thickness2)                   
craetTif(path_out+f"/Thetas_layer2.tif", maxcol, maxrow, 1, transform, projection, Thetas_sum,-9999)
craetTif(path_out+f"/Thetar_layer2.tif", maxcol, maxrow, 1, transform, projection, Thetar_sum,-9999)
craetTif(path_out+f"/Lamba_layer2.tif", maxcol, maxrow, 1, transform, projection, Lamba_sum,-9999)
craetTif(path_out+f"/Alpha_layer2.tif", maxcol, maxrow, 1, transform, projection, Alpha_sum,-9999)
craetTif(path_out+f"/Ksaturated_layer2.tif", maxcol, maxrow, 1, transform, projection, Ksaturated_sum,-9999)


def layer3(thickness2):
    finsh=np.zeros_like(thickness2)
    leiji_sum=np.zeros_like(thickness2)
    Thetas_sum=np.zeros_like(thickness2)
    Thetar_sum=np.zeros_like(thickness2)
    Lamba_sum=np.zeros_like(thickness2)
    Alpha_sum=np.zeros_like(thickness2)
    Ksaturated_sum=np.zeros_like(thickness2)
    for i in range(1,len(dept)):
        for ii in range(thickness2.shape[0]):
            for jj in range(thickness2.shape[1]):
                if finsh[ii,jj]==0:
                    if(thickness2[ii,jj]+5>dept[i]and i+1<=4 and thickness3[ii,jj]+thickness2[ii,jj]+5<dept[i+1]):
                        Thetas_sum[ii,jj]=Thetas[i+1][ii,jj]/chan[i+1]
                        Thetar_sum[ii,jj]=Thetar[1+1][ii,jj]/chan[i+1]
                        Lamba_sum[ii,jj]=Lamba[i+1][ii,jj]/chan[i+1]
                        Alpha_sum[ii,jj]=Alpha[i+1][ii,jj]/chan[i+1]
                        Ksaturated_sum[ii,jj]+=Ksaturated[i+1][ii,jj]/chan[i+1]
                        finsh[ii,jj]=1
                    elif(thickness2[ii,jj]+5<dept[i]and thickness3[ii,jj]+thickness2[ii,jj]+5>=dept[i]):
                        Thetas_sum[ii,jj]+=Thetas[i+1][ii,jj]*(dept[i]-thickness2[ii,jj]-5)/chan[i]
                        Thetar_sum[ii,jj]+=Thetar[i+1][ii,jj]*(dept[i]-thickness2[ii,jj]-5)/chan[i]
                        Lamba_sum[ii,jj]+=Lamba[i+1][ii,jj]*(dept[i]-thickness2[ii,jj]-5)/chan[i]
                        Alpha_sum[ii,jj]+=Alpha[i+1][ii,jj]*(dept[i]-thickness2[ii,jj]-5)/chan[i]
                        Ksaturated_sum[ii,jj]+=Ksaturated[i+1][ii,jj]*(dept[i]-thickness2[ii,jj]-5)/chan[i]
                        leiji_sum[ii,jj]+=thickness2[ii,jj]+5-dept[i-1]
                        Thetas_sum[ii,jj]/=leiji_sum[ii,jj]
                        Thetar_sum[ii,jj]/=leiji_sum[ii,jj]
                        Lamba_sum[ii,jj]/=leiji_sum[ii,jj]
                        Alpha_sum[ii,jj]/=leiji_sum[ii,jj]
                        Ksaturated_sum[ii,jj]/=leiji_sum[ii,jj]
                        finsh[ii,jj]=1
    return Thetas_sum,Thetar_sum,Lamba_sum,Alpha_sum,Ksaturated_sum
                        
Thetas_sum,Thetar_sum,Lamba_sum,Alpha_sum,Ksaturated_sum=layer3(thickness2)                   
craetTif(path_out+f"/Thetas_layer3.tif", maxcol, maxrow, 1, transform, projection, Thetas_sum,-9999)
craetTif(path_out+f"/Thetar_layer3.tif", maxcol, maxrow, 1, transform, projection, Thetar_sum,-9999)
craetTif(path_out+f"/Lamba_layer3.tif", maxcol, maxrow, 1, transform, projection, Lamba_sum,-9999)
craetTif(path_out+f"/Alpha_layer3.tif", maxcol, maxrow, 1, transform, projection, Alpha_sum,-9999)
craetTif(path_out+f"/Ksaturated_layer3.tif", maxcol, maxrow, 1, transform, projection, Ksaturated_sum,-9999)




