import numpy as np
from fangshanduozaizhong import readTifArray,craetTif
import os
from osgeo import gdal
path="/mnt/data2/hyc/tifdata/soil_texture/re"
name="100200"
name2="100_200"
T=0
D=readTifArray(f"/mnt/data2/hyc/tifdata/soil_texture/re/bd{name}.tif")
D[D<0]=0
D=D/1000
C=readTifArray(f"/mnt/data2/hyc/tifdata/soil_texture/re/clay_{name2}cm_mean.tif")
C[C<0]=np.median(C[C>0])
C=C/10.0
sand=readTifArray(f"/mnt/data2/hyc/tifdata/soil_texture/re/sand_{name2}cm_mean.tif")
sand[sand<0]=0
sand=sand/10.0
S=readTifArray(f"/mnt/data2/hyc/tifdata/soil_texture/re/silt_{name2}cm_mean.tif")
S[S<0]=np.median(S[S>0])
S=S/10.0
S[np.isnan(S)]=np.median(S)
pH=readTifArray(f"/mnt/data2/hyc/tifdata/soil_texture/re/ph{name}.tif")
pH[pH<0]=0
pH=pH/100
CEC=readTifArray(f"/mnt/data2/hyc/tifdata/soil_texture/re/cec{name}.tif")
CEC[CEC<0]=0
CEC=CEC/100
OC=readTifArray(f"/mnt/data2/hyc/tifdata/soil_texture/re/soc{name}.tif")
OC[OC<0]=0
OC=OC/100/10
def get_tif_para(path):
    tif = gdal.Open(path)
    projection = tif.GetProjection()
    transform = tif.GetGeoTransform()
    maxcol = tif.RasterXSize
    maxrow = tif.RasterYSize
    return projection, transform, maxcol, maxrow
def Thetas(D,T,S,C,OC,CEC,pH):    
    return 0.83080-0.28217*D+0.0002728*C+0.000187*S
def Thetar(D,T,S,C,OC,CEC,pH):
    sand=100-S-C
    res=0.041*np.ones_like(S)
    res[sand<2]=0.179
    return res
def Lamba(D,T,S,C,OC,CEC,pH):    
    return np.power(10,0.22236-0.3189*D-0.05558*T-0.005306*C-0.003084*S-0.01072*OC)
def Alpha(D,T,S,C,OC,CEC,pH):    
    return np.power(10,-0.43348-0.41729*D-0.04762*OC+0.21810*T-0.01581*C-0.01207*S)
def Ksaturated(D,T,S,C,OC,CEC,pH):    
    return np.power(10,0.40220+0.26122*pH+0.44565*T-0.02329*C-0.01265*S-0.01038*CEC)
projection, transform, maxcol, maxrow=get_tif_para("/mnt/data2/hyc/tifdata/soil_texture/re/bd05.tif")
path_out="/mnt/data2/hyc/tifdata/soil_texture"

craetTif(path_out+f"/Thetas{name2}.tif", maxcol, maxrow, 1, transform, projection, Thetas(D,T,S,C,OC,CEC,pH),-9999)
craetTif(path_out+f"/Thetar{name2}.tif", maxcol, maxrow, 1, transform, projection, Thetar(D,T,S,C,OC,CEC,pH),-9999)
craetTif(path_out+f"/Lamba{name2}.tif", maxcol, maxrow, 1, transform, projection, Lamba(D,T,S,C,OC,CEC,pH),-9999)
craetTif(path_out+f"/Alpha{name2}.tif", maxcol, maxrow, 1, transform, projection, Alpha(D,T,S,C,OC,CEC,pH),-9999)
craetTif(path_out+f"/Ksaturated{name2}.tif", maxcol, maxrow, 1, transform, projection, Ksaturated(D,T,S,C,OC,CEC,pH),-9999)


