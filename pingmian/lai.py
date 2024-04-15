from timeArray import timeArray
import shutil
name=timeArray('2021-01-01 00','2021-12-31 23',24*10)
for i in name:
    shutil.copyfile(r'F:\LAI\re_GLASS01D01.V60.A2021209.h28v05.2022151.tif',r'G:\data\lai/'+i+'.tif')