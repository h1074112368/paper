from timeArray import timeArray
import shutil,os
name=timeArray('2021-07-23 00','2021-07-29 00',24)
i=0
# [file for file if file.endswith('tif') in os.listdir(r'F:\yh_pr_hour\pre_re')]
for file in os.listdir(r'F:\yh_pr\tif_re\ta_avg'):
    if file.endswith('tif'):
        # set.append([readTifArray(r'F:\yh_pr\tif_re\ta_avg'+'/'+file)])
        shutil.copyfile(r'F:\yh_pr\tif_re\ta_avg'+'/'+file, r'G:\data\ta/' + name[i] + '.tif')
        i+=1
# name=timeArray('2021-07-23 00','2021-07-28 23',1)
# for i in range(len(name)):
#     shutil.copyfile(r'F:\LAI\re_GLASS01D01.V60.A2021209.h28v05.2022151.tif',r'G:\data\lai/'+i+'.tif')