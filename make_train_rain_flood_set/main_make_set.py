import numpy as np
import os
from hour_daily_pr_copy import prToNc
from clip_nc_lonlat import clip_nc_lonlat
import shutil
rain_set=np.load("/home/hyc/flaskFiles/make_train_rain_flood_set/rainfall_set_new_kechixu.npy")
for i in range(0,rain_set.shape[0]):
    
    prToNc(rain_set[i])
    clip_nc_lonlat()
    os.system('/home/hyc/.conda/envs/lisflood/bin/python /home/hyc/lisflood-code_re/src/lisf1.py /home/hyc/shanghai_clip_soil/default_init.xml')
    os.system('/home/hyc/.conda/envs/lisflood/bin/python /home/hyc/lisflood-code_re/src/lisf1.py /home/hyc/shanghai_clip_soil/default.xml')
    if not os.path.exists("/home/hyc/flood_train_set_new_soil"):
        os.mkdir("/home/hyc/flood_train_set_new_soil")
    shutil.copyfile(f"/home/hyc/shanghai_clip_soil/out/wdept.nc", f"/home/hyc/flood_train_set_new_soil/wdept{'%04d'% i}.nc")
    