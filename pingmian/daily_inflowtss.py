import pandas as pd
import numpy as np
path=r'D:\tensorflow\lisflood\shanghai\inflow'
file=open(path+'/daily_inflow.tss','w',encoding='utf-8')
file.write("timeseries scalar")
file.write('\n')
file.write("5")
file.write('\n')
file.write("timestep")
file.write('\n')
file.write("1")
file.write('\n')
file.write("2")
file.write('\n')
file.write("3")
file.write('\n')
file.write("4")
file.write('\n')
time=pd.date_range('2021-07-23 00:00:00','2021-07-28 00:00:00',freq="D")
#huangpu 30.162
#changjiang 92.2833
# data=((np.random.normal(size=[time.shape[0],2])+1)/2*200).astype('float32')
data=np.ones([time.shape[0],4],dtype='float32')*567
timevalue=np.array([range(time.shape[0])])+1
inp=np.concatenate([timevalue.T,data],axis=1)
inp=inp.astype('str')
# timevalue=np.array(range(time.shape[0]))+1
for i in inp:
    temp=''
    for j in i:
        temp+=j+' '
    file.write(temp)
    file.write('\n')
file.close()