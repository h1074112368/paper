import os
import numpy as np
import  netCDF4 as nc
import pickle
import pandas as pd
err=[]
def save_pickle(name,data):
    
    with open(name,'wb') as f:
        pickle.dump(data,f)
def load_pickle(name):
    
    with open(name,'rb') as f:
        return pickle.load(f)
# datasetload=load_pickle("cldasdata.npy")
def readNc(path,i):
    try:
        if path is None:
            print(f"部分缺失 : {i} {path}")
            err.append((i,path,"部分缺失"))
            # return path
        else:
            if len(list(nc.Dataset(path).variables.keys()))==3:
                for key in list(nc.Dataset(path).variables.keys()):
                    if key!='LAT' and key!='LON' :
                        if nc.Dataset(path)[key].shape[0]==1040:
                            return np.array(nc.Dataset(path)[key])
                        else:
                            print(f"分辨率{nc.Dataset(path)[key].shape}err : {i} {path}")
                            err.append((i,path,nc.Dataset(path)[key].shape))
                            # return path
            else:
                print(f"维度err {list(nc.Dataset(path).variables.keys())}: {i} {path}")
                err.append((i,path,list(nc.Dataset(path).variables.keys())))
                # return path
    except:
        print(f"err: {i} {path}")
        err.append((i,path,"文件损坏"))
        # return path
    
path='/mnt/data2/willchan/cldas'
shuxingList=os.listdir(path)
datapathset=[]
dataset=[]
timeset=[]
timeindex=pd.date_range('2017-01-16 00:00:00','2017-08-28 00:00:00',freq="1h")
timejili=np.empty((7,len(timeindex)),dtype="object")
dataset=np.zeros((7,len(timeindex),1040,1600),dtype="float32")
for shuxingi in range(len(shuxingList)):
    datapath=[]
    time_shuxing=[]
    for root,dirs,files in os.walk(os.path.join(path,shuxingList[shuxingi])):

        namesplit=root.split('/')
        day_time=f"{namesplit[-3]}-{namesplit[-2]}-{namesplit[-1]}"
    
        for f in files:
            if f.endswith('nc'):
                fsplit=f.split('.')[0].split('-')[-1]
                try:
                    index=int(fsplit[-2:])
                    hour_time=pd.Timestamp(f"{day_time} {fsplit[-2:]}")
                except:
                    continue
                time_index=timeindex.get_loc(hour_time)
                if timejili[shuxingi][time_index] is None:
                    timejili[shuxingi][time_index]=os.path.join(root,f)
    for i in range(timejili.shape[1]):
        dataset[shuxingi,i]=readNc(timejili[shuxingi,i],i)
    # break
# np.save("cldasdata.npy",dataset)
save_pickle("cldasdata.npy",dataset)
save_pickle("cldasdataerr.npy",err)
# datasetload=load_pickle("cldasdata.npy")
# print(np.array(datapathset))

       