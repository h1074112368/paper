import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # For tf.data and preprocessing only.
# import keras
# from keras import layers
# from keras import ops
import  netCDF4 as nc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
num_classes = 100
input_shape = (64,64,27)

patch_size = (2, 2)  # 2-by-2 sized patches
dropout_rate = 0.05  # Dropout rate
num_heads = 8  # Attention heads
embed_dim = 64*2  # Embedding dimension
num_mlp = 256  # MLP layer size
# Convert embedded patches to query, key, and values with a learnable additive
# value
qkv_bias = True
window_size = 2  # Size of attention window
shift_size = 1  # Size of shifting window
image_dimension = (64,64,27)  # Initial image size

num_patch_x = input_shape[0] // patch_size[0]
num_patch_y = input_shape[1] // patch_size[1]

learning_rate = 1e-3
batch_size = 16
num_epochs = 30
validation_split = 0.3
weight_decay = 0.0001
label_smoothing = 0.1

x=[]
y=[]
import pandas as pd
rain_set=np.load("/home/hyc/flaskFiles/make_train_rain_flood_set/rainfall_set_new_kechixu.npy")
dem=np.array(nc.Dataset('/home/hyc/shanghai_clip_soil/dem.nc')['value'])[:64,:]
es0=np.array(nc.Dataset(f"/home/hyc/shanghai_clip_soil/meteo/hour_e0.nc")['value'])[:24,:64,:]
lai=np.array(nc.Dataset(f"/home/hyc/shanghai_clip_soil/lai/LAIMaps.nc")['value'])[0,:64,:]
for i in range(0,3000):
    for j in range(24):
        if j==0:
            tmp=rain_set[i][j]*np.ones((64,64,1))
        else:
            tmp=np.append(tmp,rain_set[i][j]*np.ones((64,64,1)),axis=2)
        tmp=np.append(tmp,np.reshape(dem,(64,64,1)),axis=2)
        tmp=np.append(tmp,np.reshape(es0[j],(64,64,1)),axis=2)
        tmp=np.append(tmp,np.reshape(lai,(64,64,1)),axis=2)
    
    x.append(tmp) 
    
    ncDataset=nc.Dataset(f"/home/hyc/flood_train_set_new_soil/wdept{'%04d'% i}.nc")
    ncSet=np.array(ncDataset['wdept'])[:,:64]
    # ncSet=np.append(ncSet,np.reshape(ncSet[:,-1],(24,1,64)),axis=1)
    y.append([ncSet])
x=np.array(x)
y=np.concatenate(y,axis=0)
# y[y<0.1]=0

# max1 = np.max(x)  # 用于归一化处理
# x = x / max1

# x_train, x_test = x_train / 255.0, x_test / 255.0
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
index=tf.range(x.shape[0])
# index=tf.random.shuffle(index)


index_train,index_test=index[:3000-900],index[3000-900:]
x_train, x_test = x[index_train], x[index_test]
y_train, y_test = y[index_train].reshape((-1,24,64,64)), y[index_test].reshape((-1,24,64,64))



y_train, y_test=y_train.flatten(),y_test.flatten()
x_train, x_test = x_train.reshape((y_train.shape[0], -1)), x_test.reshape((y_test.shape[0], -1))


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

# regr = RandomForestRegressor(n_estimators=100,n_jobs=-1,verbose=1,random_state=0)
# regr.fit(x_train, y_train)
# print(regr.predict([[0, 0, 0, 0]]))
import joblib 
# joblib.dump(regr, 'rfg_model.pkl')



# model.save("/home/hyc/swin/swin/model_1.h5", overwrite=True, include_optimizer=True ) 
import os
regr=joblib.load("rfg_model.pkl")


# model.save("/home/hyc/swin/swin/model_1.h5") 
# del model
# model = keras.models.load_model("/home/hyc/swin/swin/my_model.keras")
result = regr.predict(x_test)

aucc=1-np.sum(np.abs(result-y_test))/np.sum(y_test)
print(aucc)
def rmse(y1,y2):
    return np.sqrt(np.sum(np.power(y1-y2,2))/len(y1))
print(rmse(result,y_test))
def rmse(y1,y2):
    return np.sum(np.abs(y1-y2))/len(y1)
print(rmse(result,y_test))
print(regr.score(x_test,y_test))
def r2(y1,y2):
    y1=y1.flatten()
    y2=y2.flatten()
    return 1-np.sum(np.power(y1-y2,2))/np.sum(np.power(y2-np.mean(y2),2))
print("r2",r2(result,y_test))
result=np.reshape(result,(-1,24,64,64))
y_test=np.reshape(y_test,(-1,24,64,64))
np.save("/home/hyc/swin/swin/yuce_rf.npy",result[207+600])
np.save("/home/hyc/swin/swin/shiji_rf.npy",y_test[207+600])


