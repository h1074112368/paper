import  netCDF4 as nc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


dataset = nc.Dataset(r"F:\bsPaper\dem.nc")
# 定义需要的经纬度范围

# 全局属性存入global_attrs_dict中
global_attrs = dataset.ncattrs()
global_attrs_dict = {}
for attr in global_attrs:
    print(attr,dataset.getncattr(attr))
    global_attrs_dict[attr] = dataset.getncattr(attr)
# 根据经纬度大小，获取索引范围x1,x2,y1,y2
old_lat = dataset.variables['y'][:]
old_lon = dataset.variables['x'][:]

# 新的经纬度
new_lat = old_lat[:64]
new_lon = old_lon[:64]
# 获取维度信息
dims = dataset.dimensions
# 获取变量名列表
variables_list = list(dataset.variables.keys())
# 新建一个nc文件
newfilename = r"F:\bsPaper\dem_clip.nc"
newdataset = nc.Dataset(newfilename,'w')
# 创建维度
for item in dims.items():
    size = item[1].size
    if item[1].name == 'y':
        size = len(new_lat)
    if item[1].name == 'x':
        size = len(new_lon)
    newdataset.createDimension(item[1].name, size)
# 设置新nc文件的全局属性
newdataset.setncatts(global_attrs_dict)
# 写入裁剪后的数据
for varname in variables_list:
    print('current variable: %s' % varname)
    var = dataset.variables[varname]
    # 创建变量
    newdataset.createVariable(varname,var.dtype,var.dimensions)
    # 获取变量属性
    var_attr_dict = {}
    for attr in var.ncattrs():
        var_attr_dict[attr] = var.getncattr(attr)
    newvar = newdataset.variables[varname]
    # 写入变量属性
    newvar.setncatts(var_attr_dict)
    # 获取维度信息
    dims_name_list = list(newvar.dimensions)
    # 若有经度或维度，则需要进行截取
    if 'x' in dims_name_list or 'y' in dims_name_list:
        # 1维数据
        if len(dims_name_list) == 1:
            if 'x' in dims_name_list:
                newvar[:] = var[:][:64]
            elif 'y' in dims_name_list:
                newvar[:] = var[:][:64]
        # 2维数据
        elif len(dims_name_list) == 2:
            newvar[:] = var[:][:64,:64]
        # 3维数据
        elif len(dims_name_list) == 3:
            newvar[:] = var[:][:,:64,:64]
        else:
            # 数据中只有三维及以下数据，因此不处理超过三维的数据
            raise ValueError('variable %s\'s dimension > 3.' %varname)
    # 无经纬度，不需要截取数据
    else:
        newvar[:] = var[:]
# 关闭文件
newdataset.close()
dataset.close()
