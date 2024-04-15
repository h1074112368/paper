import  netCDF4 as nc
import numpy as np
import cv2,os,shutil
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import json
import multiprocessing as mp
from osgeo import gdal
from joblib import Parallel, delayed, load, dump
from multiprocess import cpu_count
mask=np.array(nc.Dataset('/home/hyc/shanghai_new/lai/LAIMaps.nc')['value'])
chan=np.array(nc.Dataset(path+'/chan.nc')['value'])