import  netCDF4 as nc
from osgeo import gdal, ogr, gdalconst
import pandas as pd
def read_tss_header(tssfilename):
    """ Read header of a tss file (used in inflow)
        :param tssfilename  path and name of the tss
        :returns outlets_id  list of column names in tss file
    """
    with open(tssfilename) as fp:
        rec = fp.readline()
        if rec.split()[0] == 'timeseries':
            # LISFLOOD tss file with header
            # get total number of outlets
            outlets_tot_number = int(fp.readline())
            fp.readline()
            outlets_id = []
            for i in range(0, outlets_tot_number - 1):
                rec = fp.readline()
                # rec = int(rec.strip())
                outlets_id.append(rec)  #Lisflood ID code for output points
            # read tss data
            # tssdata = pd.read_table(tssfilename, delim_whitespace=True, header=None, names=outlets_id, index_col=0,
            #                        skiprows=outlets_tot_number + 2)
            # tssdata = pd.read_table(tssfilename, delim_whitespace=True, header=None, names=outlets_id[0], index_col=0)
            # print(tssdata)

        else:
            # LISFLOOD tss file without header (table)
            numserie = len(rec.split())
            outlets_id = []
            for i in range(1, numserie):
                outlets_id.append(i)  #Lisflood progressive ID code for output points
            # read tss data
            # tssdata = pd.read_table(tssfilename, delim_whitespace=True, header=None, names=outlets_id[0], index_col=0)
            # print(tssdata)
    fp.close()
    return outlets_id
path=r"D:\lisflod\lisflood-usecases-master\LF_ETRS89_UseCase\inflow\inflow.tss"
# path=r"D:\lisflod\lisflood-usecases-master\LF_lat_lon_UseCase\inflow\inflow.tss"
read_tss_header(path)
# from pcraster import *

# ncSet=nc.Dataset(r"D:\lisflod\lisflood-usecases-master\LF_ETRS89_UseCase\meteo\pr_hourly.nc")
# print(ncSet.variables.keys())
# print(ncSet.variables.items())
# with open(r"D:\lisflod\out\dis.tss") as fp:
#     rec = fp.readline()
#     print()
# ncSet=nc.Dataset(r"D:\lisflod\out\dis.tss")
drv_=gdal.GetDriverByName('PCRaster')
# drv_.Register()
a=gdal.Open(r"D:\lisflod\out\dis.tss")
# print(ncSet.variables.keys())
# print(ncSet.variables.items())