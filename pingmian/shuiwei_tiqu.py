import os
import netCDF4 as nc
from scipy import stats
from fangshanduozaizhong import readTifArray
import  numpy as np
import numpy as np
from osgeo import gdal, osr, __version__
from joblib import Parallel, delayed, load, dump
from multiprocessing import Pool
import multiprocessing as mp


class GdalTif(object):
    def __init__(self, tif_path):
        self.tif_path = tif_path
        self.dataset = self._read_tif(tif_path)

    @staticmethod
    def _read_tif(tif_path):
        """ 读取GDAL地理数据
        :param tif_path: tif图像路径
        :return: GDAL地理数据
        """

        dataset = gdal.Open(tif_path)
        if dataset is None:
            print(tif_path + " 文件无法打开")
        return dataset

    def get_geo_trans(self):
        """ 获取仿射矩阵信息
        :return: 仿射矩阵信息有六个参数，描述的是栅格行列号和地理坐标之间的关系:
            (
                0：左上角横坐标（投影坐标，经度）；
                1：像元宽度,影像东西/水平方向分辨率；
                2：行旋转，如果图像北方朝上，该值为0；
                3：左上角纵坐标（投影坐标，纬度）；
                4：列旋转，如果图像北方朝上，该值为0；
                5：像元高度,影像南北/垂直方向分辨率；
            )
            如果图像不含地理坐标信息，默认返回值是：(0,1,0,0,0,1)
        """

        return self.dataset.GetGeoTransform()

    def get_projection(self):
        """ 获取数据投影信息
        :return: INFO
        """

        return self.dataset.GetProjection()

    def get_srs_pair(self):
        """ 获得给定数据的投影参考系和地理参考系
        :param dataset: GDAL地理数据
        :return: 投影参考系和地理参考系
        """

        prosrs = osr.SpatialReference()

        prosrs.ImportFromWkt(self.dataset.GetProjection())
        prosrs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs

    def imagexy2geo(self, row, col):
        """ 根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
        :param dataset: GDAL地理数据
        :param row: 像素的行号(i)
        :param col: 像素的列号(j)
        :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y) - WGS84
        """

        trans = self.dataset.GetGeoTransform()
        x = trans[0] + col * trans[1] + row * trans[2]
        y = trans[3] + col * trans[4] + row * trans[5]
        return (x, y)

    def geo2imagexy(self, x, y):
        """ 根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        """

        trans = self.dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)[::-1]  # 使用numpy的linalg.solve进行二元一次方程的求解

    def geo_to_lonlat(self, x, y):
        """ 将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
        :param x: 投影坐标x
        :param y: 投影坐标y
        :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
        """

        prosrs, geosrs = self.get_srs_pair()
        ct = osr.CoordinateTransformation(prosrs, geosrs)

        def fun1(ct, x1, y1):
            coords = list(ct.TransformPoint(x1, y1))
            # coords[2]=float(wdept0)
            return coords[:2]

        # out=[]
        # args=[(x[i],y[i],self) for i in    range(x.shape[0])]
        # p=Pool(192)
        # out=p.starmap(fun1,args)
        # for i in    range(x.shape[0]):
        #     out.append(p.apply_async(fun1,args=(ct,x[i],y[i],) ))
        # p.close()
        # p.join()
        # out = Parallel(n_jobs=192)(delayed(fun1)(x[i],y[i],self) for i in    range(x.shape[0]))
        out = [fun1(ct, x[i], y[i]) for i in range(len(x))]
        return out

    def lonlat2geo(self, lon, lat):
        """ 将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param lon: 地理坐标lon经度
        :param lat: 地理坐标lat纬度
        :return: 经纬度坐标(lon, lat)对应的投影坐标
        """

        prosrs, geosrs = self.get_srs_pair()
        ct = osr.CoordinateTransformation(geosrs, prosrs)
        coords = ct.TransformPoint(lon, lat)
        return coords[:2]





def getloc2(pt):
    print(__version__)

    tif_file = r"F:\bsPaper\re_CNBH10m\CNBH10M_hebing1.tif"
    gda2 = GdalTif(tif_file)
    print("数据投影信息:", gda2.get_projection())

    # pt = (300, 500)
    print("pt = ", pt)

    print("经纬度 -> 投影坐标：")
    point = gda2.lonlat2geo(*pt)
    print("point:", point)
    res = gda2.geo2imagexy(point[0], point[1])
    print("pixel pt:", res)
    return res.astype('int')





if __name__ == '__main__':
    loc_lon=[
        [121.489834,31.347504],#新江湾泵闸	杨浦区	钱家浜	蕴南片
        # [],#朱泖河水闸	青浦区	朱泖河	青松片
        # [],#李家圩套闸	青浦区	太浦河	太北片
        # [],#李红套闸	青浦区	太浦河	太北片
        # [],#钱盛节制闸	青浦区	太浦河	太北片
        # [],#练塘北套闸	青浦区	太浦河	太北片
        # [],#金田北节制闸	青浦区	太浦河	太北片
        # [],#八百亩套闸	青浦区	太浦河	太北片
        # [],#练塘南套闸	青浦区	太浦河	太南片
        # [],#金田南节制闸	青浦区	太浦河	太南片
        # [],#泖口套闸	青浦区	太浦河	太南片
        # [],  # 中港水闸	奉贤区	中港	浦东片
        # [],  # 巨潮水闸	奉贤区	巨潮港	浦东片
        # [],  # 南竹港出海闸	奉贤区	南竹港	浦东片
        # [],  # 紫石泾水闸	松江区	紫石泾	浦南东片
        # [],  # 练塘南套闸	青浦区	太浦河	太南片
        # [],  # 金田南节制闸	青浦区	太浦河	太南片
        # [],  # 泖口套闸	青浦区	太浦河	太南片
        # [],#荻泾套闸	宝山区	狄泾	嘉宝北片
        # [],#跃进水闸	崇明区	鳗鲤港	长兴岛片
        # [],#庙港南闸	崇明区	庙港	崇明岛片
        # [],#庙港北闸	崇明区	庙港	崇明岛片
        # [],#崇西水闸	崇明区	环岛运河	崇明岛片
        # [],#东平河水闸	崇明区	东平河	崇明岛片
        # [],#三沙洪水闸	崇明区	三沙洪	崇明岛片
        # [],#六滧港南闸	崇明区	六滧港	崇明岛片
        # [],#张网港水闸	崇明区	张网港	崇明岛片
        # [],#奚家港水闸	崇明区	奚家港	崇明岛片
        # [],#新河水闸	崇明区	新河港	崇明岛片
        # [],#新建水闸	崇明区	新建港	崇明岛片
        # [],#鸽龙港南闸	崇明区	鸽龙港	崇明岛片
        # [],#中横沥北闸	闵行区	北横泾	淀北片
        # [],#祝家港水闸	松江区	南泖港	浦南东片
        # [],#祝家港水闸	松江区	南泖港	浦南东片
        # [],#姚家浜水闸	闵行区	姚家浜	浦东片
        # [],#春申塘水闸	闵行区	春申塘	淀南片
        # [],#华田泾枢纽	松江区	华田泾	青松片
        # [],#六磊塘水闸	闵行区	六磊塘	青松片
        # [],#横沥水闸	嘉定区	横沥	嘉宝北片
        # [],#孙浜套闸	嘉定区	孙浜	嘉宝北片
        # [],#盐铁北套闸	嘉定区	盐铁塘	嘉宝北片
        # [],#蕰西枢纽	嘉定区	蕴藻浜	嘉宝北片
        # [],#横沥泵闸	嘉定区	北横泾	嘉宝北片
        # [],#封浜泵闸	嘉定区	封浜	嘉宝北片
        # [],#张堰（二）	金山区	张泾河	浦南东片
        # [],#新建	崇明区	新建港	崇明岛片
        # [],#崇西闸	崇明区	长江口	崇明岛片
        # [],#新海镇	崇明区	环岛运河	崇明岛片
        # [],#草棚镇	崇明区	界河	崇明岛片
        # [],#南鸽龙	崇明区	鸽龙港	崇明岛片
        # [],#北鸽龙	崇明区	鸽龙港	None
        # [],#港西团结	崇明区	三沙洪	崇明岛片
        # [],#新河金桥	崇明区	环岛运河	崇明岛片
        # [],#大新镇	崇明区	直河港	崇明岛片
        # [],#光明桥	崇明区	环岛运河	崇明岛片
        # [],#农业园区	崇明区	七效港	崇明岛片
        # [],#陈家镇朝阳	崇明区	环岛运河	崇明岛片
        # [],#堡镇	崇明区	长江口	崇明岛片
        # [],#马家港	崇明区	长江口	长兴岛片
        # [],#横沙民星	崇明区	横沙创建河	横沙岛片
        # [],#九段沙	浦东新区	长江口	浦东片
        # [],#高桥（二）	浦东新区	长江口	None
        # [],#米市渡	松江区	黄浦江	青松片
        # [],#松浦大桥	松江区	黄浦江	浦南东片
             ]
    pic_loc=getloc2((121.927400,31.018659))
    wdept_w=40
    wdept=np.array(nc.Dataset(r"G:\keshixu_wdept\nc\wdept.nc")["wdept"])[:,max(pic_loc[0]-wdept_w,0):pic_loc[0]+wdept_w,max(pic_loc[1]-wdept_w,0):pic_loc[1]+wdept_w].reshape((6,-1))
    wdept_record=[
        # [2.8,2.88,3.07,	3.43,3.76,3.87,3.88]
        [2.8,2.88,3.05,3.42,3.77,1.13,0]
]


    def r2(y1, y2):
        y1 = y1.flatten()
        y2 = y2.flatten()
        return 1 - np.sum(np.power(y1 - y2, 2)) / np.sum(np.power(y2 - np.mean(y2), 2))
    result_max=0
    for i in range(wdept.shape[1]):
        result=r2(wdept[:,i][:4],np.array(wdept_record[0][1:])[:4])#-wdept_record[0][0])
        if result>result_max:
            result_max=result
            print(i,result_max)
    print(np.max(wdept))
    path0=r"G:\keshixu_wdept\tif"

    LAIMaps=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\lai\LAIMaps.nc")["value"]).flatten()[index9999]
    ksat1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\ksat1_o.nc")["value"]).flatten()[index9999]
    lambda1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\lambda1_o.nc")["value"]).flatten()[index9999]
    thetar1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\thetar1_o.nc")["value"]).flatten()[index9999]
    thetas1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\thetas1_o.nc")["value"]).flatten()[index9999]
    genua1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\genua1_o.nc")["value"]).flatten()[index9999]




    # wdept=[readTifArray(path)  if path.endswith("tif") for path  in wdept_path ]
    print()