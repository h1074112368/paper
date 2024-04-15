import numpy as np
from osgeo import gdal, osr


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
        coords = ct.TransformPoint(x, y)
        return coords[:2]

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


def main():
    tif_file = r"D:\tensorflow\lisflood\data_25m\dem_shanghai_25m_mask.tif"
    gda = GdalTif(tif_file)
    print("数据投影信息:", gda.get_projection())

    pt = (300, 500)
    print("pt = ", pt)

    print("图上坐标 -> 投影坐标：")
    point = gda.imagexy2geo(*pt)
    print("WGS84 point:", point)

    print("投影坐标 -> 图上坐标：")
    pt = gda.geo2imagexy(*point)
    print("pixel pt:", pt)

    print("投影坐标 -> 经纬度：")
    coord = gda.geo_to_lonlat(*point)
    print("gps coord:", coord)

    print("经纬度 -> 投影坐标：")
    point = gda.lonlat2geo(*coord)
    print("point:", point)


if __name__ == '__main__':
    main()
