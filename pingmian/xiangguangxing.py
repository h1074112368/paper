import os
import netCDF4 as nc
from scipy import stats
from fangshanduozaizhong import readTifArray
import  numpy as np
table=[]
for timei in range(0,6):
    path0=r"F:\bsPaper\out_daily\wdept"
    wdept_filepath=os.listdir(path0)
    wdept=[]
    cnbh = readTifArray(r"F:\bsPaper\re_CNBH10m\CNBH10M_hebing1.tif").flatten()
    ul = readTifArray(r"D:\tensorflow\lisflood\dem_shanghai_25m\n51_30_2020lc030_land.tif").flatten()
    wdept=readTifArray(r"G:\keshixu_wdept\421550_best\wdept\wdept_"+"{:03d}".format(timei)+".tif").flatten()
    wdept[(ul==60)|(cnbh>0)]=-9999
    index9999=np.where(wdept!=-9999)
    list_map=os.listdir(r"G:\keshixu_wdept\shanghai_new_daily\maps")
    LAIMaps=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\lai\LAIMaps.nc")["value"]).flatten()[index9999]
    # ksat1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\ksat1_o.nc")["value"]).flatten()[index9999]
    # lambda1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\lambda1_o.nc")["value"]).flatten()[index9999]
    # thetar1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\thetar1_o.nc")["value"]).flatten()[index9999]
    # thetas1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\thetas1_o.nc")["value"]).flatten()[index9999]
    # genua1_o=np.array(nc.Dataset(r"G:\keshixu_wdept\shanghai_new_daily\maps\genua1_o.nc")["value"]).flatten()[index9999]
    wdept=wdept[index9999]
    data=[]
    data_name=[]
    for i in list_map[:100]:
        if i!="area.nc" and i!="chan.nc" and i!="chanleng.nc" and i!="cropcoef_i.nc"and i!="cropcoef_o.nc"and i!="cropgrpn_f.nc"and i!="cropgrpn_o.nc"and i!="lat.nc"and i!="outlets copy.nc"and i!="outlets.nc"and i!="pixel.nc"and i!="pixel.nc"and i!="ldd_nottw.nc"and i!="pixel.nc"and i!="ldd_nottw_CNBH.nc"and i!="length.nc"and i!="mannings_f.nc"and i!="mannings_o.nc"and i!="thetar1_o.nc"and i!="thetar3_o.nc":
            path1=os.path.join(r"G:\keshixu_wdept\shanghai_new_daily\maps",i)
            data.append(np.array(nc.Dataset(path1)["value"]).flatten()[index9999])
            data_name.append(i)
    # data.append(LAIMaps)
    # data.append(ksat1_o)
    # data.append(lambda1_o)
    # # data.append(thetar1_o)
    # data.append(thetas1_o)
    # data.append(genua1_o)

    list_meteo=os.listdir(r"G:\keshixu_wdept\shanghai_new_daily\meteo")
    for i in list_meteo:
        if i.split("_")[0]=="daily":
            if i.endswith("nc"):
                path1=os.path.join(r"G:\keshixu_wdept\shanghai_new_daily\meteo",i)
                data.append(np.array(nc.Dataset(path1)["value"])[timei].flatten()[index9999])
                data_name.append(i)
    list_meteo=os.listdir(r"G:\keshixu_wdept\shanghai_new_daily\soilhyd")
    for i in list_meteo:

            if i.endswith("nc"):
                path1=os.path.join(r"G:\keshixu_wdept\shanghai_new_daily\soilhyd",i)
                data.append(np.array(nc.Dataset(path1)["value"]).flatten()[index9999])
                data_name.append(i)
    data.append(LAIMaps)
    data_name.append("LAIMaps")
    data.append(wdept)
    data_name.append("wdept")
    # corr=np.corrcoef(data)

    import  pandas as pd
    name_0=["河道深度","河道底部宽度","河滩宽度","河道梯度","河道曼宁系数","河道坡度"
            ,"高程标准偏差","表层Van Genuchten参数","上层Van Genuchten参数","下层Van Genuchten参数",
            "地表梯度",
    "表层饱和电导率","上层饱和电导率","下层饱和电导率",
    "表层孔径指数","上层孔径指数","下层孔径指数",
    "流向图","上层残余体积土壤含水量","下层残余体积土壤含水量",
    "表层饱和体积土壤含水量","上层饱和体积土壤含水量","下层饱和体积土壤含水量",
    "河道上游面积","参考潜在蒸发量","裸露土壤的蒸发量","开放水域的蒸发量",
    "降水量","日气温","表层土壤深度","上层土壤深度","下层土壤深度","叶面积指数","积水深度"
            ]
    data=np.array(data).T
    data=pd.DataFrame(data,columns=data_name)
    # corr1=data.corr("kendall")
    corr2=data.corr("spearman").iloc[-1]
    table.append(np.array(corr2))
pd.DataFrame(table,columns=name_0).to_excel(r"F:\bsPaper\相关性_3.xlsx")



# wdept=[readTifArray(path)  if path.endswith("tif") for path  in wdept_path ]
print()