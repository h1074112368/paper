from fangshanduozaizhong import shpReadTable
import numpy as np
import pandas as pd
def jiangshui():
    table=shpReadTable(r"F:\bsPaper\菲特烟花水位降水数据\鑿茬壒鐑熻姳姘翠綅闄嶆按鏁版嵁\yanhua_jiangshui.shp")
    # table.iloc[:,5:]=table.iloc[:,5:]*12
    out=table.iloc[:,5:]*12
    # pd.DataFrame(table).to_csv("yanhua_jiangshui.csv",encoding='gbk')
    t=[]
    for i in range(int(out.shape[1]/24)):
        t.append([np.sum(out.iloc[:,i*24:(i+1)*24],axis=1)])
    res=np.concatenate(t).T
    new_table=table.iloc[:,:5+6]
    new_table.iloc[:,5:]=res[:,:6]
    new_table.to_csv(r"F:\bsPaper\yanhua_pr_daily.csv",encoding='gbk')
    return res
    # res=res.T.reshape([15,int(out.shape[1]/24)])
    print()
jiangshui()