import numpy as np
import pandas as pd

data=[
    [206,29,991,13174],
    [145,12,800,13443],
    [317,8,1143,12932],
[335,21,834,13210],
    [206,127,893,13174],
    [145,18,794,13443],
    [317,32,1119,12932],
[335,68,787,13210],

]
ii=0
result=[]
for i in data:
    table=np.zeros((3,3))
    table[0,0]=i[2]
    table[0,1]=i[1]
    table[1, 0] = i[0]
    table[1, 1] = i[3]
    table[0,2]=table[0,1]+table[0,0]
    table[1, 2] = table[1, 1] + table[1, 0]
    table[2, 0] = table[0, 0] + table[1, 0]
    table[2, 1] = table[0, 1] + table[1, 1]
    table[2,2]=np.sum(table[:2,:2])
    p0=(table[0,0]+table[1,1])/table[2,2]
    pe = ((table[0, 2] * table[2, 0])+ (table[2, 1] * table[1, 2]))/ table[2, 2]/ table[2, 2]
    kappa=(p0-pe)/(1-pe)
    print(table)
    print(p0,pe,kappa)

    pd.DataFrame(table).to_excel(rf"F:\bsPaper\yanjiuqu\jiaochabiao{ii}.xlsx")
    ii+=1
    result.append((p0,pe,kappa))
pd.DataFrame(result).to_excel(rf"F:\bsPaper\yanjiuqu\result.xlsx")

