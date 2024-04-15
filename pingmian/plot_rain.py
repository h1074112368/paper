import sys
import matplotlib.pyplot as plt
import pandas as pd
import  numpy as np
data=pd.read_excel(r"F:\bsPaper\result_rain_set.xlsx",sheet_name="Sheet2")
index=list(range(data.shape[1]))
label0=[f"{i}" for i in range(5,121,5)]
plt.rcParams['font.sans-serif']=['SimHei']
fig, ax = plt.subplots(figsize = (7,3), dpi = 300)
# for i in range(10,len(data)):
    # x_s = np.arange(0, data.shape[1])
    # input_data=[0]
    # [input_data.append(i) for i in data.iloc[i,1:]]
    # linear_model = np.polyfit(x_s, input_data, 3)
    # linear_model_fn = np.poly1d(linear_model)

    # plt.plot(x_s, linear_model_fn(x_s))
    # ax.plot(label0, data.iloc[i,1:],linewidth =1.0,color='k')
plt.bar(label0, data.iloc[6,1:])
plt.ylabel("降水强度(mm/5min)")
plt.xlabel("降水历时(min)")
fig.autofmt_xdate()
plt.show()

