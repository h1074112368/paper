import numpy as np
def rainfall_intensity_formula(P,t,t_sum):
    
    c=0.846#0.8852
    n=0.656
    r=0.405
    b=7.0
    f1=9.581*(1+c*np.log10(P))#/np.power(7+t_sum,0.656)
    if t/t_sum<r:
        t=t_sum*r-t
        f2=((1-n)*t/r+b)
        f3=np.power((t/r+b),n+1)
    else:
        t=t-t_sum*r
        f2=((1-n)*t/(1-r)+b)
        f3=np.power((t/(1-r)+b),n+1)
    return f1*f2/f3
if __name__ == '__main__':
    P_set=np.abs(np.random.randn(20000)*100)+2
    P_set=P_set[(P_set>2)&(P_set<100)][:10000]
    tbs_set=[i for i in range(5,181,5)]
    rain_time_set=[tbs_set[i] for i in np.random.randint(0,len(tbs_set),10000)]
    np.save("/home/hyc/flaskFiles/make_train_rain_flood_set/P_set_new_kechixu.npy",P_set)
    np.save("/home/hyc/flaskFiles/make_train_rain_flood_set/tbs_set_new_kechixu.npy",tbs_set)
    output_rainfall_set=np.zeros((10000,36),dtype=float)
    for i in range(len(P_set)):
        P=P_set[i]
        gap=5
        tbs=[i for i in range(0,rain_time_set[i],gap)]
        sum_rainfall=0
        ii=0
        for tb in tbs:
            # tb=120*0.405
            temp=rainfall_intensity_formula(P,tb,rain_time_set[i])
            output_rainfall_set[i,ii]=temp*gap
            sum_rainfall+=temp*gap
            ii+=1
            print(temp,sum_rainfall)
    np.save("/home/hyc/flaskFiles/make_train_rain_flood_set/rainfall_set_new_kechixu.npy",output_rainfall_set)
