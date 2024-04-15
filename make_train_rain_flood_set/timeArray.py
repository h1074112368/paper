import datetime
import math
from datetime import timedelta
def timeArray(start,end,dt):
    startT=datetime.datetime.strptime(start,'%Y-%m-%d %H').timestamp()
    endT=datetime.datetime.strptime(end,'%Y-%m-%d %H').timestamp()
    a=[]
    for i in range(math.ceil((endT-startT)/60/60/dt)):
        d_time=datetime.datetime.fromtimestamp(startT+3600*i*dt)
        if dt==24:
            d_time-=timedelta(hours=d_time.hour)
        # startT+=3600*i
        d_time-=timedelta(minutes=d_time.minute,seconds=d_time.second,microseconds=d_time.microsecond)
        a.append((d_time).strftime('%Y-%m-%d %H:%M:%S'))
    return a
if __name__ == '__main__':
     start='2022-12-13 09'
     end='2022-12-13 10'
     out=timeArray(start,end,1) 
     print(out)  
