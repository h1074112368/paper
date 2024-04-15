import datetime
def timeArray(start,end,dt):
    startT=datetime.datetime.strptime(start,'%Y-%m-%d %H').timestamp()
    endT=datetime.datetime.strptime(end,'%Y-%m-%d %H').timestamp()
    a=[]
    for i in range(int((endT-startT)/60/60/dt)):
        d_time=datetime.datetime.fromtimestamp(startT+3600*i*dt)
        # startT+=3600*i
        a.append((d_time).strftime('%Y-%m-%d_%H_%M_%S'))
    return a
if __name__ == '__main__':
     start='2022-12-13 09'
     end='2022-12-14 09'
     out=timeArray(start,end,1) 
     print(out)  
