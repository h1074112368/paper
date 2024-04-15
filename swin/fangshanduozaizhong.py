#!/usr/bin/env Python
# coding=utf-8
import random
import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, load, dump
from osgeo import gdal, ogr, gdalconst
import multiprocessing.process
import time
from scipy.cluster import hierarchy
from matplotlib import pyplot as plt




# K均值聚类
def kMean( T, K, U):
    def calDistance(U, sam):
        disZhen = pd.DataFrame()
        for i in range(sam.shape[0]):
            disZhen = pd.concat([disZhen, np.sqrt(np.power(U - sam.iloc[i, :], 2).sum(axis=1))], axis=1)
        disZhen.columns = [i for i in range(disZhen.shape[1])]

        return disZhen

    eps = 1e-12
    e0 = 0.5
    eOld = 0
    sam = U.sample(K)
    sam = sam.reset_index(drop=True)
    d = pd.DataFrame(np.zeros([U.shape[0], K]))
    # Sel = pd.DataFrame(np.zeros([U.shape[0], U.shape[1]]))
    dSel = pd.DataFrame(np.zeros([U.shape[0], 1]))
    for t in range(T):

        # for i in range(sam.shape[0]):
        #     calDistance(U, sam)
        #     d.iloc[:, i] = (U.iloc[:] - sam.iloc[i, 0]).abs()
        d = calDistance(U, sam)
        # Sel = pd.DataFrame(np.zeros([U.shape[0], U.shape[1]]))
        dIdxMin = d.idxmin(axis=1)
        Sel = U.copy()
        dSel = pd.DataFrame(d.values[list(range(dIdxMin.shape[0])), dIdxMin])
        # for idSel in range(dIdxMin.size):
        #     Sel.iloc[idSel, :] =U.iloc[idSel,:] #d.iloc[idSel, dIdxMin.iloc[idSel]]
        #     dSel.iloc[idSel, 0] = d.iloc[idSel, dIdxMin.iloc[idSel]]
        Sel['class'] = dIdxMin
        SelGroup = Sel.groupby('class', as_index=False)
        sam = SelGroup.agg([('mean', 'mean')])
        sam.columns = sam.columns.get_level_values(0)
        e = sum(dSel.iloc[:, 0].pow(2))
        if np.abs(eOld - e) < e0:
            break
        eOld = e
        # print(e)
        # print(dSel)
    Sel['class'] += 1
    return Sel

def kMean1( T, K, U, sam):
    def calDistance(U, sam):
        disZhen = pd.DataFrame()
        for i in range(sam.shape[0]):
            disZhen = pd.concat([disZhen, np.sqrt(np.power(U - sam.iloc[i, :], 2).sum(axis=1))], axis=1)
        disZhen.columns = [i for i in range(disZhen.shape[1])]

        return disZhen

    e0 = 0.5
    eOld = 0
    # sam = U.sample(K)
    sam = sam.reset_index(drop=True)
    d = pd.DataFrame(np.zeros([U.shape[0], K]))
    # Sel = pd.DataFrame(np.zeros([U.shape[0], U.shape[1]]))
    dSel = pd.DataFrame(np.zeros([U.shape[0], 1]))
    for t in range(T):

        # for i in range(sam.shape[0]):
        #     calDistance(U, sam)
        #     d.iloc[:, i] = (U.iloc[:] - sam.iloc[i, 0]).abs()

        # Sel = pd.DataFrame(np.zeros([U.shape[0], U.shape[1]]))
        d = calDistance(U, sam)

        dIdxMin = d.idxmin(axis=1)
        Sel = U.copy()
        dSel = pd.DataFrame(d.values[list(range(dIdxMin.shape[0])), dIdxMin])
        # for idSel in range(dIdxMin.size):
        #     Sel.iloc[idSel, :] = U.iloc[idSel, :]  # d.iloc[idSel, dIdxMin.iloc[idSel]]
        #     dSel.iloc[idSel, 0] = d.iloc[idSel, dIdxMin.iloc[idSel]]
        Sel['class'] = dIdxMin
        SelGroup = Sel.groupby('class')
        sam = SelGroup.agg([('mean', 'mean')])
        sam.columns = sam.columns.get_level_values(0)
        e = sum(dSel.iloc[:, 0].pow(2))
        if np.abs(eOld - e) < e0:
            break
        eOld = e
        # print(e)
        # print(dSel)
    return Sel

# 迭代自组织数据分析算法
def ISODATAI( T, K, U, nMin, dMin, fenlie):
    def calDistance(u1, u2):

        return np.sqrt(np.power(u1 - u2, 2).sum())

    # k0 = random.randint(2, K)
    k0 = 10

    for t in range(T):
        if t == 0:
            dSel = kMean(1, k0, U)
        else:
            dSel = kMean1(1, k0, U, u)
        dSelGroup = dSel.groupby('class')
        # dSelCount=dSelGroup.agg({0: 'count'})
        u = dSelGroup.agg(
            [('mean', 'mean')])  # , ('count', 'count'), ('std', 'std')])  # .agg({0: ['mean', 'count','std']})
        u.columns = u.columns.get_level_values(0)
        uCount = dSelGroup[0].agg(
            [('count', 'count')])
        uStd = dSelGroup.agg(
            [('std', 'std')])
        # for i in range(u.shape[0]):
        #     if u.iloc[i,1]<nMin:
        # print('s',k0, u.shape[0])
        if k0 != u.shape[0]:
            k0 = u.shape[0]
        k0 = k0 - u[uCount.iloc[:, 0] < nMin].shape[0]
        u = u.drop(u[uCount.iloc[:, 0] < nMin].index)

        # print('s',k0, u.shape[0])
        # k0=k0-1
        def heBing(k0, u, uCount):
            for i in range(k0):
                for j in range(i, k0):
                    # print('g',i,j,u.shape[0],k0)
                    if i != j and calDistance(u.iloc[i, :], u.iloc[j, :]) < dMin:
                        #
                        u.iloc[i, :] = (uCount.iloc[i, 0] * u.iloc[i, :] + uCount.iloc[j, 0] * u.iloc[j, :]) / (
                                    uCount.iloc[i, 0] + uCount.iloc[j, 0])
                        # print(u.iloc[j, :].index)
                        # print(u.iloc[j, :])
                        u = u.drop(labels=u.index[j], axis=0)
                        k0 = k0 - 1
                        # print('g',k0, u.shape[0])
                        if k0 == j + 1:
                            break
                        break
            # print(k0,u.shape[0])
            return k0, u

        def fenLie(k0, u, uStd, uCount):
            for i in range(u.shape[0]):
                if uStd.iloc[i, :].max() > fenlie and uCount.iloc[i, 0] >= 2 * nMin:
                    # print('h',k0, u.shape[0])
                    k0 = k0 + 1
                    tempu = u.iloc[i, :].copy()
                    u.iloc[i, :] = u.iloc[i, :] + uStd.iloc[i, :].max()

                    # u=u.stack(level=0).append([[u.iloc[i,0]-u.iloc[i,2], 1, 1]])
                    u = pd.concat([u, pd.DataFrame([tempu - uStd.iloc[i, :].max()])], axis=0)
                    # print('h',k0, u.shape[0])
                    # u = u.append(pd.DataFrame([[u.iloc[i,0]-u.iloc[i,2], 1, 1]],columns=u.columns))
            # print(k0, u.shape[0])
            return k0, u

        if k0 >= 2 * K:
            k0, u = heBing(k0, u, uCount)
        elif k0 <= K / 2:
            k0, u = fenLie(k0, u, uStd, uCount)
        else:
            if (t + 1) % 2 == 1:
                k0, u = fenLie(k0, u, uStd, uCount)
            else:
                k0, u = heBing(k0, u, uCount)
    dSel['class'] += 1
    return dSel

# 高斯混合模式聚类
def GMM( U, K, T):
    def xiefangcha(Uzu, u, y):
        return (y * (Uzu - u).pow(2)).sum() / y.sum()

    # def gailvmidu(x,u,std):
    #     return 1/np.sqrt(2*np.pi*np.power(std,2))*np.exp(-np.power(x-u,2)/(2*np.power(std,2)))

    def gailvmidu(x, u, xiestd):
        eps = 1e-12
        xiestd = xiestd + (eps) * np.eye(xiestd.shape[0])
        # if  (np.power(2 * np.pi,len(x)*1/2) * np.power(np.linalg.det(xiestd), 1/2))==0 or np.fabs(np.linalg.det(xiestd))==0 or (np.linalg.inv(xiestd))==0:
        #     print()
        # return 1 / (np.power(2 * np.pi, len(x) * 1 / 2) * np.power(np.fabs(np.linalg.det(xiestd)), 1 / 2)) * np.exp(
        #     np.dot(-1 / 2 * (x - u), np.dot(np.linalg.inv(xiestd), (x - u).T)) + eps)
        return 1 / (np.power(2 * np.pi, x.shape[1] * 1 / 2) * np.power(np.fabs(np.linalg.det(xiestd)),
                                                                       1 / 2)) * np.exp(
            -1 / 2 * np.sum(np.dot((x - u), np.linalg.inv(xiestd)) * (x - u), axis=1) )#+ eps)

    # -1 / 2 *np.sum(np.multiply(np.mat(x - u)[0] * np.mat(xiestd).I, (np.mat(x - u))), axis=1))

    def qiwangjidabu(alpha, u, std, x):

        # ap = np.zeros([1, alpha.shape[1]])
        ap = []
        for i in range(alpha.shape[1]):
            ap.append(alpha[0, i] * gailvmidu(x, u.iloc[:, i], std[i]))
        apSum = np.sum(ap, axis=0)
        for i in range(alpha.shape[1]):
            ap[i] = ap[i] / apSum

        # if ap.sum() == 0:
        #     print()
        # ap=np.concatenate(ap,axis=1).T
        return pd.DataFrame(ap)

    alpha = np.ones([1, K])
    alpha = alpha * 1 / K
    u = np.zeros([U.shape[1], K])
    lenU = int(U.shape[0] / K)
    xiestd = []
    for i in range(K):
        if i != K - 1:
            tmp = U.iloc[i * lenU:(i + 1) * lenU, :]
        else:
            tmp = U.iloc[i * lenU:, :]
        u[:, i] = np.mean(tmp, axis=0)
        xiestd.append(tmp.cov())

    # u = U.sample(n=K).T
    # u = u.reset_index(drop=True)
    # u.columns = [i for i in range(K)]
    # xiestd = [U.cov() for i in range(K)]
    u = pd.DataFrame(u)

    # yik=[qiwangjidabu(alpha,u,std,U[0][i]) for i in range(len(U))]
    for iT in range(T):
        # yik = pd.DataFrame()
        yik = pd.DataFrame(np.zeros([len(U), K]))
        sucess = 0
        # maplist=list(map(qiwangjidabu,[alpha for i
        #     in range(len(U))], [u for i
        #     in range(len(U))], [xiestd for i
        #     in range(len(U))], [U.iloc[i, :] for i
        #     in range(len(U))],[i for i
        #     in range(len(U))]))
        # qiwangjidabu(alpha, u, xiestd, U.iloc[i, :])
        # parlist = Parallel(n_jobs=22)(
        #     delayed(qiwangjidabu)(alpha, u, xiestd, U.iloc[i, :],i) for i
        #     in range(len(U)))
        # map(qiwangjidabu(alpha, u, xiestd, U.iloc[i, :]),list(range(len(U))))
        # for i in range(len(U)):
        yik = qiwangjidabu(alpha, u, xiestd, U).T

        # for i in parlist:
        #     yik.iloc[i[1], :]=i[0]
        #     yik.iloc[i,:]=qiwangjidabu(alpha, u, xiestd, U.iloc[i, :])
        # yik.append(qiwangjidabu(alpha, u, xiestd, U.iloc[i, :]))
        # print(qiwangjidabu(alpha, u, xiestd, U.iloc[i,:]))
        # yik = pd.concat([yik, qiwangjidabu(alpha, u, xiestd, U.iloc[i, :])])
        # yik1=yik1.append(qiwangjidabu(alpha, u, std, U[0][i]))
        # yf = yik*U
        # yik = yik.reset_index(drop=True)
        # dels=[]
        for iu in range(u.shape[1] - 1, 0 - 1, -1):
            # if np.linalg.det(xiestd[iu])==0 or 1/np.linalg.det(xiestd)==0:
            #     print()
            # if (yik.iloc[:, iu]).sum()==yik.shape[0]:
            #     sucess=1
            #     break
            if (yik.iloc[:, iu]).sum() == 0:
                K = K - 1
                # dels.append(iu)
                alpha = np.delete(alpha, [iu], axis=1)
                del yik[yik.columns[iu]]
                del u[u.columns[iu]]
                del xiestd[iu]

        # if sucess==1:
        #
        #     break
        if K == 1:
            break
        for iu in range(u.shape[1]):
            # for j in range(U.shape[1]):
            #
            #     if (yik.iloc[:, iu]).sum() == 0:
            #         print()
            #     u.iloc[j, iu] = np.dot(yik.iloc[:, iu], U.iloc[:, j]) / (yik.iloc[:, iu]).sum()
            u.iloc[:, iu] = np.dot(yik.iloc[:, iu], U.iloc[:, :]) / (yik.iloc[:, iu]).sum()
            # test=np.dot(yik.iloc[:, iu], U.iloc[:, :]) / (yik.iloc[:, iu]).sum()
            # xiek = np.zeros([U.shape[1], U.shape[1]])
            # xiek=[]
            # xiek = np.zeros(U.shape[0],[U.shape[1], U.shape[1]])
            # def xiek(U,u,yik,j,iu):
            #     return np.array(U.iloc[j, :] - u.iloc[:, iu]).reshape((U.shape[1], 1)).dot(
            #         np.array(U.iloc[j, :] - u.iloc[:, iu]).reshape((1, U.shape[1]))) * yik.iloc[j, iu]
            #
            # xiek = Parallel(n_jobs=20)(
            #     delayed(xiek)(U,u,yik,j,iu) for j in range(U.shape[0]))
            # for j in range(U.shape[0]):
            #
            #     xiek.append(np.array(U.iloc[j, :] - u.iloc[:, iu]).reshape((U.shape[1], 1)).dot(
            #         np.array(U.iloc[j, :] - u.iloc[:, iu]).reshape((1, U.shape[1]))) * yik.iloc[j, iu])
            # xiek = xiek + np.array(U.iloc[j, :] - u.iloc[:, iu]).reshape((xiek.shape[0], 1)).dot(
            #     np.array(U.iloc[j, :] - u.iloc[:, iu]).reshape((1, xiek.shape[0]))) * yik.iloc[j, iu]
            tmpxie = np.array(U - u.iloc[:, iu]).T
            # for iTmp in range(U.shape[1]):
            #     tmpxie[iTmp,:]=tmpxie[iTmp,:]*yik.iloc[:,iu]
            tmpxie = tmpxie * np.dot(yik.iloc[:, iu].values.reshape([yik.shape[0], 1]), np.ones([1, u.shape[0]])).T
            xiek = np.dot(tmpxie, np.array(U - u.iloc[:, iu]))  # np.sum(xiek,axis=0)
            xiestd[iu] = xiek / (yik.iloc[:, iu]).sum()
            alpha[0, iu] = (yik.iloc[:, iu]).sum() / U.shape[0]
    yikIdxMax = yik.idxmax(axis=1)
    U['class'] = yikIdxMax + 1
    # print()

    return U

# 轮廓系数
def SilCoe( U):
    def par1(pi, ipi, U):
        si = 0
        if (ipi + 1) * pi <= U.shape[0]:
            tmpU = U.iloc[ipi * pi:(ipi + 1) * pi, :]
        else:
            tmpU = U.iloc[ipi * pi:, :]
        kinds = tmpU.pop(U.columns[U.shape[1] - 1])
        kindsUni = np.unique(kinds)
        tmpUDis = np.zeros([U.shape[0], tmpU.shape[0]])
        # def mapFun(i):
        #     return np.power(
        #         tmpU.iloc[:, i].values.reshape([1, tmpU.shape[0]]) - U.iloc[:, i].values.reshape(
        #             [U.shape[0], 1]), 2)
        for i in range(U.shape[1] - 1):
            tmpUDis += np.power(
                tmpU.iloc[:, i].values.reshape([1, tmpU.shape[0]]) - U.iloc[:, i].values.reshape(
                    [U.shape[0], 1]), 2)
        tmpUDis = pd.DataFrame(np.sqrt(tmpUDis))
        for ikind in kindsUni:
            # indexKind=np.where(kinds==ikind)
            # tmpUDis[:,indexKind[0]]
            # np.where(U.iloc[:, U.shape[1] - 1] == ikind)
            tmpUDisSel = tmpUDis.iloc[:, np.where(kinds == ikind)[0]]
            UzuDis = tmpUDisSel.iloc[np.where(U.iloc[:, U.shape[1] - 1] == ikind)[0], :]
            ai = np.sum(UzuDis, axis=0) / (UzuDis.shape[0] - 1)
            # del UzuDis
            # UuzuDis=tmpUDis[np.where(U.iloc[:, U.shape[1] - 1] == ikind)[0],np.where(kinds==ikind)[0]]
            UuzuDis = tmpUDisSel.iloc[np.where(U.iloc[:, U.shape[1] - 1] != ikind)[0], :]

            bi = np.mean(UuzuDis, axis=0)
            # del UuzuDis,tmpUDisSel
            maxab = np.max(np.concatenate([[ai], [bi]]), axis=0)
            si += np.sum((bi - ai) / maxab)
        # del UzuDis,UuzuDis,tmpUDisSel
        return si

    # si=0
    pi = 150
    n_jobs = 4
    if int(np.ceil(U.shape[0] / pi)) >= 50 * n_jobs:
        n_jobs = 10
    silist = Parallel(n_jobs=n_jobs)(
        delayed(par1)(pi, ipi, U) for ipi in
        range(int(np.ceil(U.shape[0] / pi))))  # int(np.ceil(U.shape[0] / pi))))
    # si=np.sum(si)
    return np.sum(silist) / U.shape[0]

# 紧密度指数
def CHI( U):
    def wkFun(U):
        kinds = U.pop(U.columns[U.shape[1] - 1])
        kind = kinds.unique()
        wk = np.zeros([U.shape[1], U.shape[1]])
        for ikind in kind:
            Uzu = U[kinds == ikind]
            UzuMean = Uzu.mean(axis=0)
            wk += np.dot(pd.DataFrame(Uzu - UzuMean).T, pd.DataFrame(Uzu - UzuMean))
            # for j in range(Uzu.shape[0]):
            #     wk+=np.dot(pd.DataFrame(Uzu.iloc[j,:]-UzuMean),pd.DataFrame(Uzu.iloc[j,:]-UzuMean).T)
        return wk, len(kind)

    def bkFun(U):
        kinds = U.pop(U.columns[U.shape[1] - 1])
        kind = kinds.unique()
        Umean = U.mean(axis=0)
        bk = np.zeros([U.shape[1], U.shape[1]])
        for ikind in kind:
            Uzu = U[kinds == ikind]
            UzuMean = Uzu.mean(axis=0)
            bk += Uzu.shape[0] * np.dot(pd.DataFrame(UzuMean - Umean), pd.DataFrame(UzuMean - Umean).T)
        return bk

        # Uuzu = U[kinds != kind]

    wk, k = wkFun(U.copy())
    bk = bkFun(U.copy())
    return bk.trace() / wk.trace() * ((U.shape[0] - k) / (k - 1))

# 区域生长
def quYuShengZhang( shange, n):
    def find(ras, used, x, y):
        dui = []
        for i in range(3):
            if i + x - 1 < 0 or i + x - 1 >= ras.shape[0]:
                continue
            for j in range(3):
                if j + y - 1 < 0 or j + y - 1 >= ras.shape[1]:
                    continue
                if i == 1 and j == 1:
                    continue
                if used[i + x - 1, j + y - 1] == 0 and ras[i + x - 1, j + y - 1] == ras[x, y]:
                    dui.append((i + x - 1, j + y - 1))
                    used[i + x - 1, j + y - 1] = 1
        return dui, used

    duiZhan = []
    kind = []

    used = np.zeros([shange.shape[0], shange.shape[1]])
    used[np.where(shange == 0)] = 1
    for i in range(shange.shape[0]):
        for j in range(shange.shape[1]):
            if used[i, j] == 0:
                kind.append(shange[i, j])
                dui = [(i, j)]
                used[i, j] = 1
                iduiZhan = 0
                while 1:
                    duiTemp, used = find(shange, used, dui[iduiZhan][0], dui[iduiZhan][1])
                    dui += duiTemp
                    iduiZhan += 1
                    if len(dui) == iduiZhan:
                        break
                duiZhan.append(dui)
    quYu = np.zeros([len(duiZhan), 3])
    for i in range(len(duiZhan)):
        quYu[i, 0:2] = np.mean(duiZhan[i], axis=0)
        quYu[i, 2] = len(duiZhan[i])

    def calDistance(u1, u2):
        return np.sqrt(np.power(u1 - u2, 2).sum(axis=1))

    qu = np.array(range(len(duiZhan)))

    for i in range(len(duiZhan)):
        if quYu[i, 2] < n:
            d = calDistance(quYu[:, 0:2], quYu[i, 0:2])
            dSortInd = np.argsort(d)
            idSortInd = 1
            while 1:
                if quYu[dSortInd[idSortInd], 2] >= n and d[dSortInd[idSortInd]] != 0:
                    kind[i] = kind[dSortInd[idSortInd]]
                    qu[i] = qu[dSortInd[idSortInd]]
                    break
                idSortInd += 1
    return [duiZhan, kind, qu + 1]

# 层次聚类
def HieClu( U, K):
    def calDistance(u1, u2):

        return np.sqrt(np.power(u1 - u2, 2).sum(axis=1))

    def classDis(cla1, cla2):
        d = 0
        for i in range(cla1.shape[0]):
            d += (calDistance(cla2, cla1.iloc[i, :]) / (cla1.shape[0] * cla2.shape[0])).sum()
        return d

    U = (U - U.min(axis=0)) / (U.max(axis=0) - U.min(axis=0))
    classU = classU = np.array(range(U.shape[0])).T
    # classU=pd.DataFrame([range(U.shape[0])]).T
    UDis = np.zeros([U.shape[0], U.shape[0]])
    for i in range(U.shape[1]):
        UDis += np.power(
            U.iloc[:, i].values.reshape([1, U.shape[0]]) - U.iloc[:, i].values.reshape(
                [U.shape[0], 1]), 2)
    UDis = np.sqrt(UDis)
    UDis[np.where(UDis == 0)] = 999999
    UDis = pd.DataFrame(UDis)
    classUSet = []
    classUSet.append(classU.copy())

    def tmpUDisFun(UDis, i, lenclassUUni):
        tmpUDis1 = np.ones([1, lenclassUUni]) * 9999999
        for j in range(i + 1, lenclassUUni):
            # if i==j:
            #     tmpUDis[i, j]=999999
            #     continue
            ilist = np.where(classU == classUUni[i])[0]
            jlist = np.where(classU == classUUni[j])[0]
            tmpUDis1[0, j] = np.sum(UDis.iloc[ilist, jlist].values) / (len(ilist) * len(jlist))
        return [i, tmpUDis1]

    addIdx = classU.shape[0]
    cluSet = []
    while 1:
        classUUni = np.unique(classU)
        # classN=classUUni.shape[0]
        if classUUni.shape[0] == 1:
            break
        tmpUDis = np.ones([classUUni.shape[0], classUUni.shape[0]]) * 9999999

        if classUUni.shape[0] >= 20:
            parSet = Parallel(n_jobs=10)(
                delayed(tmpUDisFun)(UDis, i, classUUni.shape[0]) for i in range(classUUni.shape[0]))
        else:
            parSet = Parallel(n_jobs=classUUni.shape[0])(
                delayed(tmpUDisFun)(UDis, i, classUUni.shape[0]) for i in range(classUUni.shape[0]))
        # for i in range(classUUni.shape[0]):
        # for j in range(i+1, classUUni.shape[0]):
        #     # if i==j:
        #     #     tmpUDis[i, j]=999999
        #     #     continue
        #     ilist=np.where(classU==classUUni[i])[0]
        #     jlist=np.where(classU==classUUni[j])[0]
        #     tmpUDis[i,j]=np.sum(UDis.iloc[ilist,jlist].values)/(len(ilist)*len(jlist))
        for par in parSet:
            tmpUDis[par[0], :] = par[1]
        tmpUDis = pd.DataFrame(tmpUDis)

        tempIdxMin = tmpUDis.stack().idxmin()  # np.argmin(tmpUDis)

        cluSet.append([classUUni[tempIdxMin[0]], classUUni[tempIdxMin[1]], tmpUDis.stack().min(), addIdx])
        classU[np.where(classU == classUUni[tempIdxMin[1]])] = addIdx
        classU[np.where(classU == classUUni[tempIdxMin[0]])] = addIdx
        addIdx += 1
        classUSet.append(classU.copy())

    # print()
    # print()

    # d=pd.DataFrame(np.zeros([classUUni.shape[0],classUUni.shape[0]]),index=classUUni,columns=classUUni)
    # tmpUDis = np.zeros([U.shape[0], U.shape[0]])
    # for i in range(U.shape[1]):
    #     tmpUDis += np.power(
    #         U.iloc[:, i].values.reshape([1, U.shape[0]]) - U.iloc[:, i].values.reshape(
    #             [U.shape[0], 1]), 2)
    # tmpUDis = pd.DataFrame(np.sqrt(tmpUDis))
    # tmpUDis[tmpUDis==0]=999999
    # tmpUDis.stack().idxmin()
    # for i in range(classUUni.shape[0]):
    #     for j in range(i,classUUni.shape[0]):
    #         temp=classDis(U[classU.iloc[:,0]==classUUni[i]],U[classU.iloc[:,0]==classUUni[j]])
    #         if i!=j:
    #             d.iloc[j, i] = temp
    #             d.iloc[i, j] = temp
    #         else:
    #             d.iloc[j, i]=999999
    # dmin=pd.DataFrame(d.min(axis=1)).sort_values(by=0)
    #
    # didxmin=pd.DataFrame(d.idxmin(axis=1))
    # didx0=didxmin.loc[dmin.index]
    # classUUniUse=pd.DataFrame(np.zeros([d.shape[0],1]),index=d.index)
    # for i in range(d.shape[0]):
    #     if ((classUUniUse.loc[dmin.index[i]]).max()==1).max() or ((classUUniUse.loc[didx0.iloc[i,0]]).max()==1).max():
    #         continue
    #     classU[classU.iloc[:,0]==dmin.index[i]]=didx0.iloc[i,0]
    #     classUUniUse.loc[dmin.index[i]]=1
    #     classUUniUse.loc[didx0.iloc[i,0]] = 1
    #     classN-=1
    #     if classN==K:
    #         break
    #
    # print()
    # U['class']=classU

    return [np.array(classUSet), cluSet]




def shpToRas( refore_tif, shp_file, id, output_tiff):
    # id="地震危"
    # refore_tif = r"D:\徐汇数据\徐汇地震\2021.12.10徐汇地震危险性数据汇交\地震场址影响系数（栅格）\徐汇场地类别.tif"
    # refore_tif='testClip1.tif'
    # shp_file = r"D:\徐汇数据\徐汇地震\2021.12.10徐汇地震危险性数据汇交\地震危险性等级图（矢量）\徐汇区地震危险性等级点矢量.shp"
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(shp_file)
    shp_layer = data_source.GetLayer()
    img = gdal.Open(refore_tif)
    projection = img.GetProjection()
    transform = img.GetGeoTransform()
    cols = img.RasterXSize
    rows = img.RasterYSize
    img = None  # todo 释放内存，只有强制为None才可以释放干净
    del img
    # 构建临时存储tif路径
    # index_arr = str_all_index(output_tiff, '\\')
    # last_index = index_arr[len(index_arr) - 1] + 1
    # temp_output_tiff = output_tiff[:last_index] + 'temp.tif'
    # output_tiff = "test1.tif"
    # 根据模板tif属性信息创建对应标准的目标栅格
    target_ds = gdal.GetDriverByName('GTiff').Create(output_tiff, cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(transform)
    target_ds.SetProjection(projection)

    band = target_ds.GetRasterBand(1)
    # 设置背景数值
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()

    # 调用栅格化函数。gdal.RasterizeLayer函数有四个参数，分别有栅格对象，波段，矢量对象，value的属性值将为栅格值
    gdal.RasterizeLayer(target_ds, [1], shp_layer, options=["ATTRIBUTE=" + id])
    # 直接写入？？
    y_buffer = band.ReadAsArray()
    target_ds.WriteRaster(0, 0, cols, rows, y_buffer.tobytes())
    # target_ds.WriteRaster(0, 0, cols, rows, y_buffer)
    target_ds = None  # todo 释放内存，只有强制为None才可以释放干净
    del target_ds, shp_layer

def RasToRe( desTif, srcTif, width, height):
    # shp_file = r"D:\徐汇数据\徐汇地震\2021.12.10徐汇地震危险性数据汇交\地震危险性等级图（矢量）\徐汇区地震危险性等级面矢量.shp"

    ds = gdal.Warp(desTif, srcTif, format='GTiff',
                   resampleAlg=gdalconst.GRA_NearestNeighbour, dstNodata=0, width=width,
                   height=height)

def RasToClip( desTif, srcTif, shp_file):
    # shp_file = r"D:\徐汇数据\徐汇地震\2021.12.10徐汇地震危险性数据汇交\地震危险性等级图（矢量）\徐汇区地震危险性等级面矢量.shp"

    ds = gdal.Warp(desTif, srcTif, format='GTiff', cutlineDSName=shp_file,
                   resampleAlg=gdalconst.GRA_NearestNeighbour, cropToCutline=True, dstNodata=0)

def RasToReAndClip( desTif, srcTif, shp_file, width, height):
    # shp_file = r"D:\徐汇数据\徐汇地震\2021.12.10徐汇地震危险性数据汇交\地震危险性等级图（矢量）\徐汇区地震危险性等级面矢量.shp"

    ds = gdal.Warp(desTif, srcTif, format='GTiff', cutlineDSName=shp_file,
                   resampleAlg=gdalconst.GRA_NearestNeighbour, cropToCutline=True, dstNodata=0, width=width,
                   height=height)

def readTifArray( tifname):
    tif = gdal.Open(tifname)
    band = tif.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr

def craetTif( tifname, cols, rows, bandN, transform, projection, tifArray,nodata):
    target_ds = gdal.GetDriverByName('GTiff').Create(tifname, cols, rows, bandN, gdal.GDT_Float32)
    target_ds.SetGeoTransform(transform)
    target_ds.SetProjection(projection)

    band = target_ds.GetRasterBand(1)
    band.WriteRaster(0, 0, cols, rows, tifArray)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    # bandArray = band.ReadAsArray()
    # target_ds.WriteRaster(0, 0, cols, rows,tifArray)
    del target_ds

# def shiliang2shange(shpPath):
def preProcess( tifPath, shpName, tifRePath):
    filesname = os.listdir(tifRePath)
    for filename in filesname:
        srcTif = tifRePath + '/' + filename
        if os.path.exists(srcTif):
            os.unlink(srcTif)
    filesname = os.listdir(tifPath)
    for filename in filesname:
        desTif = filename.split('.')
        desTif = tifRePath + '/' + desTif[0] + 'Clip.' + desTif[1]
        if os.path.exists(desTif):
            os.unlink(desTif)
        srcTif = tifPath + '/' + filename
        shp_file = shpName  # r"D:\徐汇数据\徐汇地震\2021.12.10徐汇地震危险性数据汇交\地震危险性等级图（矢量）\徐汇区地震危险性等级面矢量.shp"
        RasToClip(desTif, srcTif, shp_file)
    filesname = os.listdir(tifRePath)
    maxcol = 0
    for filename in filesname:
        tif = gdal.Open(tifRePath + '/' + filename)
        if maxcol < tif.RasterXSize:
            maxcol = tif.RasterXSize
            maxrow = tif.RasterYSize
            maxReso = filename
        tif = None
    for filename in filesname:

        desTif = filename.split('.')
        desTif = tifRePath + '/' + desTif[0] + 'AndRe.' + desTif[1]
        if maxReso == filename:
            maxReso = desTif
        if os.path.exists(desTif):
            os.unlink(desTif)
        srcTif = tifRePath + '/' + filename
        width, height = maxcol, maxrow
        shp_file = shpName
        RasToReAndClip(desTif, srcTif, shp_file, width, height)
        if os.path.exists(srcTif):
            os.unlink(srcTif)
    return maxReso, maxcol, maxrow

def readTifSet( maxReso, path):
    filesname = os.listdir(path)
    U = []
    for filename in filesname:

        # loc=[]
        if filename == filesname[0]:
            tif = gdal.Open(maxReso)
            projection = tif.GetProjection()
            transform = tif.GetGeoTransform()
            band = tif.GetRasterBand(1)
            arr = band.ReadAsArray()
            loc = np.where(arr != 0)
            # loc=np.where(~np.isnan(arr))
        tif = gdal.Open(path + '/' + filename)
        band = tif.GetRasterBand(1)
        arr = band.ReadAsArray()
        U.append(arr[loc])

    U = np.array(U).T
    indexDel = np.where(U == 0)
    # indexDel=np.where(np.isnan(U))
    U = np.delete(U, np.unique(indexDel[0]), axis=0)
    loc = np.array(loc).T
    loc = np.delete(loc, np.unique(indexDel[0]), axis=0)
    U = pd.DataFrame(U)
    return U, loc, projection, transform

def cluWriteTif( result, maxrow, maxcol, desTifName, transform, projection, loc):
    outTifArray = np.zeros([maxrow, maxcol], dtype=int)
    outTifArray[(loc[:, 0], loc[:, 1])] = result['class'].values
    craetTif(desTifName, maxcol, maxrow, 1, transform, projection, outTifArray)

def cluTifRead( srcName, maxReso, path):
    arr = readTifArray(srcName)
    U, loc, projection, transform = readTifSet(maxReso, path)
    U['class'] = arr[(loc[:, 0], loc[:, 1])]
    return U

def quYuShengZhangWriteTif( result, quTif, kindtif, maxrow, maxcol, transform, projection):
    outTifKindArray = np.zeros([maxrow, maxcol], dtype=int)
    outTifQuArray = np.zeros([maxrow, maxcol], dtype=int)
    for i in range(len(result[0])):
        tmplist = np.array(result[0][i])
        outTifKindArray[tmplist[:, 0], tmplist[:, 1]] = result[1][i]
        outTifQuArray[tmplist[:, 0], tmplist[:, 1]] = result[2][i]
    craetTif(kindtif, maxcol, maxrow, 1, transform, projection, outTifKindArray)
    craetTif(quTif, maxcol, maxrow, 1, transform, projection, outTifQuArray)

def DandianZongheWriteTif( result, maxrow, maxcol, desTifName, transform, projection, loc):
    outTifArray = np.zeros([maxrow, maxcol], dtype=int)
    outTifArray[(loc[:, 0], loc[:, 1])] = result
    craetTif(desTifName, maxcol, maxrow, 1, transform, projection, outTifArray)

def QuyuZongheWriteTif( result, maxrow, maxcol, desTifName, transform, projection, loc):
    outTifArray = np.zeros([maxrow, maxcol], dtype=int)
    outTifArray[loc] = result
    craetTif(desTifName, maxcol, maxrow, 1, transform, projection, outTifArray)

def dengjitu(U):
    Udj = np.zeros([U.shape[0], U.shape[1]])
    Unp = np.array(U)
    hsSet = []
    for i in range(U.shape[1]):
        # np.where(Unp[:,i]>0 and Unp[:,i]<0.3)
        hsTemp = np.zeros([maxrow, maxcol], dtype=int)

        Udj[np.where((Unp[:, i] > 0) & (Unp[:, i] < 0.3))[
                0], i] = 3  # Unp[np.where((Unp[:,i]>0) & (Unp[:,i]<0.3))[0],i]=1
        hsTemp[loc[np.where((Unp[:, i] > 0) & (Unp[:, i] < 0.3))[0], 0], loc[
            np.where((Unp[:, i] > 0) & (Unp[:, i] < 0.3))[0], 1]] = 3
        Udj[np.where((Unp[:, i] >= 0.3) & (Unp[:, i] < 0.6))[0], i] = 2  # Unp[
        hsTemp[loc[np.where((Unp[:, i] >= 0.3) & (Unp[:, i] < 0.6))[0], 0], loc[
            np.where((Unp[:, i] >= 0.3) & (Unp[:, i] < 0.6))[0], 1]] = 2
        # np.where((Unp[:, i] >= 0.3) & (Unp[:, i] < 0.6))[0], i] = 2
        Udj[np.where((Unp[:, i] >= 0.6) & (Unp[:, i] <= 1))[0], i] = 1  # Unp[
        hsTemp[loc[np.where((Unp[:, i] >= 0.6) & (Unp[:, i] <= 1))[0], 0], loc[
            np.where((Unp[:, i] >= 0.6) & (Unp[:, i] <= 1))[0], 1]] = 1
        hsSet.append(hsTemp.copy())
        # np.where((Unp[:, i] > 0.6) & (Unp[:, i] <= 1))[0], i] = 3
    for i in range(len(hsSet)):
        desTifName = 'hs' + str(i) + '.tif'
        craetTif(desTifName, maxcol, maxrow, 1, transform, projection, hsSet[i])
        # newyingxiang.hsWriteTif(hsSet[i], maxrow, maxcol, desTifName, transform, projection,loc)
    return Udj,hsSet

def dandianzonghe(Udj, maxrow, maxcol, desTifName, transform, projection, loc):
    Udj3 = np.sum(np.where(Udj == 3, 1, 0), axis=1)
    Udj2 = np.sum(np.where(Udj == 2, 1, 0), axis=1)
    Udj1 = np.sum(np.where(Udj == 1, 1, 0), axis=1)
    DandianZonghe = np.ones([Udj3.shape[0]]) * 4
    DandianZonghe[np.where(Udj1 >= 2)] = 1
    DandianZonghe[np.where(Udj1 == 1)] = 2
    DandianZonghe[np.where((Udj1 == 0) & (Udj2 >= 1))] = 3

    DandianZongheWriteTif(DandianZonghe, maxrow, maxcol, desTifName, transform, projection, loc)

def QuyuZongheFenlei(quTif,hsSet, maxcol, maxrow,  transform, projection):
    tif = readTifArray(quTif)
    quun = np.unique(tif)
    quYhFl = np.zeros((tif.shape[0], tif.shape[1]), dtype=int)
    quYhFlSet = []
    hsi = 0
    for hs in hsSet:
        for iqu in quun:
            if iqu == 0:
                continue
            else:
                loc = np.where(tif == iqu)
                hsTemp = hs[loc]
                Ph1 = np.sum(np.where(hsTemp == 1, 1, 0)) / loc[0].shape[0]
                Ph2 = np.sum(np.where(hsTemp == 2, 1, 0)) / loc[0].shape[0]
                # Ph1=np.sum(h1Tif[loc])/loc.shape[0][0]
                # Ph2 = np.sum(h2Tif[loc]) / loc.shape[0][0]
                if (Ph1 >= 0.2) and (Ph1 + Ph2 >= 0.5):
                    quYhFl[loc] = 1
                elif (Ph1 >= 0.1 and Ph1 < 0.2) and (Ph1 + Ph2 >= 0.35 and Ph1 + Ph2 < 0.5):
                    quYhFl[loc] = 2
                elif (Ph1 < 0.1):
                    quYhFl[loc] = 3

        quYhFlSet.append(quYhFl.copy())
        desTifName = 'quYhFl' + str(hsi) + '.tif'
        hsi += 1
        craetTif(desTifName, maxcol, maxrow, 1, transform, projection,
                              quYhFl)  # DandianZongheWriteTif(DandianZonghe, maxrow, maxcol, desTifName, transform, projection, loc)
    return quYhFl,quYhFlSet

def QuyuZongheFenji(quYhFl, quYhFlSet, maxrow, maxcol, desTifName, transform, projection):
    loc = np.where(quYhFl != 0)
    HS1 = np.zeros([loc[0].shape[0]])
    HS2 = np.zeros([loc[0].shape[0]])
    for quYhFl in quYhFlSet:
        temp = quYhFl[loc]
        HS1 += np.where(temp == 1, 1, 0)
        HS2 += np.where(temp == 2, 1, 0)
    QuyuZonghe = np.ones([HS2.shape[0]]) * 4
    QuyuZonghe[np.where(HS1 >= 2)] = 1
    QuyuZonghe[np.where(HS1 == 1)] = 2
    QuyuZonghe[np.where((HS1 == 0) & (HS2 >= 1))] = 3

    QuyuZongheWriteTif(QuyuZonghe, maxrow, maxcol, desTifName, transform, projection, loc)
def shpReadTable(shpName):
    driver = ogr.GetDriverByName("ESRI Shapefile")

    data_source = driver.Open(shpName, 0)  # 0读，1读写
    shp_layer = data_source.GetLayer()
    tmp = data_source.GetLayer(0)
    featureSet = []


    for feature in tmp:
        # newLayer.CreateFeature(feature)
        # newFeature.SetField('test',1)
        # data_source.GetLayer(0):
        keys = feature.keys()
        values = []
        # feature.SetField('test1',1)
        loc=feature.GetGeometryRef()
        cent=loc.Centroid()
        values.append(cent.GetX())
        values.append(cent.GetY())
        for key in keys:
            value = feature.GetField(key)
            values.append(value)
        featureSet.append(values)
    featureSet = pd.DataFrame(featureSet, columns=['Longitude','Latitude']+keys)
    return featureSet
def shpAddCol(shpName,col,colName):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    data_source = driver.Open(shpName, 1)  # 0读，1读写
    shp_layer = data_source.GetLayer()
    tmp = data_source.GetLayer(0)
    keys = tmp[0].keys()
    if colName  in  keys:
        tmp.DeleteField(tmp.FindFieldIndex(colName, 1))

    zhzs = ogr.FieldDefn(colName, ogr.OFTReal)
    tmp.CreateField(zhzs, 1)

    for feature ,coli in zip(tmp,col):
        # newLayer.CreateFeature(feature)
        feature.SetField(colName, coli)
        # feature.GetField('test')
        tmp.SetFeature(feature)
def Projection2ImageRowCol(loc,adfGeoTransform):
    try:
        dProjX=loc[:,0]
        dProjY = loc[:, 1]


        # dTemp = adfGeoTransform[1] * adfGeoTransform[5] - adfGeoTransform[2] * adfGeoTransform[4]
        x_orgigin=adfGeoTransform[0]
        y_orgigin=adfGeoTransform[3]
        plx_wid=adfGeoTransform[1]
        plx_hei = adfGeoTransform[5]
        dCol = 0.0
        dRow = 0.0
        dCol=(dProjX-x_orgigin)/plx_wid
        dRow = (dProjY - y_orgigin) / plx_hei
        # dCol = (adfGeoTransform[5] * (dProjX - adfGeoTransform[0])-adfGeoTransform[2] * (dProjY - adfGeoTransform[3])) / dTemp
        # dRow = (adfGeoTransform[1] * (dProjY - adfGeoTransform[3])-adfGeoTransform[4] * (dProjX - adfGeoTransform[0])) / dTemp

        # iCol = np.floor(dCol)
        # iRow = np.floor(dRow)
        iRow=dRow.astype(np.int32)
        iCol=dCol.astype (np.int32)
    except:
        print('false')
    else:
        return iRow,iCol
def mulGetValues(tifname,loc):
    tif = gdal.Open(tifname)
    # projection = tif.GetProjection()
    transform = tif.GetGeoTransform()
    locImage=Projection2ImageRowCol(loc.values,transform)
    TifArray=readTifArray(tifname)
    values=TifArray[locImage]

    return values


# featureSet = pd.DataFrame(featureSet, columns=keys)

if __name__ == '__main__':

    # ##可选 shp转tif
    # shpPath=''#矢量文件
    # demoTif=''#tif文件
    # id=''#提取矢量属性名
    # output_tiff=''#输出文件
    # newyingxiang.shpToRas(shpPath,demoTif,id,output_tiff)
    ##标准化
    dz_dfzjsd_shp=r'F:\qq\第六次软件考试数据\02重点隐患多灾种_房山区\02重点隐患多灾种\房山数据\地震_动峰值加速度_50年超越概率10%基岩PGA\地震_动峰值加速度_50年超越概率10%基岩PGA.shp'
    # table=shpReadTable(dz_dfzjsd_shp)
    # shpAddCol(dz_dfzjsd_shp, table['F'], 'colName')
    # table = shpReadTable(dz_dfzjsd_shp)
    cankao_tif=r'F:\qq\第六次软件考试数据\02重点隐患多灾种_房山区\02重点隐患多灾种\房山数据\地震_场址影响系数.tif'
    shpName = r"F:\qq\第六次软件考试数据\02重点隐患多灾种_房山区\02重点隐患多灾种\房山数据\地质_危险性等级\地质_危险性等级.shp"  # shp参考文件
    # table = shpReadTable(shpName)
    # shpAddCol(shpName, table['F'], 'colName')
    # table = shpReadTable(shpName)
    id='F'#需要提取字段
    Output_dz_dfzjsd_tif='dz_dfzjsd.tif'
    # mulGetValues(Output_dz_dfzjsd_tif, table.iloc[:,:2])
    shpToRas(cankao_tif,  dz_dfzjsd_shp, id, Output_dz_dfzjsd_tif)
    table = shpReadTable(dz_dfzjsd_shp)
    mulGetValues(Output_dz_dfzjsd_tif, table.iloc[:, :2])
    dz_dfzjsd_tif=Output_dz_dfzjsd_tif

    dz_czlx=r'F:\qq\第六次软件考试数据\02重点隐患多灾种_房山区\02重点隐患多灾种\房山数据\地震_场址影响系数.tif'
    dz_yhzs=readTifArray(dz_dfzjsd_tif)*0.6+readTifArray(dz_czlx)*0.4
    import matplotlib.patches as mpatches
    # fig, ax = plt.subplots()
    test=readTifArray(dz_czlx)
    test=test.astype(np.int)
    # test[test==0]=np.NaN

    im=plt.imshow(test,cmap='bwr_r',vmin=1)
    values = np.array([1,2, 3])
    colors = [im.cmap(im.norm(value)) for value in values]
    # colors[0]=[0,0,0,0]
    # colors = [im.cmap(im.norm(value)) for value in values]
    labels = ['1', '2', '3']
    patches = [
        mpatches.Patch(color=colors[i], label=labels[i])
        for i in range(len(values))]
    plt.legend(handles=patches, bbox_to_anchor=(1, 0.2), loc=2,
               borderaxespad=0., frameon=False)

    # plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.gcf().subplots_adjust(right=0.8)
    plt.savefig('test' + '_plot.png', dpi=300)
    plt.show()

    dz_fxxdj_shp = r'F:\qq\第六次软件考试数据\02重点隐患多灾种_房山区\02重点隐患多灾种\房山数据\地震_动峰值加速度_50年超越概率10%基岩PGA\地震_动峰值加速度_50年超越概率10%基岩PGA.shp'
    shpToRas(cankao_tif, dz_fxxdj_shp, id, Output_dz_dfzjsd_tif)
    dz_dfzjsd_shp = r'F:\qq\第六次软件考试数据\02重点隐患多灾种_房山区\02重点隐患多灾种\房山数据\地震_动峰值加速度_50年超越概率10%基岩PGA\地震_动峰值加速度_50年超越概率10%基岩PGA.shp'

    dz_cz = r'F:\qq\第六次软件考试数据\02重点隐患多灾种_房山区\02重点隐患多灾种\房山数据\地震_场址影响系数.tif'
    tifRePath = 'F:/隐患/testqx'  # tif重采样路径
    maxResoFile, maxcol, maxrow = newyingxiang.preProcess(tifPath, shpName, tifRePath)

    ##聚类
    ##读取tif数据集
    tifRePath = 'F:/隐患/testqx'  # tif重采样路径
    U, loc, projection, transform = newyingxiang.readTifSet(maxResoFile, tifRePath)
    #等级图生成
    Udj,hsSet=newyingxiang.dengjitu(U)
    #单点综合等级判定
    desTifName = 'DandianZonghe.tif'#输出单点综合等级图
    newyingxiang.dandianzonghe(Udj, maxrow, maxcol, desTifName, transform, projection, loc)


    yh = yinHuan()  # 新建对象

    now = time.time()
    T = 100  # 迭代次数
    K = 6  # 分类数
    result = yh.kMean(T, K, U.copy())  # k均值聚类
    desTifName = 'Kmean.tif'  # 输出聚类tif
    newyingxiang.cluWriteTif(result, maxrow, maxcol, desTifName, transform, projection, loc)
    print(time.time() - now)
    now = time.time()
    nMin = 100  # 最小样本数阈值
    dMin, fenlie = 6, 6  # 合并类间距离阈值，分裂标准差阈值
    T = 100  # 迭代次数
    K = 6  # 分类数
    result = yh.ISODATAI(T, K, U.copy(), nMin, dMin, fenlie)#自迭代组织
    desTifName = 'ISODATAI.tif'  # 输出聚类tif
    newyingxiang.cluWriteTif(result, maxrow, maxcol, desTifName, transform, projection,loc)
    print(time.time()-now)
    now = time.time()
    result = yh.GMM(U.copy(), K, T)#高斯混合模式聚类
    desTifName = 'GMM.tif'  # 输出聚类tif
    newyingxiang.cluWriteTif(result, maxrow, maxcol, desTifName, transform, projection,loc)
    print(time.time()-now)
    #评价
    srcName='GMM.tif'#聚类tif
    result=newyingxiang.cluTifRead(srcName,  maxResoFile, tifRePath)
    print('CHI:',yh.CHI(result.copy()))#紧密度指数
    print('SilCoe:',yh.SilCoe(result.copy()))#轮廓系数
    # 区域自生长
    srcName = 'GMM.tif'  # 聚类tif
    n = 300  # 最小合并单元栅格数
    outTifArray = newyingxiang.readTifArray(srcName)
    result = yh.quYuShengZhang(outTifArray, n)
    quTif = 'quYuSZqu.tif'  # 输出区tif文件
    kindtif = 'quYuSZkind.tif'  # 输出种类tif文件
    newyingxiang.quYuShengZhangWriteTif(result, quTif, kindtif, maxrow, maxcol, transform, projection)
    #区域综合致灾隐患分类
    quYhFl,quYhFlSet=newyingxiang.QuyuZongheFenlei(quTif,hsSet, maxcol, maxrow,  transform, projection)
    #区域综合致灾隐患分级
    desTifName = 'QuyuZonghe.tif'#输出区域综合致灾隐患分级图
    newyingxiang.QuyuZongheFenji(quYhFl,quYhFlSet, maxrow, maxcol, desTifName, transform, projection)



