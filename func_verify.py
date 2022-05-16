import numpy as np
import os
import calendar
from scipy import signal
from sklearn.model_selection import KFold
from datetime import date, timedelta
from numba import jit,njit
import sys
import time

@jit
def corr_coef(A,B):
    A_mA = A - np.nanmean(A)
    B_mB = B - np.nanmean(B)
    ssA = np.nansum(A_mA**2)
    ssB = np.nansum(B_mB**2)
    A_mA_m = np.ma.array(A_mA,mask=np.isnan(A_mA))
    B_mB_m = np.ma.array(B_mB,mask=np.isnan(B_mB))
    return np.ma.dot(A_mA_m,B_mB_m)/np.sqrt((ssA*ssB))

def corr_1D2ND(d1, d2):

    # d1(nt), d2(nt,nlat,nlon)
    nt = d2.shape[0]
    nd = d2.shape[1:]
    nvar = np.product(nd)
    d2 = np.reshape(d2,(nt,nvar))
    cor = [corr_coef(d1,d2[:,i]) for i in range(nvar)]
    cor = np.reshape(cor,nd)
    return cor

@jit
def corrND(d1, d2):

    # both d1, d2 should have dim of (nt,...,nlat,nlon)
    nt = d2.shape[0]
    nd = d2.shape[1:]
    nvar = np.product(nd)
    d1 = np.reshape(d1,(nt,nvar))
    d2 = np.reshape(d2,(nt,nvar))
    cor = [corr_coef(d1[:,i],d2[:,i]) for i in range(nvar)]
    cor = np.reshape(cor,nd)


@jit
def tcc(A,B):
    #centered corr_coef
    #id = ~np.isnan(A) & ~np.isnan(B)
    #A = A[id]
    #B = B[id]
    A_mA = A - np.nanmean(A)
    B_mB = B - np.nanmean(B)
    ssA = np.nansum(A_mA**2)
    ssB = np.nansum(B_mB**2)
    A_mA_m = np.ma.array(A_mA,mask=np.isnan(A_mA))
    B_mB_m = np.ma.array(B_mB,mask=np.isnan(B_mB))
    return np.ma.dot(A_mA_m,B_mB_m)/np.sqrt((ssA*ssB))

@jit
def acc(A,B):
    #uncentered corr_coef
    #id = ~np.isnan(A) & ~np.isnan(B)
    #A = A[id]
    #B = B[id]
    ssA = np.nansum(A**2)
    ssB = np.nansum(B**2)
    A_m = np.ma.array(A,mask=np.isnan(A))
    B_m = np.ma.array(B,mask=np.isnan(B))
    out = np.ma.dot(A_m,B_m)/np.sqrt((ssA*ssB))
    return out

@jit
def HR(o,f):
    # o & f: tercile  category, should be numpy array
    out = np.full(3,np.nan)
    for i in range(3):
      id = np.where(o == i)[0]
      if len(id) != 0: 
        out[i] = np.mean(f[id] == o[id])
    return(out)


@jit
def FAR(o,f):
    out = np.full(3,np.nan)
    for i in range(3):
      id = np.where(f == i)[0]
      if len(id) != 0: 
        out[i] = np.mean(f[id] == o[id])
    return(out)

@jit
def Accuracy(o,f):
    out = np.mean((o==f)*1)
    return(out)



def get_fcst_catg(p):

    srtd = np.sort(p,axis=0); ord = np.argsort(p,axis=0)

    cat_f3 = np.where(srtd[2]-srtd[0] == 0, 1, np.nan)

    cat_f2 = np.where(srtd[2]-srtd[1] == 0, ord[0], np.nan)
    cat_f2 = np.where(cat_f3 == 1., np.nan, cat_f2)

    cat_f = np.copy(ord[2])
    cat_f = np.where(cat_f3 == 1., np.nan, cat_f)
    cat_f = np.where(np.isnan(cat_f2), cat_f, np.nan)

    return(cat_f,cat_f2,cat_f3)



def hss(pb,pn,pa,obs_o):

    pb = np.array(pb)
    pn = np.array(pn)
    pa = np.array(pa)
    obs_o = np.array(obs_o)

    id = ~np.isnan(obs_o) & ~np.isnan(pa)
    s = sum(id==True)/3.
    cat_o = obs_o[id]; pb = pb[id]; pn = pn[id]; pa = pa[id]

    p = np.array([pb,pn,pa])
    cat_f, cat_f2, cat_f3 = get_fcst_catg(p)

    p1 = np.where(cat_f == cat_o, 1, np.nan)
    ox = np.logical_and(cat_f2 != cat_o, ~np.isnan(cat_f2))
    p2 = np.where(ox, 0.5, np.nan)
    p3 = np.where(cat_f3 == 1, 1/3, np.nan)

    pp = np.array([p1,p2,p3])
    point = np.nansum(pp)
    hss = (point-s)/(2*s)

    return(hss)

