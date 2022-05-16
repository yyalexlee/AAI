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
