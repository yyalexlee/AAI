import os
import xarray as xr
import numpy as np
import pandas as pd

import calendar
import datetime

import itertools
from math import radians, degrees, sin, cos, asin, acos, sqrt
from scipy import signal
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt



def get_seasonal(data0):

  seas = np.copy(data0)
  for isea in range(12):
    data = np.copy(data0,-isea,axis=1)
    seas[:,isea] = np.nanmean(data[:,isea:isea+3],axis=1)

  return seas


def get_multimonth(data0):

  for isea in range(12):
    data = np.copy(data0,-isea,axis=1)
    data1 = data[:,isea:isea+3]
    data21 = np.nanmean(data1[:,(0,1)],axis=1,keepdims=True)
    data22 = np.nanmean(data1[:,(0,2)],axis=1,keepdims=True)
    data23 = np.nanmean(data1[:,(1,2)],axis=1,keepdims=True)
    data3 = np.nanmean(data1,axis=1,keepdims=True)
    seas_ = np.concatenate((data1,data21,data22,data23,data3),axis=1)
    dim = seas_.shape
    seas_ = np.reshape(seas_,np.concatenate((np.prod(dim[:2]),1,dim[2:])))
    if isea == 0: seas = np.copy(seas_)
    else:         seas = np.concatenate((seas,seas_),axis=1)

  return seas



def get_stdz(dat1,dat2):

  mn = np.nanmean(dat1,axis=0,keepdims=True)
  sd = np.nanstd(dat1,axis=0,keepdims=True)
  out1 = (dat1 - mn)/sd
  out2 = (dat2 - mn)/sd
  out1 = np.where(sd==0,0,out1)
  out2 = np.where(sd==0,0,out2)

  return out1,out2,mn,sd



def take_stdz(dat1,mn,sd):

  out = (dat1 - mn)/sd
  out = np.where(sd==0,0,out)

  return out



def get_ano(dat1,dat2):

  mn = np.nanmean(dat1,axis=0,keepdims=True)
  sd = np.ones(mn.shape)
  out1 = (dat1 - mn)
  out2 = (dat2 - mn)

  return out1,out2,mn,sd


def split_stdz_input(X,Y,val_id=None,K=5,periodic=False,std=True):

  nYr = X.shape[0]

  if val_id == None:

  # periodic random sampling (e.g. dacadal sampling), for multiple K test
    if periodic == True:

      nYr_split = int(np.round(nYr/K))
      Ko = K
      high = np.append(np.repeat(K,nYr_split-1), nYr - K*(nYr_split-1))
      X_tr = []; X_val = []
      Y_tr = []; Y_val = []
      for ik in range(K):
        tmp = np.random.randint(0,high)
        val_id = [i*K + tmp[i] for i in range(nYr_split)]
        X_val_ = X[val_id]; X_tr_ = np.delete(X,val_id,0)
        Y_val_ = Y[val_id]; Y_tr_ = np.delete(Y,val_id,0)
        X_tr.append(X_tr_); X_val.append(X_val_)
        Y_tr.append(Y_tr_); Y_val.append(Y_val_)

  # random k-folding (for multiple(K) test)
    else:
      Ko = K
      kf = KFold(n_splits = K, shuffle=True)
      X_tr = []; X_val = []
      Y_tr = []; Y_val = []
      i = 0
      for tr_i,val_i in kf.split(np.arange(nYr)):
        #print('K='+str(i+1),len(tr_i),len(val_i))
        X_tr_ = X[tr_i]; X_val_ = X[val_i]
        Y_tr_ = Y[tr_i]; Y_val_ = Y[val_i]
        X_tr.append(X_tr_); X_val.append(X_val_)
        Y_tr.append(Y_tr_); Y_val.append(Y_val_)
        i += 1

  # subjective sampling (e.g. 1982, 1997, 2005 etc.), for single test
  else:
    Ko = 1
    X_val = X[val_id]; X_tr = np.delete(X,val_id,0)
    Y_val = Y[val_id]; Y_tr = np.delete(Y,val_id,0)


  Xmn = []; Xsd = []
  Ymn = []; Ysd = []
  for i in range(Ko):
    if std==True:
      X_tr[i],X_val[i],Xmn_,Xsd_ = get_stdz(X_tr[i],X_val[i])
      Y_tr[i],Y_val[i],Ymn_,Ysd_ = get_stdz(Y_tr[i],Y_val[i])
    else:
      X_tr[i],X_val[i],Xmn_,Xsd_ = get_ano(X_tr[i],X_val[i])
      Y_tr[i],Y_val[i],Ymn_,Ysd_ = get_ano(Y_tr[i],Y_val[i])
    Xmn.append(Xmn_); Xsd.append(Xsd_)
    Ymn.append(Ymn_); Ysd.append(Ysd_)

  return X_tr, X_val, Xmn, Xsd, Y_tr, Y_val, Ymn, Ysd, Ko



def set_data(X_H,Y_H,X_F,Y_F,ilb):

  ds = len(X_H.shape)

  if ds==5: axis_thr = (1,2,3,4)
  else: axis_thr = (1,2,3)

  id1 = np.all(~np.isnan(X_H),axis=axis_thr)
  id2 = np.all(~np.isnan(Y_H[:,ilb]))
  ind1 = np.logical_and(id1,id2)

  id3 = np.all(~np.isnan(X_F),axis=axis_thr)
  id4 = np.all(~np.isnan(Y_F[:,ilb]))
  #ind2 = np.logical_and(id3,id4)
  ind2 = id3

  # input
  X = X_H[ind1]
  if ds==5:  tdim, xdim, ydim, ldim,zdim = X.shape
  else:      tdim, xdim, ydim, zdim = X.shape
  test_x = X_F[ind2]
  # label 
  Y = Y_H[ind1,ilb]
  test_y = Y_F[ind2,ilb]

  # Shuffling
  arr = np.arange(tdim)
  np.random.shuffle(arr)
  X = X[arr]
  Y = Y[arr]

  return(X,Y,test_x,test_y,tdim,xdim,ydim,zdim,ind2)


def get_sorted_terc(ens):

    # ens in [nens,nd] dimension
    nens = ens.shape[0]
    sorted = np.sort(ens,axis=0)
    id = int(nens/3)
    LT = (sorted[id] + sorted[id+1])/2.
    UT = (sorted[-id-1] + sorted[-id])/2.

    return UT,LT



def calc_terc(ens,fit_opt):

#    start_time = time.time()
    # ens in (nens,nlat,nlat) dimension
    dim0 = ens.shape
    nens = dim0[0]
    dm = dim0[1:]
    nd = np.product(dm)

    if fit_opt=='gamma':

      ens = np.reshape(ens,[nens,nd])

      # usinig moments
      shape,scale = get_gamma_param(ens)
      LT = [Gamma(shape[i],scale[i]).quantile(1/3) for i in range(nd)]
      UT = [Gamma(shape[i],scale[i]).quantile(2/3) for i in range(nd)]
      LT = np.reshape(LT,dm)
      UT = np.reshape(UT,dm)

    elif fit_opt=='norm':
      sd = np.nanstd(ens,axis=0)
      mn = np.nanmean(ens,axis=0)
      UT = 0.43*sd + mn
      LT = -0.43*sd + mn

    else:
      UT,LT = get_sorted_terc(ens)

#    elapsed_time = time.time() - start_time
#    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return UT, LT


def get_tcat(ens,UT,LT):

    Tcat = np.copy(ens)
    Tcat[:] = 1  #NN

    Tcat = np.where(ens-LT < 0, 0, Tcat)  #BN
    Tcat = np.where(ens-UT >= 0, 2, Tcat) #AN

    return Tcat

