import numpy as np
import calendar
import matplotlib.pyplot as plt
import os
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import KFold



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


def get_stdz(dat1):

  mn = np.nanmean(dat1,axis=0,keepdims=True)
  sd = np.nanstd(dat1,axis=0,keepdims=True)
  out = (dat1 - mn)/sd

  return out,np.squeeze(mn),np.squeeze(sd)


def take_stdz(dat1,mn,sd):

  return (dat1 - mn[None])/sd[None]



def split_stdz_input(X,Y,val_id=None,K=3,periodic=False):

  nYr = X.shape[0]
  nYr_split = int(np.round(nYr/K))

  if val_id == None:

  # perioding random sampling (e.g. dacadal sampling), for single test
    if periodic == True:
      Ko = 1
      high = np.append(np.repeat(nYr_split,K-1), nYr - nYr_split*(K-1))
      tmp = np.random.randint(0,high)
      val_id = [i*nYr_split + tmp[i] for i in range(K)]
      X_val = X[val_id]; X_tr = np.delete(X,val_id,0)
      Y_val = Y[val_id]; Y_tr = np.delete(Y,val_id,0)
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

  if Ko == 1:
    X_tr,Xmn,Xsd = get_stdz(X_tr); X_val = take_stdz(X_val,Xmn,Xsd)
    Y_tr,Ymn,Ysd = get_stdz(Y_tr); Y_val = take_stdz(Y_val,Ymn,Ysd)
  else:
    Xmn = []; Xsd = []
    Ymn = []; Ysd = []
    for i in range(Ko):
      X_tr[i],Xmn_,Xsd_ = get_stdz(X_tr[i]); X_val[i] = take_stdz(X_val[i],Xmn_,Xsd_)
      Xmn.append(Xmn_); Xsd.append(Xsd_)
      Y_tr[i],Ymn_,Ysd_ = get_stdz(Y_tr[i]); Y_val[i] = take_stdz(Y_val[i],Ymn_,Ysd_)
      Ymn.append(Ymn_); Ysd.append(Ysd_)

  return X_tr, X_val, Xmn, Xsd, Y_tr, Y_val, Ymn, Ysd, Ko

