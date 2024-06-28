import os, random
import numpy as np
import scipy.io as sio
from scipy import sparse
from utils import normalize
import h5py

def load_data(data_name):
    X_list = []
    Y = None

    if data_name in ['synthetic3d']:
        mat = sio.loadmat('./data/synthetic3d.mat')
        X = mat['X']
        X_list.append(normalize(X[0][0].astype('float32')))
        X_list.append(normalize(X[1][0].astype('float32')))
        X_list.append(normalize(X[2][0].astype('float32')))
        Y = np.squeeze(mat['Y']).astype('int')

    return X_list, Y

