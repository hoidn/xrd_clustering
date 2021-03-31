import os
import re
import sys
import sklearn as skl
import math

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import sklearn.cluster

import glob

from data_grid import DataGrid

from sklearn.decomposition import PCA

from sklearn.decomposition import NMF

from data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C




def nmf_func(a=10,deal="None",dataset=1):
    
    '''
    Non Negative Matrix Factorization (NMF) 
    Performs NMF on all of the diffraction patterns in the TiNiSn dataset
    
    Parameters
    ------------------------
    
    a : int
        Number of desired basis vectors
        
    
    deal : "None", "log", "square", "fft", "optimal"
        Dataset transformation before passing to NMF
        
        "None"    : no transformation
        "log"     : take natural log of dataset
        "square"  : square the dataset
        "fft"     : take fourier transform of the dataset
        "optimal" : discard the dataset and return the optimal W&H found from NMFk (see NMFk Julia.ipynb)
        
        
   dataset : 1,2
       Dataset desired 
       
       1  : TiNiSn_500C
       2  : TiNiSn_600C
       
       Note: If dataset=2, deal cannot be set to "fft" or "optimal"
       
       
    Returns
    ------------------------
    W & H (see NMF)
    
    '''
    
#     if dataset==1:
#         dataGrid = DataGrid_TiNiSn_500C()

#         if deal=="None":
#             data = dataGrid.get_data_array()

#         elif deal=="log":    
#             data = np.log(dataGrid.get_data_array()+1)

#         elif deal=="square":
#             data = np.square(dataGrid.get_data_array())

#         elif deal=="optimal": 
#             W=np.array(pd.read_csv('C:\\Users\\oluwa\\Jupyter notebooks\\optimal_W_matrix.csv'))
#             H=np.array(pd.read_csv('C:\\Users\\oluwa\\Jupyter notebooks\\optimal_H_matrix.csv'))


#     elif dataset==2:
    
#         dataGrid = DataGrid_TiNiSn_600C()
#         if deal=="None":
#                 data = dataGrid.get_data_array()

#         elif deal=="log":    
#             data = np.log(dataGrid.get_data_array()+1)

#         elif deal=="square":
#             data = np.square(dataGrid.get_data_array())

#         #elif deal=="optimal": 
#          #   W=np.array(pd.read_csv('C:\\Users\\oluwa\\Jupyter notebooks\\optimal_W_matrix.csv'))
#           #  H=np.array(pd.read_csv('C:\\Users\\oluwa\\Jupyter notebooks\\optimal_H_matrix.csv'))

    
    
#     data_Tr = np.transpose(data) 
            
#     model = NMF(n_components=a, init='random', random_state=0)
#     W = model.fit_transform(data_Tr)
#     H = model.components_

#     return W,H
            
    
    if dataset==1:
        dataGrid = DataGrid_TiNiSn_500C()
        if deal=="None":
            data = dataGrid.get_data_array()
            data_Tr = np.transpose(data) 
            
            model = NMF(n_components=a, init='random', random_state=0)
            W = model.fit_transform(data_Tr)
            H = model.components_
            return W,H
            
        elif deal=="log":    
            data = np.log(dataGrid.get_data_array()+1)
            data_Tr = np.transpose(data) 
            
            model = NMF(n_components=a, init='random', random_state=0)
            W = model.fit_transform(data_Tr)
            H = model.components_
            return W,H
            
        elif deal=="square":
            data = np.power(dataGrid.get_data_array(),2)
            data_Tr = np.transpose(data) 
            
            model = NMF(n_components=a, init='random', random_state=0)
            W = model.fit_transform(data_Tr)
            H = model.components_
            return W,H
        
        elif deal=="fft":
            data = np.fft.fft(dataGrid.get_data_array()).real+50000
            data_Tr = np.transpose(data) 
            
            model = NMF(n_components=a, init='random', random_state=0)
            W = model.fit_transform(data_Tr)
            H = model.components_
            return W,H
        
        elif deal=="optimal": 

            W=np.array(pd.read_csv('C:\\Users\\oluwa\\Jupyter notebooks\\optimal_W_matrix.csv'))
            H=np.array(pd.read_csv('C:\\Users\\oluwa\\Jupyter notebooks\\optimal_H_matrix.csv'))
            return W,H
            
    elif dataset==2:
        dataGrid = DataGrid_TiNiSn_600C()
        if deal=="None":
            data = dataGrid.get_data_array()
            data_Tr = np.transpose(data) 
            
            model = NMF(n_components=a, init='random', random_state=0)
            W = model.fit_transform(data_Tr)
            H = model.components_
            return W,H
            
        elif deal=="log":    
            data = np.log(dataGrid.get_data_array()+1)
            data_Tr = np.transpose(data) 
            
            model = NMF(n_components=a, init='random', random_state=0)
            W = model.fit_transform(data_Tr)
            H = model.components_
            return W,H
            
        elif deal=="square":
            data = np.power(dataGrid.get_data_array(),2)
            data_Tr = np.transpose(data) 
            
            model = NMF(n_components=a, init='random', random_state=0)
            W = model.fit_transform(data_Tr)
            H = model.components_
            return W,H
            
        

 