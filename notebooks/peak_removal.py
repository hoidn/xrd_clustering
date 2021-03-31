import os
import re
import sys
import sklearn as skl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
import glob
import dataproc
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
#import dataproc.operations
#from dataproc.operations.hitp import bayesian_block_finder
#from dataproc.operations.hitp import fit_peak


def peak_rem():
    
    '''
    For an XRD dataset, peak_rem() finds the pattern with the lowest number of curves i.e curves whose parameters decribe the peaks
    in the XRD pattern.
    peak_rem() returns a dictionary of all the patterns in the dataset where all the curves in each XRD pattern has been reduced to the 
    lowest number found. The curves with the least intensities are removed first.
    
    Note: You might need to change the file path. Each file in the dataset should contain all the 8 parameters for all the curves that make     up all the peaks in the XRD pattern. These parameters can be obtained from feat_extr() function.
    
    '''
    
    

    #Directory to pull files
    path = "C:/Users/oluwa/Jupyter notebooks/"
    regex = "TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D_extr_param.csv"

    # Pull data from file
    files = os.listdir(path)

    #regex to parse grid location from file
    pattern = re.compile(regex)

    if len(files) == 0:
         print("no files in path")
         sys.exit()


    #Get the needed files only

    filess=[]
    data={}
    II=0
    for file in files:
        match = pattern.match(file)
        if(match == None):
            continue
        filess.append(match)
        num = int(match.group("num"))
        #print(num)

        data[II+1]=pd.read_csv(path+file, names = ['x0/Q','y0','I','alpha','gamma','FWHM','area','area-err','X0'])
        II+=1


    #find the minimum no of peaks/intensity

    min_no_peaks = 100
    for k in data.keys():
        min_no_peaks = min(min_no_peaks,len(data[k]['I']))
    
    print(min_no_peaks)

    #Remove the least intense peaks
    params = {}
    for k in data.keys():
        if len(data[k]['I']) > min_no_peaks:
            mm = len(data[k]['I']) - min_no_peaks
            #print(mm)
            #print(k)
            subdata = data[k].sort_values('I')
            subdata = subdata.drop(subdata.index[range(0,mm)])

            subdata = subdata.sort_index()
            params[k] = subdata
        else: 
            params[k] = data[k]
            
  
            
            
    return params
        
        
       
    
    
