import os
import re
import sys
import sklearn as skl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
import glob
import dataproc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from data_grid import DataGrid
# from data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
# from peak_removal import peak_rem
# from scipy.stats import pearsonr
# from scipy.stats import spearmanr
# from scipy.spatial.distance import jensenshannon as jsd




def Agglo_cluster(X,a=11,b=12,deal = "None",metric="None"):
    
    '''
    Performs Agglomerative clustering on the dataset passed using different metrics
    
    Parameters
    -----------------
    
    X  : Input dataset (m_samples,n_features)
    
    
    deal : int, "None", "log", "fft"
        Dataset transformation before passing to agglomerative clustering
        
        int       : power to raise the dataset
        "None"    : no transformation
        "log"     : take natural log of dataset
        "square"  : square the dataset
        "fft"     : take fourier transform of the dataset
        
    a  : int
        Number of clusters (start)
       
    b  : int
        Number of clusters (end)
        
    
    
    metric : "None", "cosine", "Manhattan", "sEuclidean", "Euclidean", "JSD", "SC", "PC", "AveSim", "AveDist"
        Desired dissimilarity or distance measure
        
        "None"         : mean of the absolute difference between two datapoint
        "cosine"       : cosine dissimilarity
        "Manhattan"    : Manhattan distance measure
        "sEuclidean"   : squared Euclidean distance measure
        "Euclidean"    : Euclidean distance measure
        "JSD"          : Jensen-Shannon Divergence dissimilarity measure
        "SC"           : Spearman Correlation 
        "PC"           : Pearson Corrlation
        "AveSim"       : Average of cosine and Pearsonn
        "AveDist"      : Average of Euclidean, squared Eucldiean, and None
        
    Returns
    -----------------
    Cluster grid(s) with the provided number of clusters
    
    labels : array (m_samples,)
        Labels of each sample in the dataset
    
    '''
    
    
    
    #dataGrid = DataGrid_TiNiSn_500C()
   

    
    def similarity(d1,d2):
        
        a = X[d1-1]
        b = X[d2-1]
        if deal == 'log':
            a = np.log(a+1)
            b = np.log(b+1)
            
        if deal == 'fft':
            a = np.fft.fft(a).real
            b = np.fft.fft(b).real
            
        
        if deal!='log' and deal!='None' and deal!='fft':
            power = deal
            a = np.power(a,power)
            b = np.power(b,power)
            
         
       
        if metric == 'JSD':
            corr = jsd(a,b)
            #corr = np.square(jsd(a,b))
            return corr
        
        if metric == 'PC':
            corr, _ = pearsonr(a,b)
            return 1-corr
        
        if metric == 'SC':
            corr, _ = spearmanr(a,b)
            return 1-corr
        
        if metric == 'cosine':
            dot_product = np.dot(b, a)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            return 1 - (dot_product / (norm_a * norm_b))
            #return np.abs((dot_product / (norm_a * norm_b)))
        
        if metric == 'AveSim':
            corr1, _ = pearsonr(a,b)
            corr1 = 1 - corr1
            dot_product = np.dot(b, a)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            corr2 = (dot_product / (norm_a * norm_b))
            corr2 = 1 - corr2
            #corr3, _ = spearmanr(a,b)
            #corr3 = 1-corr3
            corr = np.mean([corr1,corr2])
            #corr = 1 - np.mean([corr1,corr2]) 
            
            return corr 
        
        if metric == 'Euclidean':
            return np.sqrt(np.sum(np.square(b-a)))
        if metric == 'sEuclidean':
            return np.sum(np.square(b-a))
        if metric == 'Manhattan':
            return np.sum(np.abs(b-a))
        if metric == 'None':
            return np.mean(np.abs(b-a))
        
        if metric == 'AveDist':
            corr = np.mean([np.sqrt(np.sum(np.square(b-a))),np.sum(np.square(b-a)),np.sum(np.abs(b-a))])
            return corr
        


    size = len(X)

   # K_Matrix = np.zeros(shape=(size,size))
    #for x in range(1,size+1):
     #   K_Matrix[x-1,x-1] = 1
      #  for N in dataGrid.neighbors(x).values():
       #     K_Matrix[x-1,N-1] = 1
        #    for N2 in dataGrid.neighbors(N).values():
         #       K_Matrix[x-1,N2-1] = 1




    #calculate i clusters and create grid visuals
    #You are to use a dissimilarity or distance metric if affinity = precomputed

    def get_cluster_grids(i):
        agg = AgglomerativeClustering(n_clusters=i,compute_full_tree = True, affinity='precomputed',linkage='complete')
        agg.fit(D)
        labels=agg.labels_
        
        
#         #This part of the code is a savior
#         new_labels=[None]*177
#         for ii in range(0,177):
#             x,y=dataGrid.coord(ii+1)
#             new_grid=dataGrid.grid_num(16-x,y)
#             new_labels[new_grid-1]=labels[ii]
           
            
        hues = [float(float(x)/float(i)) for x in range(1,i+1)]
        #hues = [0.09090909090909091,0.18181818181818182,0.2727272727272727,0.36363636363636365,0.45454545454545453,0.5454545454545454,
                #0.6363636363636364,0.7272727272727273,0.8181818181818182,0.9090909090909091,1.0]
        cluster_grid = np.zeros(shape = (15,15,3))
        for val in range(1,178):
            x,y = dataGrid.coord(val)
            cluster = labels[val-1]
            #cluster = new_labels[val-1]
            #cluster_grid[15-y][x-1] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])
            cluster_grid[y-1][x-1] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])
        
        return cluster_grid, labels    
        #return cluster_grid, new_labels
     
    
    D = np.ones(shape=(size,size))
    
    for x in range(size):
        for y in range(size):
            D[x,y] = similarity(x+1,y+1)
            
            
            
    #C = len(np.unique(D))
    #print(C)
    start = a
    end = b
    fig = plt.figure()
    fig.tight_layout()
    Big_labels = []
    for i in range(start,end):
        cg,labels = get_cluster_grids(i)
        Big_labels.append(labels)
        ax = fig.add_subplot(1,end-start,i-start+1)
        ax.imshow(cg)
        ax.invert_yaxis()
        ax.title.set_text(i)
        ax.axis("off")

    k=.03
    plt.subplots_adjust(left=k,right=(1-k),bottom=k,top=(1-k),wspace=k,hspace=k)
    #plt.savefig("/home/sasha/Desktop/Peak_Clustering_Images/clust-" + str(delta) + "-" + str(C) + ".png")
    plt.show()
    #plt.close()
    return Big_labels

