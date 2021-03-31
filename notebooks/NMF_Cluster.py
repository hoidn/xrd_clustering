import os
import re
import sys
import glob
import sklearn as skl
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.cluster
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

from data_grid import DataGrid
from data_grid_TiNiSn import DataGrid, DataGrid_TiNiSn_500C, DataGrid_TiNiSn_600C
from NMF import nmf_func
from scipy.signal import find_peaks
from scipy.stats import pearsonr


def nmf_cluster(a=10,b=11,deal="None",use_peak=1,visualize=1,pear_threshold=0.8,peak_threshold=3,contri_threshold=0.1,dataset=1):
    
    '''
    Performs NMF clustering on the passed dataset and checks for peak shifting
    
    
    Parameters
    ------------------------
    
    a  : int
       Number of clusters (start)
       
    b  : int
       Number of clusters (end)
        
    
    deal : "None", "log", "square", "fft", "optimal"
        Dataset transformation before passing to NMF
        
        "None"    : no transformation
        "log"     : take natural log of dataset
        "square"  : square the dataset
        "fft"     : take fourier transform of the dataset
        "optimal" : discard the dataset and return the optimal W&H found from NMFk (see NMFk Julia.ipynb)
          
    
    use_peak : 0,1
        Check for the same number of peaks for highly correlated basis vectors?
        
        0 : no
        1 : yes
        
    
    visualize : 0,1
        Plot the basis vectors obtained from NMF?
        
        0 : no
        1 : yes
        
    
    pear_threshold : float64, range(0,1)
        Threshold for Pearson correlation
        Recommended: 0.7, 0.8, 0.9
        
        
    peak_threshold : float64
        Threshold for finding the number of peaks
        
        
    contri_threshold : float64, range(0,1)
        Threshold for the minimum conribution of phases at a composition point on the wafer map
        Recommended: 0.08, 0.1, 0.15, 0.25, 0.3

    
    dataset : 1,2
        Dataset desired 

        1  : TiNiSn_500C
        2  : TiNiSn_600C
        
    Returns 
    ------------------------
    Cluster grid(s) with the provided number of clusters
    
    labels : array (m_samples,)
        Labels of each sample in the dataset

    '''
    
    
    
     
    # Visualize all the patterns for the different cluster no.
    def clust_visuals(a,b,deal,dataset=1):
        off_set = 0
        fig,ax = plt.subplots(figsize=(8,10))
        
        if deal=="None":
            for j in range(a,b):
                W,H = nmf_func(j,deal,dataset)
                for i in range(np.shape(W)[1]):
 
                    ax.plot(dataGrid.data[1][:,0], W[:,i]+off_set)

                ax.annotate(j,(3.5,off_set+50))
                off_set+=100
            ax.set_title('Diffraction Clusters from W matrix')
        
        elif deal=="log":
            for j in range(a,b):
                W,H = nmf_func(j,deal,dataset)
                for i in range(np.shape(W)[1]):
                  
                    ax.plot(dataGrid.data[1][:,0], W[:,i]+off_set)

                ax.annotate(j,(3.5,off_set+np.log(50)))
                off_set+=np.log(100)
            ax.set_title('Diffraction Clusters from W matrix')
            
        elif deal=="square":
            for j in range(a,b):
                W,H = nmf_func(j,deal,dataset)
                for i in range(np.shape(W)[1]):

                    ax.plot(dataGrid.data[1][:,0], W[:,i]+off_set)

                ax.annotate(j,(3.5,off_set+np.square(50)))
                off_set+=np.square(100)
            ax.set_title('Diffraction Clusters from W matrix')
            
        elif deal=="fft":
            for j in range(a,b):
                W,H = nmf_func(j,deal,dataset)
                for i in range(np.shape(W)[1]):

                    ax.plot(dataGrid.data[1][:,0], W[:,i]+off_set)

                ax.annotate(j,(3.5,off_set+np.square(50)))
                off_set+=100
            ax.set_title('Diffraction Clusters from W matrix')
         
        elif deal=="optimal":
            for j in range(a,b):
                W,H = nmf_func(j,deal,dataset)
                for i in range(np.shape(W)[1]):

                    ax.plot(dataGrid.data[1][:,0], W[:,i]+off_set)

                ax.annotate(j,(3.5,off_set+50))
                off_set+=100
            ax.set_title('Diffraction Clusters from W matrix')
                              
                              
                            
    # Check for clusters with same number of peaks
    # Took time to finally write this code 

    def peaks_func (W,H,peak_threshold):
         
        # Visualize the clusters and their peaks
        peaks_length=[]
        for i in range(0,len(H)):
            peaks, _ = find_peaks(W[:,i], height=peak_threshold)
            peaks_length.append(len(peaks))
            #fig,ax = plt.subplots()
            #ax.plot(dataGrid.data[1][:,0],W[:,i])
            #ax.plot((peaks*0.005)+1.5, W[peaks,i], "x")
            #ax.set_title(str(i+1))

        #print(peaks_length)
                              
        same_peaks=[]
        for i in range(0,len(peaks_length)-1):
            smpeaks=[]

            if peaks_length[i] in list(peaks_length[:i]):
                #print(i)
                continue

            #print('Yes')
            for j in range(i+1,len(peaks_length)):
                    if peaks_length[i]==peaks_length[j]:
                        if len(smpeaks)==0:
                            smpeaks.append(i)
                        smpeaks.append(j)

            if len(smpeaks)!=0:
                same_peaks.append(smpeaks)

        #print(same_peaks)
#         for i in range(0,len(same_peaks)):
#             fig,ax = plt.subplots()
#             for j in same_peaks[i]:
#                 ax.plot(dataGrid.data[1][:,0],W[:,j])
#             ax.set_title('Same No. of Peaks '+str(same_peaks[i]))
        return same_peaks
                              
    
    def Pearson_func(W,H,pear_threshold):
                              
        # Pearson Correlation for each cluster against another
        p_corr = np.zeros((len(H),len(H)))

        for i in range(0,len(H)):
            for j in range(0,len(H)):
                corr, _ = pearsonr(W[:,i],W[:,j])
                p_corr[i,j]=corr


        # Find correlations higher than a certain threshold
        corr_idx = [[j,i] for j in range(len(p_corr)) for i in range(j+1,len(p_corr)) if np.abs(p_corr[j][i])>=pear_threshold]

            
        #print(corr_idx)
        #for i in range(0,len(corr_idx)):
         #   fig,ax = plt.subplots()
          #  for j in corr_idx[i]:
           #     ax.plot(W[:,j])
            #ax.set_title('Pearson Corr. '+ str(corr_idx[i]))
                              
                              
        return corr_idx 
    
    def peak_pearson_match(corr_idx,same_peaks):
        #Check for clusters with high correlation and same peaks
        peak_shift = []
        for i in range(len(corr_idx)):
            for j in range(len(same_peaks)):
                if corr_idx[i][0] in same_peaks[j] and corr_idx[i][1] in same_peaks[j]:
                    print('.........Found Peak and Pearson match!!!..........')
                    peak_shift.append(corr_idx[i])
            
        return peak_shift
    
                             
                              
                              
    def reshape_func(H,use_idx):
       
                              
        #Combine the weights of all the peakshifted ones in H and also delete the peakshifted phases leaving the first phase in W
        #corr_idx=[[1,6],[2,3,4],[5,7]]
                              
        H_reshaped1=np.zeros(shape=(len(H),177))
        for i in range(len(H)):
            H_reshaped1[i]=H[i]



        #H_reshaped1= H_reshaped  #this code will modify H_reshaped as H_reshaped1 is modified: very weird!!!!!!
        
                              
        #This is done outside this function
        #use_idx =[]
        #for i in range(len(peak_shift)):
            #use_idx.append(peak_shift[i])


        for i in range(len(use_idx)):
            #print(i)
            b0 = H[use_idx[i][0]]

            for j in range(len(use_idx[i])):
                if j==0:
                    #print(j)
                    #print(b0)
                    continue

                #print(j) 
                #if np.all(b0==H_reshaped[1]):
                 #   print('Yes')

                #print(b0)

                b0=b0 + H[use_idx[i][j]]


            H_reshaped1[use_idx[i][0]]=b0



        idx=[]
        for i in range(len(use_idx)):
            idx = idx + list(use_idx[i][1:])
        #idx = [i+1 for i in idx]
        print('Here are the clusters indices taken out', idx)
        new_H = np.delete(H_reshaped1,idx,0)
        print('Former H is of shape', np.shape(H_reshaped1))
        print('new_H is of shape', np.shape(new_H)) 
        
        return new_H
    
    
    
    def contributions(new_H,contri_threshold):
    
        #Reshape new_H cutting off those clusters contributing below a threshold

        H_reshape = np.zeros(shape=(np.shape(new_H)))

        for j in range(0,177):
            p = new_H[:,j]/np.sum(new_H[:,j])
            for i in range(len(p)):
                if p[i]<contri_threshold:
                    p[i]=0
            H_reshape[:,j]=p


        #Reshape the H where the 3 most abundant clusters remain for each datapoint/dot

        Big_locations=[]
        H_reshaped = np.zeros(shape=(np.shape(new_H)))         

            
        for j in range(0,177):            
            b = list(H_reshape[:,j])
            locations = []
            minimum = min(b)
            for i in range(3):
                maxIndex = b.index(max(b))
                if max(b)==0:
                    maxIndex=0
                    continue
                locations.append(maxIndex)
                b[maxIndex] = minimum
                H_reshaped[maxIndex,j]=H_reshape[maxIndex,j]

            Big_locations.append(locations)
            new_H_reshaped = H_reshaped
            #print(locations,j)

        #print(np.shape(new_H_reshaped))
        
        return new_H_reshaped,Big_locations
        
        
        
    def form_labels(new_H_reshaped,Big_locations):
           
        #Form labels for single-phases point
        labels1=[]
        labels_clust1=[]
        oo=-1

        for i in range(len(Big_locations)):
            if len(Big_locations[i])==1:
                #print(Big_locations[i],i,oo)

                if Big_locations[i] in list(labels_clust1[:]):
                    #print(Big_locations[i],i,oo)
                    continue
                oo+=1    
                labels1.append(oo)
                labels_clust1.append(Big_locations[i])
                #print(Big_locations[i],i,oo,'Yes')



        #Form labels for double-phased points
        #This took a while :-( but I finally made it more efficient
        labels2=[]
        labels_clust2=[]
        ooo=oo

        for i in range(len(Big_locations)):
            p=0

            if len(Big_locations[i])==2:
                #print(i)

                for ii,v in enumerate(labels_clust2):
                    if set(Big_locations[i]) == set(v):   #Check if they are equal regardless of arrangement
                        #print(Big_locations[i],i,ooo)
                        p=1
                        continue

                if p==1:
                    continue

                ooo+=1 
                labels2.append(ooo)
                labels_clust2.append(sorted(Big_locations[i]))
                #print(Big_locations[i],i,ooo,"Yes")



        #Form labels for triple-phased points
        #This took a while too :-(

        labels3=[]
        labels_clust3=[]
        oooo=ooo

        for i in range(len(Big_locations)):
            p=0

            if len(Big_locations[i])==3:
                #print(i)

                for ii,v in enumerate(labels_clust3):
                    if set(Big_locations[i]) == set(v):  #Check if they are equal regardless of arrangement
                        #print(Big_locations[i],i,oooo)
                        p=1
                        continue

                if p==1:
                    continue

                oooo+=1
                labels3.append(oooo)
                labels_clust3.append(sorted(Big_locations[i]))
                #print(Big_locations[i],i,oooo,"Yes")


        all_labels=labels1+labels2+labels3
        labels_clust = labels_clust1+labels_clust2+labels_clust3


        #Get the cluster for each point
        labels=[]
        for val in range(0,177):
            labels.append(labels_clust.index(sorted(Big_locations[val])))

        #print(labels)
        #print(list(labels_clust))
        
#         #This part of the code is a savior
#         new_labels=[None]*177
#         for ii in range(0,177):
#             x,y=dataGrid.coord(ii+1)
#             new_grid=dataGrid.grid_num(16-x,y)
#             new_labels[new_grid-1]=labels[ii]
            
            

        return labels,all_labels,labels_clust
        #return new_labels,all_labels,labels_clust


    def get_cluster_grid(all_labels,labels):

        hues = [float(float(x)/float(len(all_labels))) for x in range(1,len(all_labels)+1)]
        cluster_grid = np.zeros(shape = (15,15,3))
        #cluster_grid = np.ones(shape = (15,15,3))
        for val in range(1,178):
            x,y = dataGrid.coord(val)
            cluster = labels[val-1]
            #cluster_grid[15-y][x-1] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])
            cluster_grid[y-1][x-1] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])

        return cluster_grid
       
                                
         
                              
    def main_func(i,deal,pear_threshold,peak_threshold,contri_threshold,dataset=1):

        W,H=nmf_func(i,deal,dataset)

        corr_idx = Pearson_func(W,H,pear_threshold)
        same_peaks = peaks_func(W,H,peak_threshold)

        if same_peaks ==[]:
            print("No clusters with the same number of peaks for the given threshold and no. of clusters")
        if corr_idx ==[]:
            print("No clusters with correlation above or equal to the given correlation threshold. No peak shifting will be considered")

        print(np.shape(corr_idx),np.shape(same_peaks))
        
        
        
        if corr_idx==[]:

            new_H = H

        elif same_peaks==[] or use_peak==0:
            use_idx =[]
            for i in range(len(corr_idx)):
                use_idx.append(corr_idx[i])

            new_H=reshape_func(H,use_idx)
       
        elif use_peak==1:

            peak_shift = peak_pearson_match(corr_idx,same_peaks)
            
            if peak_shift==[]:
                
                print('No match between Pearson and same peaks found. Just Pearson will be considered')
                use_idx =[]
                for i in range(len(corr_idx)):
                    use_idx.append(corr_idx[i])
                 
                new_H=reshape_func(H,use_idx)
                
            else:
               
                use_idx =[]
                for i in range(len(peak_shift)):
                    use_idx.append(peak_shift[i])

                new_H=reshape_func(H,use_idx)






        new_H_reshaped,Big_locations = contributions(new_H,contri_threshold)
        
        #print(np.shape(Big_locations))

        labels,all_labels,labels_clust = form_labels(new_H_reshaped,Big_locations) 

        cluster_grid = get_cluster_grid(all_labels,labels)

        return cluster_grid,labels

          

            
   # This is where the interesting stuffs happen

    if dataset==1:
        dataGrid = DataGrid_TiNiSn_500C()
        data = dataGrid.get_data_array()
    elif dataset==2:
        dataGrid = DataGrid_TiNiSn_600C()
        data = dataGrid.get_data_array()
        
    
    if visualize==1:
            clust_visuals(a,b,deal,dataset)
    
    if b-a > 8: x_width=15
    else: x_width=10
    
    
    Big_labels=[]
    fig = plt.figure(figsize=(x_width,8))
    fig.tight_layout()
    
    
    for i in range(a,b):
        
        print('.....................................'+str(i)+' Clusters'+'...........................................')
        
        cg,labels = main_func(i,deal,pear_threshold,peak_threshold,contri_threshold,dataset)
        Big_labels.append(labels)
        ax = fig.add_subplot(2,np.ceil((b-a)/2),i-a+1)
        ax.imshow(cg)
        ax.invert_yaxis()
        #ax.invert_xaxis()
        ax.title.set_text(i)
        ax.axis("off")
        
        
    
    #print(".................There are %d clusters in total..........." %len(all_labels))  
    k=.05
    plt.subplots_adjust(left=k,right=(1-k),bottom=k,top=(1-k),wspace=k,hspace=k)
    plt.show()
    
    return Big_labels
        
        
            
        
             
                              
                              
                              
                              
