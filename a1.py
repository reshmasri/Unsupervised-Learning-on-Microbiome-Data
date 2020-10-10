#!/usr/bin/env python
# coding: utf-8

# In[1]:


# required libraries 
import numpy as np
import pandas as pd
import urllib.request
import os
import sys
import gzip 
import shutil
import matplotlib.pyplot as plt
from itertools import product
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy.cluster.hierarchy as shc
from sklearn.cluster import KMeans

def main():
    script = sys.argv[0]
    number = sys.argv[1]
    path_data = sys.argv[2]
    l=int(number)
#Section ---1
#path_data= 'https://www.eecs.uottawa.ca/~turcotte/teaching/csi-5180/assignments/1/human_skin_microbiome.csv'
    data = pd.read_csv(path_data)

#Creating a folder dataset to sstore the genome files
    path="Dataset"
    if not os.path.exists(path):
        os.makedirs(path)
    
    labels=[]

#to download the files for each genome and unzip them
    for i in range(len(data)):
        fullfilename = os.path.join(path, data.Organism[i] +'.fna.gz')
        urllib.request.urlretrieve(data.URL[i], fullfilename)
        with gzip.open(fullfilename, 'rb') as f_in:
            fullfilename1 = os.path.join(path, data.Organism[i] +'.txt')
            with open(fullfilename1, 'wb') as f_out:  
                shutil.copyfileobj(f_in, f_out)
                           
#function to find the frequency pairs of "ACGT"
    def all_repeat(str1, rno):
        chars = list(str1)
        results = []
        for c in product(chars, repeat = rno):
            s = ''
            r = s.join(c)
            results.append(r)
        return results

#dealing with lines starting with ">"
    for i in range(len(data)):
        fullfilename1 = os.path.join(path, data.Organism[i] +'.txt')
        labels.append(data.Organism[i])
        with open(fullfilename1,"r") as f:
            lines = f.readlines()
        with open(fullfilename1, "w") as f:
            for line in lines:
                line = line.upper()
                if  line.startswith(">"):
                    line = " "
                f.write(line)

    def replaceMultiple(mainString, toBeReplaces, newString):
    # Iterate over the strings to be replaced
        for elem in toBeReplaces :
        # Check if string is in the main string
            if elem in mainString :
            # Replace the string
               mainString = mainString.replace(elem, newString)
    
        return  mainString    

    #finding the frequency vector
    frequency_vector_allGenomes = []
    letter=["B","D","E","F","H","I","J","K","L","M","N","O","P","Q","R","S","U","V","W","X","Y","Z"]
    for i in range(len(data)):   
        fullfilename1 = os.path.join(path, data.Organism[i] +'.txt')
        with open(fullfilename1,"r") as f:
            seq = f.read().replace('\n', '')
            seq = replaceMultiple(seq,letter,"")
        denominator_list = [(seq[i:i+l] )  for i in range(len(seq) - (l-1))] 
        denominator = []
        for i in denominator_list:
            if(i.find(" ") == -1 ):
                denominator.append(i)
        numerator_list = all_repeat('ACGT', l)
        encoded_vector = []
        for i in range(len(numerator_list)):
            numerator = seq.count(numerator_list[i])
        #print(numerator)
            vector_elements = numerator/len(denominator)
            encoded_vector.append(vector_elements)
        frequency_vector_allGenomes.append(encoded_vector)
    frequency_vector_allGenomes =pd.DataFrame(frequency_vector_allGenomes)
    
    
#Section ---2.1


#K means and inertia graph
    inertia = []
    Genomes_size= range(1,28)
    for i in Genomes_size:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(frequency_vector_allGenomes)
        preds = kmeans.fit_predict(frequency_vector_allGenomes)
        inertia.append(kmeans.inertia_)
    #print("showing the inertia of the clusters for all possible values of k")
    plt.plot(Genomes_size, inertia,'bx-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('inertia')
    plt.show()


#K means and silhouette score graph
    silhouette = []
    x=range(2,(len(Genomes_size)))
    for i in x:
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(frequency_vector_allGenomes)
        preds = kmeans.labels_
        silhouette.append(silhouette_score (frequency_vector_allGenomes, preds,metric="euclidean"))
    dictionary = dict(zip(x, silhouette))
    #print("showing the silhouette score of the clusters for all possible values of k")
    plt.plot(x, silhouette,'bx-')
    plt.title('silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('silhouette score')
    plt.show()
    print("optimal number of clusters :",max(dictionary, key=dictionary.get))

#Section ---2.2

#Dendrogram graph
    plt.figure(figsize=(12, 8))
    plt.title("Dendograms")

    dend = shc.dendrogram(shc.linkage(frequency_vector_allGenomes, method='single'), leaf_rotation=90, leaf_font_size=14, labels=labels)
    plt.show()
    
main()

