#this program generates the transition matrix from bill / word space to topic space
#end product is to write transition matrix to file

import numpy as np
import nltk
import os
import BeautifulSoup as bs
import re

import matplotlib.cm as cm
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
#clustering
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch
import clustergram as cplot
#

import sys

#==================================================================
#==================================================================
#load library of all data
#==================================================================
#==================================================================
data = ['']
folder = '../BILLSUM-113-s/'
files = os.listdir('../BILLSUM-113-s/.')
fcount = 0
for w in files:
    if re.findall('BILL',w) != []:
        fid = open(folder + w,'r')
        tdata = fid.read()
        fid.close()
        soup = bs.BeautifulSoup(tdata)
        td = soup.summary.contents
        count = 0
        while re.findall('summary-text',str(td[count])) == []:
            count+=1
        data.extend(re.split('[<>\W]',td[count].contents[0]))
    fcount+=1
#==================================================================
#==================================================================
#compute dictionary of words from the set of all bills
#==================================================================
#==================================================================

#make all words lower case and get rid of words of length 3 or smaller
data = [w.lower() for w in data if len(w) >2]
ndata = nltk.Text(data)
#compute the minimal set
sdata = set(data)
#compute word frequencies
cdata = nltk.FreqDist(data)
#cdata.plot(50,cumulative = True)
#dictionary of words, definition of vector space for bills
fdict = {str(w):float(cdata[w]) for w in cdata.keys() if cdata[w] > 1}


#==================================================================
#==================================================================
#compute frequency of each of the dictionary words in each bill
#==================================================================
#==================================================================
bill_data = []
folder = '../BILLSUM-113-s/'
files = os.listdir('../BILLSUM-113-s/.')
for w in files:
    if re.findall('BILL',w) != []:
        #read in data xml
        fid = open(folder + w,'r')
        tdata = fid.read()
        fid.close()
        #give data string structure with beautiful soup
        soup = bs.BeautifulSoup(tdata)
        #select summary text parent 
        tdata = soup.summary.contents
        count = 0
        #find summary section
        while re.findall('summary-text',str(tdata[count])) == []:
            count+=1
        #transform summary string into a list of words and punctuation
        tdata = re.split('[<>\W]',tdata[count].contents[0])
        #make sure data are all lower case, and then get the frquency dist of bill
        freqdata = nltk.FreqDist([wtemp.lower() for wtemp in tdata if len(wtemp) > 3])
        #make freqdata a dictionary
        fdata = {str(wd):freqdata[wd] for wd in freqdata.keys() if freqdata[wd] > 1}
        data = []
        #go through the dictionary words and extract frequecies, 
        for wd in fdict.keys():
            if fdata.keys().count(wd) == 0:
                data.append(0)
            else:
                data.append(fdata[wd])
        bill_data.append(data)
#==================================================================
#==================================================================
#normalize and make numpy array of bills in dictionary space
#==================================================================
#==================================================================
#data.append([freqdata[wd] for wd in fdict.keys() if freqdata.freq(wd) > 0])
bdata = np.array(bill_data)
bdata = bdata[np.sum(bdata,1) != 0,:]
norm = np.tile(np.sum(bdata,1).reshape(bdata.shape[0],1),
               (1,bdata.shape[1])) * 1.0
norm_bdata = bdata / norm

#==================================================================
#==================================================================
#PCA the vectors, diagonalizing and covariance takes 11 minutes
#==================================================================
#==================================================================

#covariance matrix at least 3 minutes
c = np.cov(norm_bdata.transpose())
#unable to compute eigenvectors
pc = np.linalg.eig(c)

lams = np.cumsum(np.sort(np.real(pc[0]))[::-1] / np.sum(pc[0]))
major_lambdas = lams[lams <= 0.95]
plt.figure(figsize = (5.5,4.5))
plt.plot(major_lambdas,'-',linewidth = 4,color ='k')
plt.tick_params(axis='both',which='major',labelsize = 17.5)
plt.ylabel(r'Cumulant of $\sigma^2_i$',fontsize = 25)
plt.xlabel(r'Rank $\sigma^2_i$',fontsize = 25)
plt.tight_layout()
plt.savefig('../figures/eigenvectors_dimension_reduction.pdf',
            format='pdf')
plt.show(block = False)

'''
PCA does not adequately reduce the dimensionality of the problem
'''


#====================================================================
#====================================================================
#====================================================================
#====================================================================
#====================================================================

#need to increase recursion limit for this large of a problem

sys.setrecursionlimit(10000)
#variance minimization
method = 'ward'
#compute distance matrix
pdist = distance.squareform(distance.pdist(bdata))
#cluster according to wards minimal variance algorithm
clusters = sch.linkage(pdist,method = method)
#get cluster assigments
T = sch.fcluster(clusters,
                 t = 0.1*clusters.max(),
                 criterion='distance')
#show that the order from fcluster complements the order derived by den['leaves']
den = sch.dendrogram(clusters,no_plot=True)

#plot
plt.figure(figsize = (5.5,4.5))
plt.plot(T[den['leaves']],'-',linewidth=4,color='k')
plt.tick_params(axis='both',which='major',labelsize = 17.5)
plt.xlabel('Bills',fontsize = 25)
plt.ylabel('Unsupervised Topics',fontsize = 25)
plt.tight_layout()
plt.savefig('../figures/topics_clusters_{}.pdf'.format(T.max()),
            format = 'pdf')
plt.show(block = False)

#organize the distance matrix by dendrogram
tdist = pdist[:,den['leaves']]
tdist = tdist[den['leaves'],:]

#get clustergram
import clustergram as clplot
clplot.plot_clustergram(np.log(tdist),clusters,
                        savename='../figures/bill_clustergram.pdf')


#================================================================================
#average over clusters
topic_transform = []
for w in range(1,T.max()+1):
    topic_transform.append(np.mean(bdata[T == w,:],0))
topic_transform = np.array(topic_transform)
norm = np.sum(topic_transform,1)
norm = np.tile(norm.reshape(norm.size,1),(1,topic_transform.shape[1]))
topic_transform = topic_transform / norm

def get_coluer(N):
    coluer = np.linspace(0,N-1,N)
    coluer = np.tile(coluer.reshape(N,1),(1,3))
    coluer[:,1] = 0
    coluer[:,2] = coluer[:,2][::-1]
    return coluer / coluer.max()

coluer = get_coluer(topic_transform.shape[0])
plt.figure()
for w in range(topic_transform.shape[0]):
    plt.plot(topic_transform[w,:],'-',color=coluer[w,:])
plt.show(block = False)

topic_transform[topic_transform > 0] = 1
plt.figure(figsize=(15,3.5))
plt.imshow(topic_transform,interpolation='nearest',
           cmap=cm.bone,origin='lower',
           aspect='auto')
plt.xlabel('Word Weights',fontsize = 25)
plt.ylabel('Topics',fontsize = 25)
plt.tight_layout()
plt.savefig('../figures/word_topic_transform.pdf',format='pdf')
plt.show(block = False)


#==============================================================================
#==============================================================================
#print top rated words in each vector
#==============================================================================
#==============================================================================
dict_words = np.array(fdict.keys())
for w in range(topic_transform.shape[0]):
    idx = np.argsort(topic_transform[w,:])
    print '============================'
    for wf in dict_words[idx[:10]]:
        print wf

#==============================================================================
#==============================================================================
#print top rated words in each vector
#==============================================================================
#==============================================================================

fid = open('../parameters/word_to_topic.txt','w')
temp = str(fdict.keys()[0])
for w in fdict.keys()[1:]:
    temp = temp + '\t' + str(w)
temp = temp + '\n'
fid.write(temp)

for w in range(topic_transform.shape[0]):
    temp = str(topic_transform[w,0])
    for elem in range(1,topic_transform.shape[1]):
        temp = temp + '\t' + str(topic_transform[w,elem])
    temp = temp + '\n'
    fid.write(temp)
fid.close()
    
