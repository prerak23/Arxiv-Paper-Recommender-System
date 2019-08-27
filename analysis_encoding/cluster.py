#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
import seaborn as sns
import matplotlib.pyplot as plt
import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from pyclustering.cluster.kmeans import kmeans
from spherecluster import SphericalKMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
from sklearn.metrics import pairwise_distances
from sklearn import metrics



nps=np.load("/home/psrivastava/Intern_Summer/data/tfs_encode.npy")
df=pd.DataFrame(nps,columns=['embds','title','sets','catg'])

embed=[np.array(x) for x in df.iloc[:20000,0].to_list()]
sets=[x for x in df.iloc[:20000,2].to_list()]
catg=[x for x in df.iloc[:20000,3].to_list()]
title=[x for x in df.iloc[:20000,1].to_list()]
#dim_reduc=KernelPCA(n_components=2000,kernel='cosine').fit_transform(np.array(embed))
#print(dim_reduc.shape)
print(np.squeeze(np.array(embed)).shape)
skm=SphericalKMeans(n_clusters=11)
skm.fit(np.squeeze(np.array(embed)))

X_embeded=TSNE(n_components=2,metric="cosine").fit_transform(np.squeeze(np.array(embed)))
dim_reduc=X_embeded
uni,coun=np.unique(np.array(sets),return_counts=True)
print("Dimension reduc",X_embeded.shape)
print(dict(zip(uni,coun)))




la=skm.labels_
print(metrics.silhouette_score(np.squeeze(np.array(embed)),la,metric="cosine"))
#cluster_center=skm.cluster_centers_
#print(cluster_center.shape)
#print(skm.inertia_.shape)
unique,count=np.unique(la,return_counts=True)
#clust_center=TSNE(n_components=3,metric="correlation").fit_transform(cluster_center)
print(dict(zip(unique,count)))


col=['blue','red','green','black','pink','violet','salmon','gold','skyblue','darkcyan','cyan','crimson','lime','teal','indianred','chocolate','sienna','orange','dimgrey','lightpink']

fig,ax=plt.subplots(1,1,figsize=(15,15))

zip_col_set={x:p for x,p in zip(unique,col)}
print(zip_col_set)

for y in range(len(X_embeded)):

    ax.scatter(dim_reduc[y][0]+1, dim_reduc[y][1]+1,  c=zip_col_set[la[y]])





    






dic={x:[] for x in unique}
dic1=copy.deepcopy(dic)
dic2=copy.deepcopy(dic)
for i,x in enumerate(la):
    dic[x].append(sets[i])
    dic1[x].append(catg[i])
    dic2[x].append(title[i])

#dic={x:np.unique(p) for x,p in dic.items()}
#dic1={x:np.unique(p) for x,p in dic1.items()}
#dic2={x:np.unique(p) for x,p in dic2.items()}
#dics=[p for x,p in dic.items()]

with open("for_less_data_cluster_data.txt","w+",encoding="utf8") as fof:
    for x,p in dic.items():
        fof.write("cluster "+str(x)+" ->"+str(p)+"\n")
    fof.write("------------------------------------------")
    for k,l in dic1.items():
        fof.write("cluster "+str(k)+" ->"+str(l)+"\n")
    fof.write("------------------------------------------")
    for k,l in dic2.items():
        fof.write("cluster "+str(k)+" ->"+str(l)+"\n")
    


patches=[mpatches.Patch(color=cl,label=ll) for cl,ll in zip(col,unique)]
locs,labels=plt.yticks()
plt.legend(handles=patches)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
plt.show()
plt.savefig("tfs_encode_11_cluster.png")








