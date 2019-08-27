#!/usr/bin/env python3
import pandas as pd
import numpy as np
from spherecluster import SphericalKMeans
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style 

dp=np.load('/home/psrivastava/Intern_Summer/data/tfs_encode.npy')
df=pd.DataFrame(dp,columns=['embeds','sets','catg','title'])

embed=[np.array(xp) for xp in df.ix[:10000,'embeds'].tolist()]

no_of_cluster=[5,7,9,11,13,15,17]

score=[]
for x in no_of_cluster:
    print(x)
    skm=SphericalKMeans(n_clusters=x)
    skm.fit(np.squeeze(np.array(embed)))
    labels=skm.labels_
    sc=metrics.silhouette_score(np.squeeze(np.array(embed)),labels,metric="cosine")
    score.append(sc)



fig,ax=plt.subplots(1,1,figsize=(15,15))
ax.plot(np.arange(len(no_of_cluster)),score)
style.use('seaborn')
ax.set_xlabel('no of clusters')
ax.set_ylabel('Sillhoute Score')
ax.set_xticklabels(no_of_cluster)
ax.grid()
plt.show()
fig.savefig('tfs_elbow.png')



