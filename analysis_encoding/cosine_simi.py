import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as mpatches
data=pd.read_csv("/home/psrivastava/Intern_Summer/data/removal_stopwords_embeds_sentence.csv")
print(data.columns)
sets=data.ix[2000:5000,'set'].to_list()
catg=data.ix[2000:5000,'catg'].to_list()
cltxt=data.ix[2000:5000,'clean_text'].to_list()
uni,count=np.unique(sets,return_counts=True)
unique_sets=dict(zip(uni,count))
print(sets)
embeds=data.ix[2000:5000,'embds'].to_list()
dict_cosine={}
dict_catg={}
dict_simi={}
dict_cltxt={}
for xp in unique_sets:
    for kl in range(len(sets)):
        if xp == sets[kl] and xp not in dict_cosine:
            print(xp)
            vector_1=np.array(eval(embeds[kl])).reshape(1,-1)
            dicts={}
            catt=[]
            simis=[]
            abss=[cltxt[kl]]
            for mp in range(len(sets)):
                if mp > kl:
                    if sets[mp] not in dicts:
                        vector_2=np.array(eval(embeds[mp])).reshape(1,-1)
                        
                        simi=cosine_similarity(vector_1,vector_2)[0][0]
                        
                        dicts[sets[mp]]=simi
                        
                        catt.append(catg[mp])
                        abss.append(cltxt[mp])
                        simis.append(simi)
            dict_catg[sets[kl]]=catt
            dict_cltxt[sets[kl]]=abss
            print(sets[kl],'-------',dicts)
            print(dict_catg[sets[kl]])
            dict_cosine[sets[kl]]=dicts
            dict_simi[sets[kl]]=simis
df=pd.DataFrame.from_dict(dict_cosine,orient='index')
print(df.isnull().sum())
col=['blue','red','green','black','pink','violet','salmon','gold','skyblue','darkcyan','cyan','crimson','lime','teal','indianred','chocolate','sienna','orange','dimgrey']

fig,axs=plt.subplots(1,1,figsize=(15,15))

for idx in df.index.to_list():
    axs.scatter(df.ix[idx,:],[df.index.to_list().index(idx) for x in range(len(df.index.to_list()))],color=col,s=20*2**4)
    argma=np.argmax(np.array(dict_simi[idx]))

    x1=dict_simi[idx][argma]
    y1=df.index.to_list().index(idx)
    print(x1,y1,argma,'-----------------',dict_cltxt[idx][0],'------------------',dict_cltxt[idx][argma+1]) 
    axs.annotate(dict_catg[idx][argma],xy=(x1,y1),xycoords='data',xytext=(x1+0.012,y1))   
patches=[mpatches.Patch(color=cl,label=ll) for cl,ll in zip(col,df.columns.to_list())]
locs,labels=plt.yticks()
plt.yticks(np.arange(19),df.index.to_list())
plt.legend(handles=patches)
plt.show()
plt.xlabel("cosine distance")
plt.ylabel("sets")
fig.savefig("cosine_distance_sentences_wo_stopword.png")



