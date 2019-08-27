
#print(labels)
#cluster_centers=clustering.cluster_centers_
#labels_unique,counts=np.unique(labels,return_counts=True)
#print(dict(zip(labels_unique,counts)))

#sets_per_cluster={xp:[] for xp in range(len(labels_unique))}
#for xp in range(len(labels)):
    #if sets[xp] not in sets_per_cluster[labels[xp]]:
        #sets_per_cluster[labels[xp]].append(sets[xp])
#print(sets_per_cluster)
#print(labels_unique)
#ds=dict(zip(labels_unique,counts))
#print(ds)
 #print(X_embeded.shape)
#no_of_clusters=len(labels_unique)
#print(no_of_clusters)
#colors=['blue','red','green','black','pink','violet','salmon','gold','skyblue','darkcyan','cyan','plum','orchid','teal']
#new_df=pd.DataFrame(columns=['x','y','z','clus'])
#fig=plt.figure()
#fig,ax=plt.subplots(1,1,figsize=(15,15))

#for k,col in zip(range(no_of_clusters), colors):

#for y in range(len(X_embeded)):
    #ax.scatter(dim_reduc[y][0], dim_reduc[y][1], c='darkcyan', alpha=0.5)



#ax.set_xlabel('X Axis')
#ax.set_ylabel('Y Axis')
#ax.set_zlabel('Z Axis')
    #ax.legend()
#plt.show()

#fig.savefig("/home/psrivastava/Intern_Summer/code/prerak/Plots/3d_cluster_after_avg.png")


#fig,ax1=plt.subplots(1, 1, figsize=(13,8), sharex=True)
#axp=sns.scatterplot(x=new_df.x, y=new_df.y ,z=new_df.z , hue=new_df.clus , data=new_df, ax=ax1)
#axp.set()
#figs=axp.get_figure()
#figs.savefig("cluster2.png")

#print(new_df.iloc[:100,2])
#print(clustering)
