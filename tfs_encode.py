#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os 
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def encode(file_name,update=False):
    module_url="https://tfhub.dev/google/universal-sentence-encoder-large/3"
    #Look out for windows or linux operating system
    data_path=os.getcwd() 
    os_specific="/" if "posix" in os.name else "\\" 
    
    if update:
        df=pd.read_csv(data_path+os_specific+file_name)
    else:

        df=pd.read_csv(data_path+os_specific+"database_clean.csv")
   
    print("Data To Be Encoded ",df.shape)
    
    str_arr=df.ix[:,'abstract'].tolist()
    title_arr=df.ix[:,'title'].tolist()
    catg_arr=df.ix[:,'categories'].tolist()
    id_arr=df.ix[:,'id'].tolist()
    abc=[]
    
    
    embed=hub.Module(module_url)
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        print("sessionrun")
        perv=0
        aas=[]

        if not update:
            for i in range(0,len(str_arr),100000): #Problem Can Arise Here 
                print("encoding iteration",i)
                if i == 0:
                    aa=embed([str_arr[i]]).eval()
                    aas=aa
                else:
            
                    aa=embed(str_arr[perv:i]).eval()
                    aas=np.concatenate([aas,aa],axis=0)
                print(aas.shape)
                perv=i

            aas=np.delete(aas,0,axis=0)
        else:

            aas=embed(str_arr).eval()
            print(aas.shape)

                
        for x in range(aas.shape[0]):
            
            aac=aas[x,:].reshape(1,-1)
            abc.append((aac,title_arr[x],catg_arr[x],id_arr[x]))
        
        nps=np.array(abc)
        
        if update:
            return nps
        else:
            np.save('/home/psrivastava/Intern_Summer/data/db_encoded',nps)





