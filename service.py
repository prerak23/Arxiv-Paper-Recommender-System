import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import spacy
import torch
import re
import time
import sys
from sklearn.metrics.pairwise import cosine_similarity



module_url="https://tfhub.dev/google/universal-sentence-encoder/2"
big_data=pd.DataFrame(np.load('db_encoded.npy',allow_pickle=True),columns=['embds','title','catg','id'])
all_emb=big_data.ix[:,'embds'].tolist()
se=big_data.ix[:,'id'].tolist()
cat=big_data.ix[:,'catg'].tolist()
tit=big_data.ix[:,'title'].tolist()
#MODEL_PATH="/home/psrivastava/Intern_Summer/infersent/encoder/infersent2.pkl"
#W2V_PATH="/home/psrivastava/Intern_Summer/infersent/fastText/crawl-300d-2M.vec"
#params_model={'bsize':64,'word_emb_dim':300,'enc_lstm_dim':2048,'pool_type':'max','dpout_model':0.0,'version':2}
#infersent=InferSent(params_model)
#infersent.load_state_dict(torch.load(MODEL_PATH))
#infersent.set_w2v_path(W2V_PATH)
#use_cuda=True
#infersent=infersent.cuda() if use_cuda else infersent
nlp=spacy.load('en_core_web_sm')
embed=hub.Module(module_url)
def start(txts):
    text=txts
    doc=nlp(text)
    sents_arr=[]
    sents=""
    a=0
    for token in doc:
    

    #if "PUNCT" in token.pos_:
        #sents_arr.append(sents)
        #sents=""

        if token.is_stop:
            a+=1
        else :
            sents=sents+" "+token.text
    
    
    arrs=[sents]
#infersent.build_vocab(arrs,tokenize=True)
#abs_emb=infersent.encode(sents_arr,bsize=len(sents_arr),tokenize=True)

#if len(sents_arr) > 1:
    #abs_emb_avg=np.mean(abs_emb,axis=0).reshape(1,-1)
#else:
    #abs_emb_avg=abs_emb.reshape(1,-1)

    
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        aa=embed(arrs).eval()

    abs_emb=aa.reshape(1,-1)
    nor_abs=np.linalg.norm(abs_emb)
    top_simi=[]
    all_simi=[]
    stra=time.time()
    nor=[]
    for z in range(len(all_emb)):
        nor.append(np.linalg.norm(np.array(all_emb[z]).reshape(1,-1)))
    print("Time Taken To Find Norm ",time.time()-stra)
    stra=time.time()

    for z in range(len(all_emb)):

        all_simi.append(np.vdot(abs_emb.reshape(1,-1),all_emb[z].reshape(1,-1))/(nor[z]*nor_abs))
    print("Time Taken To Find Similarity Of Whole",len(all_emb)," Database ",time.time()-stra)
    sr=np.argsort(all_simi)
    sr=sr[::-1][:5]
    for xk in sr:
        top_simi.append((tit[xk],se[xk],cat[xk],all_simi[xk]))
   
    return top_simi

        
'''
    for x in top_simi:
        print("title:-"+" ",x[0])
        print("Id:-"+" ",x[1])
        print("Sub-Category:-"+" ",x[2])
        print("Similarity Score:-"+" ",x[3])
        print("--------------------------")
    tt=input("Do You Want Another Recommendation type y or n")
    if "y" in tt:
        start()
    else:
        print("Thanks For Using this service")
        sys.exit()
'''


