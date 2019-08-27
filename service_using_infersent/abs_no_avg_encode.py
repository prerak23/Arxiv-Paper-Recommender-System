#!/usr/bin/env python
import pandas as pd
import torch
from models import InferSent
import time
import numpy as np
import spacy
import nltk
from random import randint



df=pd.read_csv("/home/psrivastava/Intern_Summer/data/new_output.csv")
dfs=df.sample(n=4000,random_state=1).reset_index(drop=True)
current_idx=0
nlp=spacy.load("en_core_web_sm")
MODEL_PATH="/home/psrivastava/Intern_Summer/infersent/encoder/infersent2.pkl"
W2V_PATH="/home/psrivastava/Intern_Summer/infersent/fastText/crawl-300d-2M.vec"
params_model={'bsize':64,'word_emb_dim':300,'enc_lstm_dim':2048,'pool_type':'max','dpout_model':0.0,'version':2}
infersent=InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
infersent.set_w2v_path(W2V_PATH)
use_cuda=True
infersent=infersent.cuda() if use_cuda else infersent






def get_batch_from_dataframe(currentidx):

        to_fetch=currentidx+640
        abs_arr=dfs.ix[currentidx:to_fetch,'clean_text'].tolist()
        catg_arr=dfs.ix[currentidx:to_fetch,'category'].tolist()
        subj_arr=dfs.ix[currentidx:to_fetch,'set'].tolist()
        
        currentidx=currentidx+640
        return abs_arr,catg_arr,subj_arr,title_arr,currentidx

def with_stopwords():
    pds=pd.DataFrame(columns=['embds','set','catg'])
    start=time.time()
    global current_idx
    for x in range(3):
        crix=current_idx
        abss,catg,sets,title,crix=get_batch_from_dataframe(crix)
        if x == 0:
            infersent.build_vocab(abss,tokenize=True)
        else:
            infersent.update_vocab(abss,tokenize=True)

        embed=infersent.encode(abss,tokenize=True)
        print("Length", len(abss),len(catg),len(sets))
        df2=pd.DataFrame({'embds':embed.tolist(),'set':sets,'catg':catg})
        pds=pds.append(df2,ignore_index=True)
        current_idx=crix
    end=time.time()-start
    print("Time with stopwords", end)
    pds.to_csv("/home/psrivastava/Intern_Summer/data/embeds_with_stopwords.csv")










def no_stopwords():
    infersent2=InferSent(params_model)
    infersent2.load_state_dict(torch.load(MODEL_PATH))
    infersent2.set_w2v_path(W2V_PATH)
    use_cuda=True
    infersent2=infersent.cuda() if use_cuda else infersent
    pdss=pd.DataFrame(columns=['embds','set','catg'])
    start=time.time()
    global current_idx
    for x in range(3):
        crix=current_idx
        abss,catg,sets,crix=get_batch_from_dataframe(crix)
        for index in range(len(abss)):
            doc=nlp(abss[index])
            strs_after_stop_arr=[]
            for token in doc:
                if not token.is_stop:
                    strs_after_stop_arr.append(token.text)
    
            abss[index]=' '.join(strs_after_stop_arr)


        if x==0:
            infersent2.build_vocab(abss,tokenize=True)
        else:
            infersent2.update_vocab(abss,tokenize=True)

        embed=infersent2.encode(abss,tokenize=True)
        df2=pd.DataFrame({'embds':embed.tolist(),'set':sets,'catg':catg})
        pdss=pdss.append(df2,ignore_index=True)
        
        current_idx=crix
    end=time.time()-start
    print("Time without stopwords",end)
    pdss.to_csv("/home/psrivastava/Intern_Summer/data/embeds_no_stopwords.csv")


with_stopwords()
current_idx=0
no_stopwords()

   




