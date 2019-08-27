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
dfs=df.sample(n=270000,random_state=1).reset_index(drop=True)
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
        title_arr=dfs.ix[currentidx:to_fetch,'title'].tolist()
        currentidx=currentidx+640
        return abs_arr,catg_arr,subj_arr,title_arr,currentidx

def no_stopwords():
   # pdss=pd.DataFrame(columns=['embds','set','catg','title'])
    abc=[]
    start=time.time()
    global current_idx
    for x in range(420):
        crix=current_idx
        abss,catg,sets,title,crix=get_batch_from_dataframe(crix)
        print("get_the_batch",x)
        copy_abss=abss.copy()
        for index in range(len(abss)):
            doc=nlp(abss[index])
            sents_arr=[]
            sents=""
            for token in doc:

                if "PUNCT" in token.pos_:
                    sents_arr.append(sents)
                    sents=""

                elif  token.is_stop:
                    print(token.text)

                else :
                    sents=sents+" "+token.text
                    
                    
            abss[index]=sents_arr
            

        
        if x==0:
            
            infersent.build_vocab(copy_abss,tokenize=True)
        
        else:
            
            infersent.update_vocab(copy_abss,tokenize=True)
        
        print("aBSS lENGTH",len(abss))
        
        for indexx in range(len(abss)):

            if len(abss[indexx]) > 0 :
                
                embed=infersent.encode(abss[indexx],bsize=len(abss[indexx]),tokenize=True)
            
                embed_avg=np.mean(embed,axis=0)
            
                abc.append((embed_avg,sets[indexx],catg[indexx],title[indexx]))
                #df2=pd.DataFrame({'embds':list(embed_avg),'set':sets[indexx],'catg':catg[indexx]})
                
                #pdss.loc[(len(pdss))]=[embed_avg,sets[indexx],catg[indexx],title[indexx]]
        
        current_idx=crix

    end=time.time()-start
    print("Time Abstract To Sentences stopwords",end)
    nps=np.array(abc)
    np.save("/home/psrivastava/Intern_Summer/data/removal_stopwords_embeds_sentence",nps)

no_stopwords()

