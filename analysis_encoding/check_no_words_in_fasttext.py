import pandas as pd
import spacy
import nltk
import numpy as np
import torch
from models import InferSent
df=pd.read_csv("/home/psrivastava/Intern_Summer/data/new_output.csv")
abs_arr=df.ix[:4,'clean_text']
nlp=spacy.load("en_core_web_sm")
MODEL_PATH="/home/psrivastava/Intern_Summer/infersent/encoder/infersent2.pkl"
W2V_PATH="/home/psrivastava/Intern_Summer/infersent/fastText/crawl-300d-2M.vec"
params_model={'bsize':64,'word_emb_dim':300,'enc_lstm_dim':2048,'pool_type':'max','dpout_model':0.0,'version':2}
infersent=InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
infersent.set_w2v_path(W2V_PATH)

for index in range(len(abs_arr)):
    doc=nlp(abs_arr[index])
    strs_after_stop_arr=[]
    for token in doc:
        if not token.is_stop:
            strs_after_stop_arr.append(token.text)
    
    abs_arr[index]=' '.join(strs_after_stop_arr)
    
infersent.build_vocab(abs_arr) #But Actually they are abstracts of diffrent papers
print(infersent.encode(abs_arr)[0][:])
