#!/usr/bin/env python3
import pandas as pd

import re
import numpy as np
import os


def clean_abs(file_name):


    cs=pd.read_csv(file_name)
    arrs=cs.ix[:,'abstract']
    
    dict_for_avg_word_tokens={}
    new_df=pd.DataFrame(columns=["abstract"])#tokens,tokens_wo_stopwords
    diffrent_tokens_per_subj={}
    to_print_count=0
    ars=[]
    for abss in arrs:
        print(to_print_count)
        regex=r"\$[-\\\"#\/@^( ),.;:<>{}\[\]`'+=~|!?_a-zA-Z0-9]*\*"
        regex2=r"\$"
        text_str=abss 
        matches2=re.finditer(regex2, text_str,re.MULTILINE) #First stage of pipeline cleaning math symbols (first part) replacing the second dollars sign with the * symbol  
        counter_to_check_even_do_symbols=0
        for y in matches2:
            counter_to_check_even_do_symbols+=1
            if counter_to_check_even_do_symbols % 2 == 0:
                text_str=text_str[:y.start()]+"*"+text_str[y.start()+1:]

        
        text_str=text_str.replace("\n"," ") #Second Stage of the pipeline removing the new-line characters

        matches=re.finditer(regex, text_str, re.MULTILINE) #Third stage of the pipeline removing the whole math symbols from the abstracts
        for x in matches:
            text_str=text_str.replace(str(x.group()), "")

        abs_wo_punc=re.sub("[^a-zA-Z0-9;:!?.,]"," ",text_str) #Fourth stage of the pipeline remove the symbols except from the strong punctuations marks from the abstract
        #doc=nlp(abs_wo_punc)    #without , . punctuations
        ars.append(abs_wo_punc)
        to_print_count+=1
    new_df.iloc[:,0]=ars
    ds=cs.drop(columns=['abstract'])

    result=pd.concat([ds,new_df],axis=1) 
    return result


'''
    count_no_tokens=0
    wo_stopwords=""
    tokens_with_stopwords=""
    for tokens in doc:
        count_no_tokens+=1
        tokens_with_stopwords=tokens_with_stopwords+tokens.text+"@"
        if not tokens.is_stop:
            wo_stopwords=wo_stopwords+tokens.text+"@"
        

    new_df.loc[len(new_df)]=[abs_wo_punc,tokens_with_stopwords,wo_stopwords]
    for tok in wo_stopwords.split("@"):
        if subj not in diffrent_tokens_per_subj:
            diffrent_tokens_per_subj[subj]=[tok]
        elif tok not in diffrent_tokens_per_subj[subj]:
            diffrent_tokens_per_subj[subj].append(tok)
        else:
            print("Already There Token")
    
    if subj not in dict_for_avg_word_tokens:
        dict_for_avg_word_tokens[subj]=[count_no_tokens]
        
    else:
        dict_for_avg_word_tokens[subj].append(count_no_tokens)

    to_print_count+=1

diffrent_tok_pd=pd.DataFrame.from_dict(diffrent_tokens_per_subj)
diffrent_tok_pd.to_csv("diffrent_tokens_per_subj.csv")
'''

'''
with open("avg_token_per_subj.txt","w+",encoding="utf8") as fofs:
    li=[]
    for x in dict_for_avg_word_tokens:
        li.append((x,np.sum(dict_for_avg_word_tokens[x]),np.mean(dict_for_avg_word_tokens[x])))
    fofs.write(str(li))
'''
