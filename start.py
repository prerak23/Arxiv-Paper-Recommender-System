import pandas as pd
import numpy as np
from mastodon import Mastodon 
from bs4 import BeautifulSoup
import re
import urllib2 as libreq
import service
import time
mastadon=Mastodon(access_token='your_id',api_base_url='your_url')

def sta():
    kc=mastadon.notifications()
    print(len(kc))
    if len(kc) != 0:
        allp(kc)
    else:
        time.sleep(60)
        sta()
        


def allp(kc):
    for x in kc:
        
        match=re.findall("[0-9]+\.[0-9a-z]+", BeautifulSoup(x['status']['content']).get_text())
        usr=x['status']['account']['username']
        mastadon.status_post("@"+x['status']['account']['username']+" "+"Recived Your Request Proceesing The Information ",visibility='direct')
        try : 
        	for xp in match:
            		arxiv_link="http://export.arxiv.org/api/query?id_list="+xp
            		print("-----------------------")
          
	  
            		text_str=BeautifulSoup(libreq.urlopen(arxiv_link).read()).find('summary').get_text()
		   
            		regex=r"\$[-\\\"#\/@^( ),.;:<>{}\[\]`'+=~|!?_a-zA-Z0-9]*\*"
            		regex2=r"\$"
            		matches2=re.finditer(regex2, text_str,re.MULTILINE)
            		counter_to_check_even_do_symbols=0
            		for y in matches2:
                		counter_to_check_even_do_symbols+=1
                		if counter_to_check_even_do_symbols % 2 == 0:
                     			text_str=text_str[:y.start()]+"*"+text_str[y.start()+1:]
            		text_str=text_str.replace("\n"," ")
            		matches=re.finditer(regex, text_str, re.MULTILINE)
            		for on in matches:
                		text_str=text_str.replace(str(on.group()), "")

            		abs_wo_punc=re.sub("[^a-zA-Z0-9;:!?.,]"," ",text_str)
            		
            
            		arrs=service.start(abs_wo_punc)
            		
            		op=""
            		for np in arrs:
                		npp="http://arxiv.org/abs/"+str(np[1])+"   "+"Score "+str(np[3])+"\n"
                		op=op+npp
            		mastadon.status_post("@"+usr+" "+op ,visibility='direct')
        except:
                mastadon.status_post("@"+usr+" "+"Bad-Request for "+xp,visibility='direct')
                
        mastadon.notifications_dismiss(x['id'])
    sta()

sta()





