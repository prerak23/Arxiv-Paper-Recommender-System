import pandas as pd
import numpy as np
import datetime
from arxivscraper import Scraper
import clean_abs_stat
import tfs_encode
import os
import gc
import drop_duplicate

'''
ll_sets=['physics','math','cs','econ','eess','physics:astro-ph','physics:cond-mat','physics:gr-qc','physics:hep-ex','physics:hep-lat','physics:hep-ph','physics:hep-th','physics:math-ph','physics:nlin','physics:nucl-ex','physics:nucl-th','physics:physics','physics:quant-ph','q-bio','q-fin','stat']
'''
ll_sets=['math']
date_load=np.load('last_update.npy')
update_time=date_load[0][0]
print(str(update_time))
print(datetime.date.today())


updated_data=pd.DataFrame([],columns=['id','title','categories','abstract'])

for xp in ll_sets:
    ax = Scraper(category=xp, date_from=str(update_time) ,date_until=str(datetime.date.today()))
    output=ax.scrape()
    df=pd.DataFrame(output,columns=['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors','url'])
    df=df.drop(columns=['doi','authors','created','updated','url'])
    updated_data=pd.concat([updated_data,df],sort=False,ignore_index=True)


updated_data.to_csv('updated_data.csv')


dir_path=os.getcwd()
os_specific="/" if "posix" in os.name else "\\"

clean_abs_stat.clean_abs(dir_path+os_specific+'updated_data.csv').to_csv('updated_data.csv',index=False)

nps=tfs_encode.encode('updated_data.csv',True)
print("New Data Encoded",nps.shape)
kp=np.load('db_encoded.npy',allow_pickle=True)
print(kp.shape)
kp=np.concatenate([kp,nps],axis=0)
print("Encoded Data Added With The Main Database",kp.shape)
np.save('db_encoded',kp)
print("More Cleaning Of The Data")
drop_duplicate.dupli("db_encoded.npy")
print("Database Updated Succesfully")
date_load[0][0]=str(datetime.date.today())
np.save("last_update.npy",date_load)
#db=pd.concat([db,clean_abs_stat.clean_abs('updated_data.csv')],sort=False,ignore_index=True)
#db.to_csv('database.csv')





