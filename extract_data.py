#!/usr/bin/env python3
import pandas as pd
import numpy as np
import arxivscraper
import pandas as pd
from arxivscraper import Scraper
ll_sets=['physics','math','cs','econ','eess','physics:astro-ph','physics:cond-mat','physics:gr-qc','physics:hep-ex','physics:hep-lat','physics:hep-ph','physics:hep-th','physics:math-ph','physics:nlin','physics:nucl-ex','physics:nucl-th','physics:physics','physics:quant-ph','q-bio','q-fin','stat']

big_pandas=pd.DataFrame([])
for xp in ll_sets:
    print(xp)
    ax =Scraper(category=xp, date_from='2015-01-01',date_until='2019-08-01')
    output=ax.scrape()
    

    df=pd.DataFrame(output,columns=['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors','url'])
    df=df.drop(columns=['doi','authors','created','updated','url'])
    print(len(df))
    big_pandas=pd.concat([big_pandas,df],sort=False,ignore_index=True)

big_pandas.to_csv('database.csv')






