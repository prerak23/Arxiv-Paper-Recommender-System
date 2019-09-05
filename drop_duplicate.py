#!/usr/bin/env/ python3
import numpy as np
import pandas as pd


def dupli(file_name):
    

    kd=np.load(file_name,allow_pickle=True)
    dp=pd.DataFrame(kd,columns=["embeds","title","catg","id"])
    id_dupli=dp[dp.duplicated(["title"])]
    id_dupli=id_dupli.ix[:,"title"].index.tolist()
    np.save("db_encoded",np.delete(kd,id_dupli,0))


