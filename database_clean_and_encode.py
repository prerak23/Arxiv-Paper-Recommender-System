#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import re
import clean_abs_stat
import tfs_encode
import drop_duplicate

dir_path=os.getcwd()
os_specific='/' if 'posix' in os.name else '\\'
final_path=dir_path+os_specific+"database.csv"
clean_abs_stat.clean_abs(final_path).to_csv("database_clean.csv")

tfs_encode.encode("database_clean.csv",False)
drop_duplicate.dupli(dir_path+os_specific+"db_encoded.npy")





