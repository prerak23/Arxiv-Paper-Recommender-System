#!/usr/bin/env python3
import pandas as pd
import spacy
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
cd=pd.read_csv("/home/psrivastava/Intern_Summer/data/output.csv")
subj_arr=cd.iloc[:,-1].value_counts()
x_barplot,y_barplot=list(zip(*[[x,y] for x,y in subj_arr.items()]))
fig,ax1=plt.subplots(1, 1, figsize=(13,8), sharex=True)

sns_bar=sns.barplot(np.array(list(y_barplot))[:20],np.array(list(x_barplot))[:20], palette="cubehelix", ax=ax1)
ax1.set(ylabel="Diffrent Subjects",xlabel="Number Of Papers Published After 01-01-2018")
sns_bar.set()
figs=sns_bar.get_figure()
figs.savefig("subj_distribution_category.png")
