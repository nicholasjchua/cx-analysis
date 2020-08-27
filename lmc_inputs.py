# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: wasp
#     language: python
#     name: wasp
# ---

# # LMC inputs
# Exploratory data analysis: view the distributions of the number and types of (intraommatidia) inputs to the 5 classes of lamina monopolar cells. 

import numpy as np
import pandas as pd
from typing import Tuple, Union, List
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# +
tp = '200218'
linkdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')
cxdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_cxdf.pickle')

subtypes = np.unique([*linkdf["pre_type"], *linkdf["post_type"]])
lmcs = [s for s in subtypes if s[0] is 'L']
print(lmcs)

all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  
all_ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]
ommatidia = np.unique(linkdf['pre_om'])


# +
def filter_rows(ldf: pd.DataFrame, key: 'str', values: List, remove: bool=False):

    row_bool = [(v in values) for v in ldf[key]]
    if remove:  # remove rows with row[key] in values
        row_bool = [not i for i in row_bool]
    return pd.DataFrame(ldf.loc[row_bool])

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    data[om] = links['post_type'].value_counts()
n_all_in = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R1R4', 'R2R5', 'R3R6'])
    data[om] = filtered['post_type'].value_counts()
n_pr_in = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R1R4', 'R3R6'])
    data[om] = filtered['post_type'].value_counts()
n_ppr_in = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R2R5'])
    data[om] = filtered['post_type'].value_counts()
n_cpr_in = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R1R4', 'R2R5', 'R3R6'], remove=True)
    data[om] = filtered['post_type'].value_counts()
n_notpr_in = pd.DataFrame(data=data).T#.fillna(0)
##################################################

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    data[om] = links['pre_type'].value_counts()
n_all_out = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R1R4', 'R2R5', 'R3R6'])
    data[om] = filtered['pre_type'].value_counts()
n_pr_out = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4', 'LMC_N'])
    data[om] = filtered['pre_type'].value_counts()
n_lmc_out = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R1R4', 'R3R6'])
    data[om] = filtered['pre_type'].value_counts()
n_ppr_out = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R2R5'])
    data[om] = filtered['pre_type'].value_counts()
n_cpr_out = pd.DataFrame(data=data).T#.fillna(0)

data = dict.fromkeys(ommatidia)
for om, links in linkdf.groupby('post_om'):
    filtered = filter_rows(links, 'pre_type', ['R1R4', 'R2R5', 'R3R6'], remove=True)
    data[om] = filtered['pre_type'].value_counts()
n_notpr_out = pd.DataFrame(data=data).T#.fillna(0)

display(n_all_out)
display(n_pr_out)


# +
all_lmc = n_all_in.filter(regex='^LMC')
pr_lmc = n_pr_in.filter(regex='^LMC')
ppr_lmc = n_ppr_in.filter(regex='^LMC')
cpr_lmc = n_cpr_in.filter(regex='^LMC')


# -


