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

# +
import os.path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from src.utils import index_by_om

import matplotlib as mpl
mpl.rc('font', size=14)
# -

data_path = '~/Data/200115_lamina/200115_cxdf.pickle'
cxdf = pd.read_pickle(data_path)
df_all = index_by_om(cxdf)
criteria = [(df_all[ct].mean() >= 5.0) for ct in df_all.columns]
df = df_all.loc[:, criteria]


# +
def id_to_ascii(om: str) -> Tuple:
    return ord(om[0]), int(om[1])

def coord_to_id(coord: Tuple) -> str:
    return ''.join((chr(coord[0]), str(coord[1])))

def triangle_subregions(om_list, d=1):
    subgroups = []
    for om in om_list:
        x, y = id_to_ascii(om)  # each ommatidia's hex coordinates
       
        up_left = {(x, y), (x, y + d), (x + d, y + d)}
        down_right = {(x, y), (x, y - d), (x - d, y - d)}
        
        # E4 and C6 are part of a pentagonal subgroup (no neighbor directly below/above in 2d hexgrid)
        if om == 'E4':  # neighbors are D3 and D2
            down_right = {(x, y), (x - d, y - d), (x - d, y - 2*d)}
        elif om == 'C6':  # neighbors are B6 and D7
            up_left = {(x, y), (x - d, y + d), (x + d, y + d)}

        # for hex subgroups, make a set out of {*up_left, *down_right, (x-1, y), (x+1, y)} 
        # discard subgroups with members that don't exist
        if all(coord_to_id(c) in om_list for c in up_left):
            subgroups.append(up_left)
        if all(coord_to_id(c) in om_list for c in down_right):
            subgroups.append(down_right)
        
    return [{coord_to_id(a), coord_to_id(b), coord_to_id(c)} for a, b, c in subgroups]
            
       
        
# -

om_list = df.index
subgroups = triangle_subregions(om_list)
display(type(om_list[0]))

# +


trios_var = []
trios_mn_fano = []
for members in subgroups:
    trios_var.append(df.loc[members].var().sum())
    trios_mn_fano.append((df.loc[members].var()/df.loc[members].mean()).mean())

trio_df = pd.DataFrame(data={'trio': [a+b+c for a, b, c in subgroups], 
                             'total_var': trios_var,
                            'mean_fano': trios_mn_fano})
display(trio_df)
    

# +
n_rand_trios = 10000
rand_trios = []
rand_trios_var = []
rand_fano = []

for i in range(0, n_rand_trios):
    members = random.sample(set(om_list), 3)
    rand_trios.append(members)
    rand_trios_var.append(df.loc[members].var().sum())
    rand_fano.append((df.loc[members].var()/df.loc[members].mean()).mean())
rand_trio_df = pd.DataFrame(data={'trio': rand_trios, 'total_var': rand_trios_var, 'mean_fano': rand_fano})


# -

fig, ax = plt.subplots(1)
sns.distplot(trio_df['mean_fano'], ax=ax)
sns.distplot(rand_trio_df['mean_fano'], ax=ax)


