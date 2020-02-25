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
from scipy.special import comb
from scipy.stats import mannwhitneyu
from typing import Tuple
from itertools import combinations

from src.utils import index_by_om

import matplotlib as mpl
mpl.rc('font', size=14)
# -

tp = '200205'
data_path = f'~/Data/{tp}_lamina/{tp}_cxdf.pickle'
cxdf = pd.read_pickle(data_path)
df_all = index_by_om(cxdf)
criteria = [(df_all[ct].mean() >= 1.0) for ct in df_all.columns]
df = df_all.loc[:, criteria]
om_list = df.index


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
        
    return [sorted([coord_to_id(a), coord_to_id(b), coord_to_id(c)]) for a, b, c in subgroups]
            
       
        
# -



# +
local_trios = triangle_subregions(om_list)

local_var = []
local_fano = []
for trio in local_trios:
    local_var.append(df.loc[trio].var().sum())
    local_fano.append((df.loc[trio].var()/df.loc[trio].mean()).mean())

local = pd.DataFrame(data={'trio': [a+b+c for a, b, c in local_trios], 
                           'total_var': local_var, 'mean_fano': local_fano})

# +
# Draw all possible trios (excluding the neighboring trios) 
non_local_trios = [sorted(list(trio)) for trio in combinations(om_list, 3) if sorted(list(trio)) not in local_trios]
# number of non-local trios should = NC3 - number of neighboring trios, where N is the number of ommatidia
assert(len(non_local_trios) == comb(len(om_list), 3, exact=True) - len(local_trios))

non_local_var = []
non_local_fano = []
for trio in non_local_trios:

    non_local_var.append(df.loc[trio].var().sum())
    non_local_fano.append((df.loc[trio].var()/df.loc[trio].mean()).mean())

non_local = pd.DataFrame(data={'trio': [a+b+c for a, b, c in non_local_trios], 
                               'total_var': non_local_var, 'mean_fano': non_local_fano})
# -

# ## Circuit variability of local retinotopic subgroups
#
#
# Two sample Mann-Whitney U test (one tailed)
#
# $H_{0}: P(\sigma^2_{local} > \sigma^2_{non-local}) >= 1/2 $
#
# The variance of neighboring trios is more than or equal to the variance of non-local trios  
#
# $H_{1}: P(\sigma^2_{local} > \sigma^2_{non-local}) < 1/2 $  
#
# The variance of neighboring trios is less than the variance of non-local trios 

s, p = mannwhitneyu(local['total_var'], non_local['total_var'], alternative='less')
print(f"Test statistic: {s}, p-value: {p: .6f}")

s, p = mannwhitneyu(local['mean_fano'], non_local['mean_fano'], alternative='less')
print(f"Test statistic: {s}, p-value: {p: .6f}")

# +
fig, ax = plt.subplots(1, figsize=[15, 15])

sns.distplot(local['mean_fano'], bins=np.linspace(0,6,64), rug=True, hist=False,
             ax=ax, label=f'Neighboring subgroups (n={len(local_trios)})')
sns.distplot(non_local['mean_fano'], bins=np.linspace(0,6,64), ax=ax, label=f'Non-neighboring subgroups (n={len(non_local_trios)})')

ax.set_title(f'Average fano factor of circuit trios\n(ctypes.mean()>=1.0)\n{tp}')
ax.set_xlabel("Fano-factor of trios's connectivity vectors (averaged across connection type)")
ax.set_ylabel("% trios")

ax.legend()

# +
fig, ax = plt.subplots(1, figsize=[15, 15])

sns.distplot(local['total_var'], bins=np.linspace(0,6,64),
             ax=ax, label=f'Neighboring subgroups (n={len(local_trios)})')
sns.distplot(non_local['total_var'], bins=np.linspace(0,6,64), ax=ax, label=f'Non-neighboring subgroups (n={len(non_local_trios)}')

ax.set_title('The circuitry of neighboring cartridges has less variation than non-neighboring trios')
ax.set_xlabel("Fano-factor of trios's connectivity vectors (averaged across connection type)")
ax.set_ylabel("% trios")

ax.legend()

# -


