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

# # Adjacency matrices 
# A quick way to look at:
# - the average adjacency matrix of our lamina circuits 
# - the variance of each type of synaptic connection

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression

from vis.hex_lattice import hexplot
from vis.fig_tools import linear_cmap, subtype_cm

# +
tp = '200914'
cx = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_cxdf.pickle')
subtypes = np.unique([*cx["pre_type"], *cx["post_type"]])

adj_mn = pd.pivot_table(cx, values='n_connect', index='pre_type', columns='post_type')
display(adj_mn.round(decimals=1))
adj_var = pd.pivot_table(cx, values='n_connect', index='pre_type', columns='post_type', aggfunc=np.var)
display((adj_var).round(decimals=1))
adj_sd = pd.pivot_table(cx, values='n_connect', index='pre_type', columns='post_type', aggfunc=np.std)
display(adj_sd.round(decimals=1))

# post_sd_sum = (adj_var**0.5).sum(axis=0)
# display(post_sd_sum)
# -

ctype_order = ['R1R4', 'R2R5', 'R3R6', 'R7', 'R8', 'R7p', 'LMC_1', 'LMC_2', 'LMC_3', 'LMC_4', 'LMC_N', 'centri']
ordered_mn = adj_mn.reindex(ctype_order).reindex(ctype_order, axis=1)
ordered_sd = adj_sd.reindex(ctype_order).reindex(ctype_order, axis=1)


# +
fig, ax = plt.subplots(1, figsize=(10, 8))

labels = ['R1R4', 'R2R5', 'R3R6', 'R7', 'R8', "R7'", 'L1', 'L2', 'L3', 'L4', 'LN', 'Am']

sns.heatmap(ordered_mn.round(), annot=True, ax=ax,cmap='YlOrRd', xticklabels=labels, yticklabels=labels)
ax.set_title('Average ajacency matrix (n = 29)')
fig.savefig("/mnt/home/nchua/Dropbox/200615_mean-adj-order.pdf")

# +
fig, ax = plt.subplots(1, figsize=(10, 8))

sns.heatmap(ordered_sd.round(), annot=True, ax=ax,cmap='YlOrRd', xticklabels=labels, yticklabels=labels)
ax.set_title('Standard deviations of adjacency matrix elements (n = 29)')
fig.savefig("/mnt/home/nchua/Dropbox/200615_sd-adj-order.pdf")
# -

# Individial ommatidia
ommatidia = np.unique(cx['om']).astype(str)
display(ommatidia)
adj_mats = dict.fromkeys(ommatidia)
adj_err = dict.fromkeys(ommatidia)
post_sd = pd.pivot_table(cx, values='n_connect', index='pre_type', columns='post_type', aggfunc=np.var)
for o in ommatidia:
    adj_mats[o] = pd.pivot_table(cx.loc[cx['om'] == o], values='n_connect', index='pre_type', columns='post_type')
    adj_err[o] = adj_mats[o] - adj_mn.round(decimals=0)
    
    display(f"Ommatidium {o}")
    display(adj_mats[o])
#     display("~~Error relative to SD~~")
#     display((adj_err[o]/adj_sd).round(decimals=2))

    display(f"~"*80)




