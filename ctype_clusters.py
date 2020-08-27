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

# # Correlated connection types across ommatidia
# - The connection counts of the basic lamina circuit varies across ommatidia
# - Do certain connection counts vary in concert with others? 
# - Can we group correlated connection types into clusters? 
#
# 1. Calculate (across ommatidia) correlation matrix containing an element for each pairwise correlation between each connection count
# 2. Sort rows and columns of correlation matrix to minimize distance between each connection type's vector of correlation coefficients
# 3. Present a hierachy of connection types with similar correlation vectors 
#
#
# 4. Is the correlation structure different when r is computed within retinotopic clusters

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression

from src.dataframe_tools import assemble_cxvectors
from vis.hex_lattice import hexplot
from vis.colour_palettes import subtype_cm
from vis.fig_tools import linear_cmap

# +
tp = '200507'
linkdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')
cxdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_cxdf.pickle')

cxvecs = assemble_cxvectors(linkdf)

# +
thresh = cxvecs.mean()>1.0  # what about >= ?? CHECK
cxvecs = cxvecs.loc[:, thresh]

cm = subtype_cm() # a dict
# -

cxvecs

# ## Clustering the correlation matrix of all connections including those between cartridges

# +
# INCLUDING INTEROM CONNECTIONS

# cx_corr = cxvecs.corr().dropna(axis=1)
# display(len(cx_corr))
# row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
# col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
# sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.columns, 
#                row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
#                figsize=[15, 15], metric='cosine', cmap='vlag')
# -

# ## Clustering the correlation matrix of only home (intra-ommatidial) connections

# +
# EXCLUDE INTEROM CONNECTIONS

homevecs = cxvecs.loc[:, [i for i in cxvecs.columns if '->e' not in i]]
print(f"{len(homevecs.columns)} connection types")

cx_corr = homevecs.corr().dropna(axis=1)
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.columns, 
                      row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
                      figsize=[11, 11], metric='cosine', 
                      cmap='vlag', vmax=1.0, vmin=-1.0)
clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")
# -

# Cluster 1: L2->L3, centri->(L1, L2, L3, R1-6, R7p), (L2, R1-6, centri)->centri
#
# Cluster 2: (R1R4, R3R6)->R2R5, centri->L4, (R1R4, R3R6)->LN, (L2, centri)->R7
#
# Cluster 3: R2R5->(L1, L2), LN->(L1, L2, L3), L2->L4, (R1-6)->L4, L2->L1
#
# Cluster 4: *L2->R2R5*, (R2R5, centri, L2)->R8, (R1R4, R3R6)->L3, (R1R4, R3R6)->L1, R2R5->R2R5, (R2R5, L2)->R7p
#
# Cluster 5: L2->L2, (R1R4, R3R6)->L2, R2R5->L3, (R1R4, R3R6)->R7p 
#

# ## Amacrine connections
# - All inputs to am are present in the 1st cluster, only am outputs outside this are centri->L4, centri->R7, centri->R8

# +
centri_corr = cx_corr.filter(like='centri')

row_colors = [cm[x.split('->')[0]] for x in centri_corr.index]
col_colors = [cm[x.split('->')[1]] for x in centri_corr.columns]
sns.clustermap(centri_corr, xticklabels=centri_corr.columns, yticklabels=centri_corr.index, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[11, 11], metric='cosine', 
               cmap='vlag', vmax=1.0, vmin=-1.0)

# +
centri_corr = cx_corr.filter(like='LMC_2->')

row_colors = [cm[x.split('->')[0]] for x in centri_corr.index]
col_colors = [cm[x.split('->')[1]] for x in centri_corr.columns]
sns.clustermap(centri_corr, xticklabels=centri_corr.columns, yticklabels=centri_corr.index, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[11, 11], metric='cosine', 
               cmap='vlag', vmax=1.0, vmin=-1.0)
# + {}
df_allom = homevecs.filter(like='->R7')
#df_allom = df_allom.loc[:, [i for i in df_allom.columns if (i[0] != 'R')]]
all_corr = df_allom.corr().dropna(axis=1)
all_corr = all_corr.loc[[i for i in df_allom.columns if (i[0] != 'R')],
                       [i for i in df_allom.columns if (i[0] != 'R')]]

row_colors = [cm[x.split('->')[0]] for x in all_corr.index]
col_colors = [cm[x.split('->')[1]] for x in all_corr.columns]
sns.clustermap(all_corr, xticklabels=all_corr.columns, yticklabels=all_corr.index, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[11, 11], metric='cosine', 
               cmap='vlag', vmax=1.0, vmin=-1.0)
# + {}
#dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
dra_om = ['A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E7']
df_dra = homevecs.filter(items=dra_om, axis=0).filter(like='->R7')
#df_dra = df_dra.loc[:, [i for i in df_allom.columns if (i[0] != 'R')]]
dra_corr = df_dra.corr().dropna(axis=1)
dra_corr = dra_corr.loc[[i for i in df_allom.columns if (i[0] != 'R')], 
                        [i for i in df_allom.columns if (i[0] != 'R')]]

row_colors = [cm[x.split('->')[0]] for x in dra_corr.index]
col_colors = [cm[x.split('->')[1]] for x in dra_corr.columns]
sns.clustermap(dra_corr, xticklabels=dra_corr.columns, yticklabels=dra_corr.index, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[11, 11], metric='cosine', 
               cmap='vlag', vmax=1.0, vmin=-1.0)
# + {}
#dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
ndra_om = [i for i in homevecs.index if i not in dra_om]
display(ndra_om)
df_dra = homevecs.filter(items=ndra_om, axis=0).filter(like='->R7')
#df_dra = df_dra.loc[:, [i for i in df_allom.columns if (i[0] != 'R')]]
dra_corr = df_dra.corr().dropna(axis=1)
dra_corr = dra_corr.loc[[i for i in df_allom.columns if (i[0] != 'R')], 
                        [i for i in df_allom.columns if (i[0] != 'R')]]

row_colors = [cm[x.split('->')[0]] for x in dra_corr.index]
col_colors = [cm[x.split('->')[1]] for x in dra_corr.columns]
sns.clustermap(dra_corr, xticklabels=dra_corr.columns, yticklabels=dra_corr.index, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[11, 11], metric='cosine', 
               cmap='vlag', vmax=1.0, vmin=-1.0)
# -
fig, ax = plt.subplots()
sns.regplot(x=homevecs['R1R4->R2R5'] + homevecs['R3R6->R2R5'], y=homevecs['centri->R8'], ax=ax)
ax.set_xlabel('centri -> R8')
ax.set_ylabel('R1,R3,R4,R6 -> R2,R5')

fig, ax = plt.subplots()
sns.regplot(x=homevecs.filter(like='->R8').sum(axis=1), y=homevecs['R1R4->R2R5'] + homevecs['R3R6->R2R5'],
            x_jitter=0.5, y_jitter=0.5, ax=ax)
ax.set_xlabel('Inputs to R8')
ax.set_ylabel('R1, R3, R4, R6 -> R2, R5')

# +
# pairs = itertools.combinations(homevecs.columns, 2)
# for c1, c2 in pairs:
#     sns.regplot(x=homevecs[c1], y=homevecs[c2])
#     plt.show()
# -



