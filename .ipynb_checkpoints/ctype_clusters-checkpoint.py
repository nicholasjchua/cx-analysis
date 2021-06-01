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
# -

thresh = cxvecs.mean()>1
cxvecs = cxvecs.loc[:, thresh]
display(cxvecs.columns)

# ## Clustering the correlation matrix of all connections including those between cartridges

# +
cm = subtype_cm() # a dict

cx_corr = cxvecs.corr().dropna(axis=1)
#display(cx_corr)
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.columns, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[15, 15], metric='cosine', cmap='vlag')
# -

# ## Clustering the correlation matrix of only home (intra-ommatidial) connections

homevecs = cxvecs.loc[:, [i for i in cxvecs.columns if '->e' not in i]]
cx_corr = homevecs.corr().dropna(axis=1)
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.columns, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[15, 15], metric='cosine', cmap='vlag')


