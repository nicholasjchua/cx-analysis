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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Clustering ommatidia by connections

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
# -



# +
# Load dataframe of om->[connection counts]
tp = "200507"
lamina_links = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')
subtypes = np.unique([*lamina_links["pre_type"], *lamina_links["post_type"]])

all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  
all_ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]
ommatidia = ommatidia = np.unique(lamina_links['pre_om'])

cxvecs = assemble_cxvectors(lamina_links)

# df_lamina = pd.DataFrame(index=ommatidia, columns=all_ctype_labels).astype('Int64')

# for om, row in df_lamina.iterrows():
#     for c in all_ctype_labels:
#         pre_t, post_t = c.split('->')
#         # Cartridges on the posterior edge lack L4, so their counts for these connections are NaNed 
#         if om in ['B0', 'E4', 'E5', 'E6', 'E7', 'D2', 'C1'] and post_t == 'LMC_4':
#             df_lamina.loc[om, c] = 0
#             # df_lamina.loc[om, c] if you want to remove the L4 connections completely
#         else:
#             df_lamina.loc[om, c] = sum((lamina_links.pre_om == om) & (lamina_links.post_om == om) & 
#                                        (lamina_links.pre_type == pre_t) & (lamina_links.post_type == post_t))

# +
# Filtering criteria
#unknowns = [c for c in df_lamina.columns if 'UNKNOWN' in c]   # discard columns involving connections to unidentified arbors
#df = df_lamina.drop(unknowns, axis=1).astype(float).dropna('columns')  # dropna effectively discards L4 associated connections
thresh = cxvecs.mean() > 1
cxvecs = cxvecs.loc[:, thresh].fillna(0)  # filter out connections with mean less than 1

cxvecs = cxvecs.rename_axis(index='om')
# -

# ## Clustering by all connection types

# +

region_color = {'A0': 'b',
                'A1': 'b',
               'A2': 'b',
               'A3': 'gray',
               'A4': 'gray',
               'A5': 'm',
               'B0': 'b',
               'B1': 'darkgreen',
               'B2': 'gray',
               'B3': 'b',
               'B4': 'gray',
               'B5': 'm',
               'B6': 'm',
               'C1': 'darkgreen',
               'C2': 'darkgreen',
               'C3': 'b',
               'C4': 'gray',
               'C5': 'gray',
               'C6': 'm',
               'D2': 'b',
               'D3': 'b',
               'D4': 'b',
               'D5': 'gray',
               'D6': 'm',
               'D7': 'm',
               'E4': 'gray',
               'E5': 'gray',
               'E6': 'gray',
               'E7': 'm'}
region_list = list(region_color.values())
# -

display(len(cxvecs.T))
clus = sns.clustermap(cxvecs.T, row_cluster=False, col_colors=region_list, yticklabels=cxvecs.columns, metric='cosine',
                      cmap='Reds')
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ret-clus.svg")

homedf = cxvecs.loc[:, [i for i in cxvecs.columns if ('LMC_4' not in i) and ('eLMC_2' not in i)]]
display(len(homedf.T))
cbar_kws = {'label': 'Connection Counts'}
clus = sns.clustermap(homedf.T, row_cluster=False, col_colors=region_list, figsize=[12,12], cmap='Reds',
                      yticklabels=homedf.T.index, metric='cosine', cbar_kws=cbar_kws)
clus.savefig("/mnt/home/nchua/Dropbox/200610_ret-clus.svg")

# +
om_corr = cxvecs.T.corr()


sns.clustermap(om_corr, metric='cosine', 
               col_colors=region_list,linewidth=0.5, cmap='YlGnBu', vmin=0.75, vmax=1)
# sns.clustermap(om_corr, xticklabels=om_corr.columns, yticklabels=om_corr.columns, metric='cosine',
#                row_colors=region_color,linewidth=0.5)

# +
fig, ax = plt.subplots(1, figsize=[8, 12])
hexplot(node_data={k: {'colour': v} for k, v in region_color.items()}, ax=ax)

fig.savefig(fig.savefig("/mnt/home/nchua/Dropbox/200610_clus-assign.svg"))
# -




