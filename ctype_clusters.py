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

# # Characterising different lamina connections 
#
# Is there a useful way to group the different connection types by the variability of their counts? 
# - Fixed/invariant -> Observed in most circuits, mean counts high/var low
# - Retinotopically-dependent -> High variance, but the variance of these connection counts decrease 
# - Stochastic/variant/noise
#
# We observed many of the possible 'pre->post' permutations that can be formed between each lamina subtype
# 1. Neurons and their synapses organize in ways that expend more resources to form connections that are necessary for function, while minimizing errors that expend limited synaptic contact area or have a negative influence on performance. 
# 2. Under normal circumstances, costs associated with the amount of space occupied by a system puts constrains on network architecture and synaptic connectivity. For Megaphragma, an insect that is subject to extreme evolutionary pressure for miniaturization, space constraints could play a dominant role in the design of its neural circuits. 
# 3. 
#
# If a given connection (between a pre->post pair of neurons) is facilitated by many synapses, this connection is more likely to be a physiologically meaningful feature of the system compared to a different connection with fewer synapses. (the high metabolic/space cost of maintaining many synapses) 
# 2. If a given connection is observed throughout many reconstructions of the same circuit, this connection is more likely to be physiologically meaningful
#
# 1. Calculate (across ommatidia) correlation matrix containing an element for each pairwise correlation between each connection count
# 2. Sort rows and columns of correlation matrix to minimize distance between each connection type's vector of correlation coefficients
# 3. Present a hierachy of connection types with similar correlation vectors 

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
from vis.fig_tools import linear_cmap, subtype_cm
# -

tp = '200914'
linkdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')
cxdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_cxdf.pickle')


# ## Connections observed in the lamina
# - When opposite pairs of short photoreceptors are combined (i.e. R1R4->L3 = R1->L3 + R4->L3), there are 13 classes of cells that participate in the lamina circuit
# - The 12 subtypes can form at most 169 pre->post permutations. Of the 169 possibilities, we observed 161 of them at least once throughout the whole lamina
# - 106 of the 161 connections observed (more than 65%) were found at an average rate of less than 1 synapse per ommatidia
# - Connections that are entirely absent in some ommatidium (e.g. R1R4->eLMC_4) can still be strongly connected (large average synapse count despite many zeros) 

# +
cxvecs = assemble_cxvectors(linkdf).astype(float)   # each ommatidium has a vector of the different connection counts
# Default args: external=True, interom connections included; excl_unknowns=True, filter cxs w unknown partner
cxs_before = cxvecs.columns  # hold on to all connection types incl. those with sub-threshold averages
nan_missing_L4s = True

if nan_missing_L4s:
    L4_cx = [c for c in cxvecs.columns if (c[-5: len(c)] == 'LMC_4')]
    cxvecs.loc[['B0', 'C1', 'D2', 'E4', 'E5', 'E6', 'E7'], L4_cx] = np.nan

display(f"Number of connections observed (all): {len(cxs_before)}")
mean_thresh = 1.0
cxvecs = cxvecs.loc[:, cxvecs.mean() >= mean_thresh]


display(f"Number of connections observed (where mean count >= {mean_thresh}): {len(cxvecs.columns)}")
display(f"Presyn subtypes: {linkdf['pre_type'].unique()}")
display(f"Postsyn subtypes: {linkdf['post_type'].unique()}")
display(f"Number of different connections: {len(cxvecs.columns)}")
cm = subtype_cm() # a dict
# -

c = 'LMC_2->LMC_4'
c[-5:len(c)]

# +
summary = pd.DataFrame([cxvecs.mean(), cxvecs.var(), cxvecs.std()]).T
summary = summary.rename(columns={0: 'mean', 1: 'var', 2: 'sd'})
summary = summary.rename_axis('cx')

summary['cv'] = summary['sd'] / summary['mean']
summary['ff'] = summary['var'] / summary['mean']

#fig, ax = plt.subplots()
#ax.set_title('fano')
sns.jointplot(data=summary, x='mean', y='var', marker='+')
# -

display(summary.filter(axis=0, like='->eLMC').sort_values(by='cx', ascending=True))
display(summary.filter(axis=0, like='centri->'))

# +
from itertools import product
svfs = ['R1R4', 'R2R5', 'R3R6']
lmcs = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4']
these_cx = [pre + '->' + post for pre, post in product(svfs, lmcs)]


svf_lmc = cxvecs.filter(items=these_cx)
combined_svf = pd.DataFrame()
cp_ratio = pd.DataFrame()

for l in lmcs:
    combined_svf[f'R1R4+R3R6->{l}'] = svf_lmc[f'R1R4->{l}'] + svf_lmc[f'R3R6->{l}']
    combined_svf[f'R2R5->{l}'] = svf_lmc[f'R2R5->{l}']
    cp_ratio[l] = (2.0 * combined_svf[f'R2R5->{l}'])/ combined_svf[f'R1R4+R3R6->{l}']
# -

combined_svf.describe()

cp_ratio.describe()

sns.distplot(summary['cv'])

sns.distplot(summary['ff'])

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
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")
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



