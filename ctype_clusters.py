# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: wasp
#     language: python
#     name: wasp
# ---

# # Characterising different lamina connections 
# ## Inputs to LMCs
# - Feedforward inputs from photoreceptors
# - Feedback received from AC and L2 

# +
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression

from src.dataframe_tools import assemble_cxvectors
from vis.hex_lattice import hexplot
from vis.fig_tools import linear_cmap, subtype_cm

# +
plt.rcdefaults()
plt.style.use('vis/lamina.mplstyle')
cm = subtype_cm() # a dict containing the color for each lamina subtype

savefigs = True
savepath = '/Users/nchua/Dropbox (Simons Foundation)/lamina_figures'
# if savefigs:
#     fig.savefig(savepath + '/NAME.svg')
#     fig.savefig(savepath + '/NAME.png')
# -

tp = '200914'
linkdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')
cxdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_cxdf.pickle')


# ## Connections observed in the lamina
# - When opposite pairs of short photoreceptors are combined (i.e. R1R4->L3 = R1->L3 + R4->L3), there are 13 classes of cells that participate in the lamina circuit
# - The 12 subtypes can form at most 169 pre->post permutations. Of the 169 possibilities, we observed 161 of them at least once throughout the whole lamina
# - 106 of the 161 connections observed (more than 65%) were found at an average rate of less than 1 synapse per ommatidia
# - Connections that are entirely absent in some ommatidium (e.g. R1R4->eLMC_4) can still be strongly connected (large average synapse count despite many zeros) 

cxvecs = assemble_cxvectors(linkdf).astype(float)   # each ommatidium has a vector of the different connection counts
# Default args: external=True, interom connections included; excl_unknowns=True, filter cxs w unknown partner
cxs_before = cxvecs.columns  # hold on to all connection types incl. those with sub-threshold averages

# +
nan_missing_L4s = True

if nan_missing_L4s:
    L4_cx = [c for c in cxvecs.columns if (c[-5: len(c)] == 'LMC_4')]
    cxvecs.loc[['B0', 'C1', 'D2', 'E4', 'E5', 'E6', 'E7'], L4_cx] = np.nan
# -

display(f"Number of connections observed (all): {len(cxs_before)}")
mean_thresh = 0.5
#cxvecs = cxvecs.loc[:, cxvecs.mean() >= mean_thresh]
display(f"Number of connections observed (where mean count >= {mean_thresh}): {len(cxvecs.columns)}")

only_home_connections = True
if only_home_connections:
    cxvecs = cxvecs.loc[:, [i for i in cxvecs.columns if ('eLMC_4' not in i) and ('eLMC_2' not in i)]]
display(f"Number of connections after removing inter-ommatidial: {len(cxvecs.columns)}")

# +
from itertools import product
svfs = ['R1R4', 'R2R5', 'R3R6']
lmcs = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4', 'LMC_N']
feedforward = [pre + '->' + post for pre, post in product(svfs, lmcs)]


svf_lmc = cxvecs.filter(items=feedforward)
combined_svf = pd.DataFrame()
cp_ratio = pd.DataFrame()

for l in lmcs:
    combined_svf[f'R1R4+R3R6->{l}'] = svf_lmc[f'R1R4->{l}'] + svf_lmc[f'R3R6->{l}']
    combined_svf[f'R2R5->{l}'] = svf_lmc[f'R2R5->{l}']
    cp_ratio[l] = (combined_svf[f'R2R5->{l}'])/ combined_svf[f'R1R4+R3R6->{l}']

# +
fig, ax = plt.subplots(1, figsize=[2.3, 4.6])
m = {'LMC_N': 'D', 'LMC_1': 's', 'LMC_2': '^', 'LMC_3': 'o', 'LMC_4': 'P'}
for l in lmcs:
    ax.scatter(x=combined_svf.loc[:, f'R2R5->{l}'], y=combined_svf.loc[:, f'R1R4+R3R6->{l}'], 
               c=cm[l], marker=m[l], s=10.0, label=f"L{l[-1]}")
ax.legend()
ax.set_aspect('equal')
ax.set_title('R1-6 outputs to LMCs')
ax.set_xlabel('# inputs from R1, R3, R4, R6')
ax.set_ylabel('# inputs from R2, R5')

if savefigs:
    fig.savefig(savepath + '/LMC_r2r5-r1r3r4r6.svg')
    fig.savefig(savepath + '/LMC_r2r5-r1r3r4r6.png')

# +
fig, ax = plt.subplots(1, figsize=[2.3, 2.3])
m = {'LMC_N': 'D', 'LMC_1': 's', 'LMC_2': '^', 'LMC_3': 'o', 'LMC_4': 'P'}
for l in lmcs:
    ax.scatter(x=cxvecs.loc[:, f'centri->{l}'], y=cxvecs.loc[:, f'LMC_2->{l}'], 
               c=cm[l], marker=m[l], s=10.0, label=f"L{l[-1]}")
#ax.legend()
ax.set_aspect('equal')
ax.set_title('LMC feedback')
ax.set_xlabel('# inputs from AC')
ax.set_ylabel('# inputs from L2')

if savefigs:
    fig.savefig(savepath + '/LMC_feedback_scatter.svg')
    fig.savefig(savepath + '/LMC_feedback_scatter.png')
# -

# ### Feedback connections
# Scatterplot depicting the number of inputs each LMC receives from AC and L2 in the lamina. LN receives few feedback connections from AC and L2. 

# +
fig, ax = plt.subplots(1, figsize=[2.3, 2.3])
post_types = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4']
colors = [cm[s] for s in post_types]

bplot = ax.boxplot([cp_ratio[p].dropna() for p in post_types], patch_artist=True, medianprops={'color': 'k', 'linestyle': '--'})
# fill box with subtype color
for b, c in zip(bplot['boxes'], colors):
    b.set_facecolor(c)

ax.set_title('Bias for R2 and R5 inputs')
ax.set_xticklabels(['L1', 'L2', 'L3', 'L4'])
ax.set_xlabel('Subtype')
ax.set_ylabel(r'# inputs $\frac{R2+R5}{R1+R3+R4+R6}$')

if savefigs:
    fig.savefig(savepath + '/LMC_r1-6_bias.svg')
    fig.savefig(savepath + '/LMC_r1-6_bias.png')
    

# -

# ### LMC subtypes sample from R1-6 at different proportions
# **Boxplot depicting the ratio of photoreceptor inputs received by L1-4**. L1 and L2 receive approximately the same proportion of inputs from R2 and R5 as they do from each of the other four short photoreceptors (R1, R3, R4, and R6). L3 and L4 show a bias for R2 and R5 inputs. Boxes depict the interquartile range observed among lamina cartridges. Whiskers and point beyond describe data points beyond the interquartile range.  

# +
# fig, ax = plt.subplots(1, figsize=[2.3, 2.3])
# post_types = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4']
# colors = [cm[s] for s in post_types]

# bplot = ax.boxplot([cxvecs[f'centri->{p}'].dropna()/cxvecs[f'LMC_2->{p}'].dropna() for p in post_types], 
#                    patch_artist=True, medianprops={'color': 'k', 'linestyle': '--'})
# # fill box with subtype color
# for b, c in zip(bplot['boxes'], colors):
#     b.set_facecolor(c)

# ax.set_title('Bias for R2 and R5 inputs')
# ax.set_xticklabels(['L1', 'L2', 'L3', 'L4'])
# ax.set_xlabel('Subtype')
# ax.set_ylabel(r'# inputs $\frac{R2+R5}{R1+R3+R4+R6}$')

# if savefigs:
#     fig.savefig(savepath + '/LMC_r1-6_bias.svg')
#     fig.savefig(savepath + '/LMC_r1-6_bias.png')
    

# -

# ## Clustering the correlation matrix of all connections including those between cartridges

# ## Clustering the correlation matrix of only home (intra-ommatidial) connections

# +
print(f"All Home Connections")
print(f"{len(data.columns)} connection types")

cx_corr = data.corr().dropna()
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.index, #figsize=[13, 13],
                      row_colors=row_colors, col_colors=col_colors,
                      metric='cosine', method='complete',
                      cmap='vlag', vmax=1.0, vmin=-1.0)
#clus.savefig('/Users/nchua/clus.png')
#print(clus.get('colors_ratio'))
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")

# +
print("Feedforward connections correlations with others")
print(f"{len(data.columns)} connection types")

cx_corr = data.corr().dropna().loc[:, feedforward]
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.index, #figsize=[13, 13],
                      row_colors=row_colors, col_colors=col_colors,
                      metric='cosine', method='complete',
                      cmap='vlag', vmax=1.0, vmin=-1.0)
#clus.savefig('/Users/nchua/clus.png')
#print(clus.get('colors_ratio'))
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")

# +
# EXCLUDE INTEROM CONNECTIONS
data = cxvecs.loc[:, [i for i in cxvecs.columns if '->e' not in i]]
print(f"{len(data.columns)} connection types")

cx_corr = data.corr().dropna().filter(items=feed_forward)
#cx_corr = cx_corr.loc[[c for c in cx_corr.index if c not in feed_forward]]
cx_corr = cx_corr.filter(regex='->R[7-8]', axis=0)

row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.index, #figsize=[13, 13],
                      row_colors=row_colors, col_colors=col_colors,
                      metric='cosine', method='complete',
                      cmap='vlag', vmax=1.0, vmin=-1.0)
clus.savefig('/Users/nchua/clus.png')
#print(clus.get('colors_ratio'))
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")

# +
# EXCLUDE INTEROM CONNECTIONS
data = cxvecs.loc[:, [i for i in cxvecs.columns if '->e' not in i]]
print(f"{len(data.columns)} connection types")

cx_corr = data.corr().dropna().filter(items=feed_forward)
cx_corr = cx_corr.loc[[c for c in cx_corr.index if c not in feed_forward]]
#cx_corr = cx_corr.filter(regex='->R[7-8]', axis=0)

row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.index, #figsize=[13, 13],
                      row_colors=row_colors, col_colors=col_colors,
                      metric='cosine', method='average',
                      cmap='vlag', vmax=1.0, vmin=-1.0)
clus.savefig('/Users/nchua/clus.png')
#print(clus.get('colors_ratio'))
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")

# +
# EXCLUDE INTEROM CONNECTIONS

#data = cxvecs.loc[:, [i for i in cxvecs.columns if 'LMC_4' not in i]]
data = cxvecs.loc[:, [i for i in cxvecs.columns if 'LMC_N' not in i]]
print(f"{len(data.columns)} connection types")

y_connections = [c for c in cx_corr.index if c not in [*feed_forward]]

cx_corr = data.corr().dropna()
cx_corr = cx_corr.loc[[c for c in cx_corr.index if c not in feed_forward],feed_forward]
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.index, 
                      row_colors=row_colors, col_colors=col_colors,
                      metric='euclidean', method='complete',
                      cmap='vlag', vmax=1.0, vmin=-1.0)
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")

# +
# EXCLUDE INTEROM CONNECTIONS

#data = cxvecs.loc[:, [i for i in cxvecs.columns if 'LMC_4' not in i]]
data = cxvecs.loc[:, [i for i in data.columns if 'LMC_N' not in i]]
data = data.loc[:, [i for i in data.columns if i.split('->')[1][0:3] == 'LMC']]
print(f"{len(data.columns)} connection types")

cx_corr = data.corr().dropna(axis=1)
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.columns, 
                      row_colors=row_colors, col_colors=col_colors,
                      metric='cosine', method='complete',
                      cmap='vlag', vmax=1.0, vmin=-1.0)
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")

# +
# EXCLUDE INTEROM CONNECTIONS

data = cxvecs.loc[:, [i for i in cxvecs.columns if 'LMC_4' not in i]]
data = data.loc[:, [i for i in data.columns if 'LMC_N' not in i]]
data = data.loc[:, [i for i in data.columns if i.split('->')[1][0:3] == 'LMC']]
print(f"{len(data.columns)} connection types")

cx_corr = data.corr().dropna()
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.columns, 
                      row_colors=row_colors, col_colors=col_colors,
                      metric='euclidean', method='complete',
                      cmap='vlag', vmax=1.0, vmin=-1.0)
#clus.savefig("/mnt/home/nchua/Dropbox/200610_ctype_clus.svg")

# +
# EXCLUDE INTEROM CONNECTIONS

data = cxvecs.filter(like='->LMC')

print(f"{len(data.columns)} connection types")

cx_corr = data.corr().dropna(axis=1)
row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
col_colors = [cm[x.split('->')[1]] for x in cx_corr.columns]
clus = sns.clustermap(cx_corr, xticklabels=cx_corr.columns, yticklabels=cx_corr.columns, 
                      row_colors=row_colors, col_colors=col_colors,
                      metric='cosine', 
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
               figsize=[11, 11], metric='euclidean', method='complete',
               cmap='vlag', vmax=1.0, vmin=-1.0)

# +
centri_corr = cx_corr.filter(like='LMC_2->')

row_colors = [cm[x.split('->')[0]] for x in centri_corr.index]
col_colors = [cm[x.split('->')[1]] for x in centri_corr.columns]
sns.clustermap(centri_corr, xticklabels=centri_corr.columns, yticklabels=centri_corr.index, 
               row_colors=row_colors, col_colors=col_colors, linewidth=0.1,
               figsize=[11, 11], metric='cosine', 
               cmap='vlag', vmax=1.0, vmin=-1.0)
# +
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
# +
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
# +
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



