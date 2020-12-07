# ---
# jupyter:
#   jupytext:
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

# # Amacrine
# Exploratory analysis specific to the lamina amacrine cell  
# Note: amacrine is annotated as 'centri' in catmaid, so these terms may be used interchangeble. 

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
data_path = f"~/Data/{tp}_lamina/{tp}"
cx = pd.read_pickle(data_path + "_cxdf.pickle")
links = pd.read_pickle(data_path + "_linkdf.pickle")

subtypes = np.unique([*cx['pre_type'], *cx['post_type']])
ommatidia = np.unique(cx['om']).astype(str)
all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  
all_ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]

# +
# Adjacency matrix style pivot table
adj_mn = pd.pivot_table(cx, values='n_connect', index='pre_type', columns='post_type')
adj_var = pd.pivot_table(cx, values='n_connect', index='pre_type', columns='post_type', aggfunc=np.nanvar)
display(f"Mean adjacency matrix across {len(ommatidia)} circuits")
display(adj_mn.astype(int).filter(axis=0, like='centri'))

adj_mats = dict.fromkeys(ommatidia)  # For each ommatidium
for o in ommatidia:
    adj_mats[o] = pd.pivot_table(cx.loc[cx['om'] == o], values='n_connect', index='pre_type', columns='post_type')

# +
df_lamina = pd.DataFrame(index=ommatidia, columns=all_ctype_labels).astype('Int64')

for om, row in df_lamina.iterrows():
    for c in all_ctype_labels:
        pre_t, post_t = c.split('->')
        # Cartridges on the posterior edge lack L4, so their counts for these connections are NaNed 
        if om in ['B0', 'E4', 'E5', 'E6', 'E7', 'D2', 'C1'] and post_t == 'LMC_4':
            df_lamina.loc[om, c] = None
        else:
            df_lamina.loc[om, c] = sum((links.pre_om == om) & (links.post_om == om) & 
                                       (links.pre_type == pre_t) & (links.post_type == post_t))

# +
# Filtering criteria
unknowns = [c for c in df_lamina.columns if 'UNKNOWN' in c]   # discard columns involving connections to unidentified arbors
df = df_lamina.drop(unknowns, axis=1).astype(float).dropna('columns')  # dropna effectively discards L4 associated connections
df = df.loc[:, df.mean() >= 3.0]  # filter out connections with mean less than 1

df = df.rename_axis(index='om')

# +
tmp = dict.fromkeys(ommatidia)
for o in ommatidia:
    if o in ['A0', 'A1', 'A2', 'B0', 'C1', 'D2', 'D3']:
        tmp[o] = 'ventral'
    elif o in ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']:
        tmp[o] = 'dorsal'
    else:
        tmp[o] = 'central'
        
ret_clust = pd.Series(data=tmp, name='ret')
# -

# ## The principle inputs to amacrine are R1-6

am_inputs = pd.DataFrame(data={o: adj_mats[o]['centri'] for o in ommatidia})
fig, ax = plt.subplots(1)
sns.distplot(am_inputs.loc['R1R4'], ax=ax, label='R1R4', kde=False)
sns.distplot(am_inputs.loc["R2R5"], ax=ax, label='R2R5', kde=False)
sns.distplot(am_inputs.loc["R3R6"], ax=ax, label='R3R6', kde=False)
ax.set_xlabel("Number of inputs from short photoreceptor pairs to amacrine")
ax.legend()
display(am_inputs)

# +
fig, ax = plt.subplots(1, figsize=[15, 8])

ret_hues = {'ventral': 'maroon',
           'dorsal': 'darkmagenta',
           'central': 'steelblue'}
cx_order = ['R1R4', 'R2R5', 'R3R6']

display(am_inputs.T.loc[ret_clust=='dorsal'])


ax.plot(am_inputs.T.loc[ret_clust=='dorsal'].mean()[cx_order].to_numpy(), color=ret_hues['dorsal'], linestyle='--')
ax.plot(am_inputs.T.loc[ret_clust=='central'].mean()[cx_order].to_numpy(), color=ret_hues['central'], linestyle='-')
ax.plot(am_inputs.T.loc[ret_clust=='ventral'].mean()[cx_order].to_numpy(), color=ret_hues['ventral'], linestyle='-.')

sns.swarmplot(x='ctype', order=cx_order, y='value', hue='ret', palette=ret_hues, data=melted, ax=ax, size=8)

plt.show()


# +
outputs = df.filter(regex='^centri')
outputs = outputs.reindex(columns=outputs.mean().sort_values().index)

melted = outputs.copy()
melted.loc[:, 'ret'] = ret_clust
melted = melted.melt('ret', value_vars=df.filter(regex='^centri').columns, var_name='ctype')

display(melted)


# +
fig, ax = plt.subplots(1, figsize=[15, 8])

ret_hues = {'ventral': 'maroon',
           'dorsal': 'darkmagenta',
           'central': 'steelblue'}
cx_order = ['centri->R7', 'centri->R7p', 'centri->LMC_1', 'centri->LMC_2', 'centri->LMC_3', 'centri->R2R5', 'centri->R8']


ax.plot(outputs.loc[ret_clust=='dorsal'].mean()[cx_order].to_numpy(), color=ret_hues['dorsal'], linestyle='--')
ax.plot(outputs.loc[ret_clust=='central'].mean()[cx_order].to_numpy(), color=ret_hues['central'], linestyle='-')
ax.plot(outputs.loc[ret_clust=='ventral'].mean()[cx_order].to_numpy(), color=ret_hues['ventral'], linestyle='-.')

sns.swarmplot(x='ctype', order=cx_order, y='value', hue='ret', palette=ret_hues, data=melted, ax=ax, size=8)

plt.show()


# +
centri = df.filter(regex='^centri')
centri.loc[:, 'ret'] = ret_clust
centri = centri.melt('ret', value_vars=df.filter(regex='^centri').columns, var_name='counts')

sns.swarmplot(x='variable', y='value', hue='ret', data=centri_ret, ax=ax)
ax.tick_params(axis='x', rotation=45)
# ax.scatter(centri_outputs.loc[ret_clust =='dorsal'].T, mean_order)
# ax.scatter(centri_outputs.loc[ret_clust =='central'].T, mean_order)
# ax.scatter(centri_outputs.loc[ret_clust =='ventral'].T, mean_order)

# +
fig, ax = plt.subplots(1, fig_size=)
centri_sums = df.filter(regex='^centri').sum(axis=1)

sns.distplot(centri_sums[ret_clust=='dorsal'], color='darkmagenta')
sns.distplot(centri_sums[ret_clust=='central'], color='steelblue')
sns.distplot(centri_sums[ret_clust=='ventral'], color='maroon')
# -

# ### Amacrine output correlation structure (all ommatidia)

c = df.corr()
outputs = df.filter(regex='^centri')
sns.clustermap(c.loc[:, outputs.columns], xticklabels=outputs.columns, yticklabels=df.columns, metric='cosine', cmap=cmap, linewidth=0.7, center=0)

# ## Amacrine output correlation structure (within retinotopic clusters)

dra_c = df.loc[ret_clust == 'dorsal'].corr()
sns.clustermap(dra_c.loc[:, outputs.columns], xticklabels=outputs.columns, yticklabels=df.columns, metric='cosine', cmap=cmap, linewidth=0.7, center=0.0)

ctr_c = df.loc[ret_clust == 'central'].corr()
sns.clustermap(ctr_c.loc[:, outputs.columns], xticklabels=outputs.columns, yticklabels=df.columns, metric='cosine', cmap=cmap, linewidth=0.7, center=0.0)

vra_c = df.loc[ret_clust == 'ventral'].corr()
sns.clustermap(vra_c.loc[:, outputs.columns], xticklabels=outputs.columns, yticklabels=df.columns, metric='cosine', cmap=cmap, linewidth=0.7, center=0.0)


# +
mean_order = df.filter(regex='^centri').mean().sort_values().index
display(mean_order)
centri_outputs = df.filter(regex='^centri').loc[:, mean_order]

# display(centri_outputs.loc[ret_clust =='dorsal'].mean())
# display(centri_outputs.loc[ret_clust =='central'].mean())
# display(centri_outputs.loc[ret_clust =='ventral'].mean())



# +
fig, ax = plt.subplots(1, figsize=[15, 10])

centri = df.filter(regex='^centri')
centri.loc[:, 'ret'] = ret_clust
centri = centri.melt('ret', value_vars=df.filter(regex='^centri').columns, var_name='counts')

sns.swarmplot(x='variable', y='value', hue='ret', data=centri_ret, ax=ax)
ax.tick_params(axis='x', rotation=45)
# ax.scatter(centri_outputs.loc[ret_clust =='dorsal'].T, mean_order)
# ax.scatter(centri_outputs.loc[ret_clust =='central'].T, mean_order)
# ax.scatter(centri_outputs.loc[ret_clust =='ventral'].T, mean_order)
# -

centri_sums = df.filter(regex='^centri').sum(axis=1)
display(centri_sums)
sns.distplot(centri_sums[ret_clust=='dorsal'])
sns.distplot(centri_sums[ret_clust=='central'])
sns.distplot(centri_sums[ret_clust=='ventral'])



# +
out_vecs = pd.DataFrame(data={o: adj_mats[o].loc['centri'] for o in ommatidia})
mn_o = out_vecs.mean(axis=1).sort_values(ascending=False)

mn_o

# +
fig, ax = plt.subplots(1, 2, figsize=[20, 15])

cm = linear_cmap('k')

r7_inputs = cx.filter(items=['R1R4->L3', 'R3R6->L3']).sum(axis=1)
r7cm = linear_cmap(n_vals=r7_inputs.max() - r7_inputs.min(), max_colour=cm['R7'])
node_data = {k: {'colour': cm(v/r7_inputs.max()),
                'label': str(int(v))} for k, v in r7_inputs.items()}
hexplot(node_data=node_data, ax=ax[0])
ax[0].set_title("Number of inputs to R7")

r7p_inputs = cx.filter(items=['centri->R8'])
r7pcm = linear_cmap(n_vals = r7p_inputs.max() - r7p_inputs.min(), max_colour=cm['R7p'])
node_data = {k: {'colour': r7pcm(v/r7p_inputs.max()),
                'label': str(int(v))} for k, v in r7p_inputs.items()}
ax[1].set_title("Number of inputs to R7'")

hexplot(node_data=node_data, ax=ax[1])
plt.show()

# +
order = mn_o.index
am_outputs = cx.loc[cx['pre_type'] == 'centri']
g = sns.FacetGrid(am_outputs, row='post_type', aspect=8, row_order=order[0: -5], sharex=True)

def left_label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)

#g.map(sns.distplot, "n_connect", kde=False, bins=np.arange(0, am_outputs['n_connect'].max() + 1, 3).astype(int))
g.map(sns.boxplot, "n_connect", order=order)
#g.map(left_label, color='k', label="post_type")
g.despine(bottom=True, left=True)
#g.set_titles("")


#g.fig.subplots_adjust(hspace=-.25)
# -

out_corr = out_vecs.corr()
display(out_corr)
sns.clustermap(out_corr, metric='cosine')
#sns.clustermap()

sns.clustermap(out_vecs.T.corr(), metric='cosine')


