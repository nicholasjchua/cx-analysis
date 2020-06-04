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

# # Photoreceptor Diversity

# +
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from dataframe_org import extract_connector_table
from src.cartridge_metadata import ret_clusters
from vis.colour_palettes import subtype_cm

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
mpl.rc('font', size=14)
mpl.rc('figure', figsize=[12, 12])

# +
# synaptic links (contacts)
tp = '200507'
data_path = f'~/Data/{tp}_lamina/{tp}_linkdf.pickle'
df = pd.read_pickle(data_path)

# Rhabdom vols from Anastasia
rb = pd.read_csv('~/Data/lamina_additional_data/ret_cell_vol.csv').set_index('rtype').T
rb.index.name = 'om'
rb = rb.loc[sorted(rb.index), sorted(rb.columns)]
rb_frac = (rb.T/rb.sum(axis=1)).T

rtypes = rb.columns
subtypes = np.unique(df['post_type'])
ommatidia = np.unique(df['pre_om'])

ret_clust = ret_clusters()

cm = subtype_cm()

ct_df = extract_connector_table(df) # DataFrame of connectors (presyn terminals)

# +
# We are interested in both the number of contacts and the number of terminals 
n_terminals = {om: dict.fromkeys(rtypes) for om in ommatidia}
n_contacts = {om: dict.fromkeys(rtypes) for om in ommatidia}
n_outs = df['pre_neuron'].value_counts().to_dict() # count of links associated with every neuron

# Filter out non-short PR contacts/terminals
# TODO: helper function to add column for sub sub type (e.g. 'R1' instead of R1R4)
# This can be done from link_df, TODO: MAKE MORE GENERAL AND PUT IN DATAFRAME TOOLS
for pre_name, these_rows in ct_df.groupby('pre_neuron'):
    # using our neuron name pattern to get ommatidium/rtypes of indv photoreceptors
    if pre_name[0:2] == 'om' and pre_name[5] == 'R':  
        om = pre_name[2: 4]
        r = pre_name.split('_')[1]
        assert(len(r) in (2, 3))
        n_terminals[om][r] = len(these_rows)
        n_contacts[om][r] = n_outs.get(pre_name, 0)
    else:
        continue

terms = pd.DataFrame(n_terminals).fillna(0).astype(int).T
ctx = pd.DataFrame(n_contacts).fillna(0).astype(int).T

terms.index.name = 'om'
ctx.index.name = 'om'

# +
# Longform dataframes for seaborn functions
# Alternatives: number of contacts or number of pre terminals, rhabdomere vol or faction of rhabdom total
# Decide which is concatenated in combined

l_terms = pd.melt(terms.reset_index(), id_vars='om', value_vars=terms.columns, var_name='subtype', value_name='syn_count')
l_rbfrac = pd.melt(rb_frac.reset_index(), id_vars='om', value_vars=rb_frac.columns,  var_name='subtype', value_name='vol')

l_ctx = pd.melt(ctx.reset_index(), id_vars='om', value_vars=ctx.columns, var_name='subtype', value_name='syn_count')
l_rb = pd.melt(rb.reset_index(), id_vars='om', value_vars=rb.columns,  var_name='subtype', value_name='vol')

combined = pd.concat([l_rbfrac, l_ctx['syn_count']], axis=1)
# combined = pd.concat([l_rbfrac, l_ctx['syn_count']], axis=1)
#combined = pd.concat([l_rb, l_ctx['syn_count']], axis=1)

# -


# ## Differences between short photoreceptor subtypes 

# +
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]
xmax = data['vol'].max()
ymax = data['syn_count'].max()

g = sns.JointGrid(x="vol", y="syn_count", data=data, height=8,
                  xlim=[0, xmax + (xmax*0.1)], ylim=[0, ymax + (ymax*0.1)])


spr_pairs = (('R1', 'R4'), ('R2' , 'R5'), ('R3', 'R6'))
pt = ('o', 'x', '+')

for i, p in enumerate(spr_pairs):
    rows = (data['subtype'] == p[0]) | (data['subtype'] == p[1])
    
    c = cm[p[0]+p[1]]
    
    g.ax_joint.scatter(x=data.loc[rows, 'vol'], y=data.loc[rows, 'syn_count'], label=f'{p[0]}/{p[1]}', marker=pt[i], color=c)
    g.ax_joint.legend()
    
    sns.kdeplot(data.loc[rows, 'vol'], legend=False, ax=g.ax_marg_x, color=c)
    sns.kdeplot(data.loc[rows, 'syn_count'], legend=False, ax=g.ax_marg_y, vertical=True, color=c)
    
g.ax_joint.set_ylabel('Number of postsynaptic contacts')
g.ax_joint.set_xlabel("Fraction of fused rhabdom volume")
#g.ax_joint.set_xlabel("Rhabdomere volume (\u03BC" + "$m^3$)")



# +

combined = pd.concat([l_rb, l_ctx['syn_count']], axis=1)
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]
xmax = data['vol'].max()
ymax = data['syn_count'].max()

g = sns.JointGrid(x="vol", y="syn_count", data=data, height=8,
                  xlim=[0, xmax + (xmax*0.1)], ylim=[0, ymax + (ymax*0.1)])


spr_pairs = (('R1', 'R4'), ('R2' , 'R5'), ('R3', 'R6'))
pt = ('o', 'x', '+')

for i, p in enumerate(spr_pairs):
    rows = (data['subtype'] == p[0]) | (data['subtype'] == p[1])
    
    c = cm[p[0]+p[1]]
    
    g.ax_joint.scatter(x=data.loc[rows, 'vol'], y=data.loc[rows, 'syn_count'], label=f'{p[0]}/{p[1]}', marker=pt[i], color=c)
    g.ax_joint.legend()
    
    sns.kdeplot(data.loc[rows, 'vol'], legend=False, ax=g.ax_marg_x, color=c)
    sns.kdeplot(data.loc[rows, 'syn_count'], legend=False, ax=g.ax_marg_y, vertical=True, color=c)
    
g.ax_joint.set_ylabel('Number of postsynaptic contacts')
g.ax_joint.set_xlabel("Rhabdomere volume (\u03BC" + "$m^3$)")
#g.ax_joint.set_xlabel("Rhabdomere volume (\u03BC" + "$m^3$)")
# -

# ### Hypothesis test (number of outputs) 

# +
central = [*ctx['R2'].tolist(), *ctx['R5'].tolist()]
periperal = [*ctx['R1'].tolist(), *ctx['R4'].tolist(), *ctx['R3'].tolist(), *ctx['R6'].tolist()]

s, p = mannwhitneyu(central, periperal, alternative='greater')
print("###### RESULTS ######")
print(f"Test statistic: {s}, p-value: {p}")
if p > 0.001:
    print("Fail to reject null at p = 0.001")
else:
    print("Reject null: R2R5 inputs significantly larger")
# -

# ### Hypothesis test (rhabdomere volume)

# +
central = [*rb['R2'].tolist(), *rb['R5'].tolist()]
periperal = [*rb['R1'].tolist(), *rb['R4'].tolist(), *rb['R3'].tolist(), *rb['R6'].tolist()]

s, p = mannwhitneyu(central, periperal, alternative='greater')
print("###### RESULTS ######")
print(f"Test statistic: {s}, p-value: {p}")
if p > 0.001:
    print("Fail to reject null at p = 0.001")
else:
    print("Reject null: R2R5 rhabdomere volume significantly larger")
# -

# ## Variability of long photoreceptor volumes

# +
# Rhabdomere volume
data = rb.filter(items=['R7', 'R8', 'R7p'], axis=1)

fig, ax = plt.subplots(1, figsize=[10, 15])
for i, row in data.iterrows():
    if str(i) in ret_clust['dra']:
        l = '--'
    #elif str(i) in [*ret_clust['v_trio'], *ret_clust['vra']]:
#     elif str(i) in ret_clust['vra']:
#         l = '-.'
    else:
        l = '-'
    ax.plot([0, 1, 2], row[['R7', 'R8', 'R7p']].tolist(), color='dimgray', linestyle=l)
    ax.set_xticklabels(['', 'R7', '', '', '', 'R8', '', '', '', 'R7p'])
    
m = ['x', 'o', '+']
for i, rt in enumerate(['R7', 'R8', 'R7p']):
    data = rb[rt]
    ax.scatter([i]*len(data), data, marker=m[i], color=cm[rt], label=f'{rt}')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
ax.set_xlabel('Long photoreceptor subtype')
ax.set_ylabel("Rhabdomere volume (\u03BC" + "$m^3$)")

lines = [Line2D([0], [0], color='dimgray', linestyle='--'),
        Line2D([0], [0], color='dimgray', linestyle='-')]

point_leg = ax.legend()
line_leg = ax.legend(lines, ['DRA ommatidia', 'Non-DRA ommatidia'], loc=[0.015, 0.85]) 
ax.add_artist(point_leg)
plt.show()

# +
# Rhabdomere volume
data = rb.filter(items=['R7', 'R8', 'R7p'], axis=1)

fig, ax = plt.subplots(1, figsize=[10, 15])
for i, row in data.iterrows():
    if str(i) in ret_clust['dra']:
        continue
    elif row['R7'] < row['R8']*0.8:
        l = '-.'
    else:
        l = '-'
    ax.plot([0, 1, 2], row[['R7', 'R8', 'R7p']].tolist(), color='dimgray', linestyle=l)
    ax.set_xticklabels(['', 'R7', '', '', '', 'R8', '', '', '', 'R7p'])
    
m = ['x', 'o', '+']
for i, rt in enumerate(['R7', 'R8', 'R7p']):
    data = rb[rt]
    ax.scatter([i]*len(data), data, marker=m[i], color=cm[rt], label=f'{rt}')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
ax.set_xlabel('Long photoreceptor subtype')
ax.set_ylabel("Rhabdomere volume (\u03BC" + "$m^3$)")

lines = [Line2D([0], [0], color='dimgray', linestyle='--'),
        Line2D([0], [0], color='dimgray', linestyle='-')]

point_leg = ax.legend()
line_leg = ax.legend(lines, ['DRA ommatidia', 'Non-DRA ommatidia'], loc=[0.015, 0.85]) 
ax.add_artist(point_leg)
plt.show()

# +
# Fraction of rhabdom volume
data = rb_frac.filter(items=['R7', 'R8', 'R7p'], axis=1)

fig, ax = plt.subplots(1, figsize=[10, 15])
for i, row in data.iterrows():
    
    if str(i) in ret_clust['dra']:
        l = '--'
#     elif str(i) in ret_clust['vra']:
#         l = '-.'
    else:
        l = '-'
        
    ax.plot([0, 1, 2], row[['R7', 'R8', 'R7p']].tolist(), color='dimgray', linestyle=l)
    ax.set_xticklabels(['', 'R7', '', '', '', 'R8', '', '', '', 'R7p'])
    
    if row['R7'] < row['R8']:
        print(f"{i} R7frac: {row['R7']: .2f} R8frac: {row['R8']: .2f}")
    
m = ['x', 'o', '+']
for i, rt in enumerate(['R7', 'R8', 'R7p']):
    data = rb_frac[rt]
    ax.scatter([i]*len(data), data, marker=m[i], color=cm[rt], label=f'{rt}')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
ax.set_xlabel('Long photoreceptor subtype')
ax.set_ylabel("Fraction of rhabdom volume")

lines = [Line2D([0], [0], color='dimgray', linestyle='--'),
        Line2D([0], [0], color='dimgray', linestyle='-')]

point_leg = ax.legend()
line_leg = ax.legend(lines, ['DRA ommatidia', 'Non-DRA ommatidia'], loc=[0.015, 0.85]) 
ax.add_artist(point_leg)
    


# +
# Fraction of rhabdom volume
data = rb_frac.filter(items=['R7', 'R8', 'R7p'], axis=1)

fig, ax = plt.subplots(1, figsize=[10, 15])
for i, row in data.iterrows():
    
    if str(i) in ret_clust['v_trio']:
        l = '--'
    elif str(i) in ret_clust['vra']:
        l = '-'
    elif str(i) in ret_clust['dra']:
        continue
    else:
        l = '-.'
        
    ax.plot([0, 1, 2], row[['R7', 'R8', 'R7p']].tolist(), color='dimgray', linestyle=l)
    ax.set_xticklabels(['', 'R7', '', '', '', 'R8', '', '', '', 'R7p'])
    
    if row['R7'] < row['R8']:
        print(f"{i} R7frac: {row['R7']} R8frac: {row['R8']:.f02}")
    
m = ['x', 'o', '+']
for i, rt in enumerate(['R7', 'R8', 'R7p']):
    data = rb_frac[rt]
    ax.scatter([i]*len(data), data, marker=m[i], color=cm[rt], label=f'{rt}')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)   
ax.set_xlabel('Long photoreceptor subtype')
ax.set_ylabel("Fraction of rhabdom volume")

lines = [Line2D([0], [0], color='dimgray', linestyle='--'),
        Line2D([0], [0], color='dimgray', linestyle='-')]

point_leg = ax.legend()
line_leg = ax.legend(lines, ['DRA ommatidia', 'Non-DRA ommatidia'], loc=[0.015, 0.85]) 
ax.add_artist(point_leg)


# -



