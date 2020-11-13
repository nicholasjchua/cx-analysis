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

# # Short Photoreceptor Diversity 
# Explore differences in the connectivity of short photoreceptor (R1-6) subtypes

# +
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import statsmodels.api as sm

from src.dataframe_tools import extract_connector_table
from src.cartridge_metadata import ret_clusters
from vis.fig_tools import subtype_cm

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
mpl.rc('font', size=14)
mpl.rc('figure', figsize=[12, 12])

# +
# synaptic links (contacts)
tp = '200914'
data_path = f'~/Data/{tp}_lamina/{tp}_linkdf.pickle'
df = pd.read_pickle(data_path)

# Rhabdom vols from Anastasia
rb = pd.read_csv('~/Data/lamina_additional_data/ret_cell_vol.csv').set_index('rtype').T
rb.index.name = 'om'
rb = rb.loc[sorted(rb.index), sorted(rb.columns)]
rb_frac = (rb.T/rb.sum(axis=1)).T.rename(mapper={'vol': 'fvol'}, axis=1)

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

ctx
# + {}
# Longform dataframes for seaborn functions
# Alternatives: number of contacts or number of pre terminals, rhabdomere vol or faction of rhabdom total
# Decide which is concatenated in combined

l_terms = pd.melt(terms.reset_index(), id_vars='om', value_vars=terms.columns, var_name='subtype', value_name='term_count')
l_rbfrac = pd.melt(rb_frac.reset_index(), id_vars='om', value_vars=rb_frac.columns,  var_name='subtype', value_name='fvol')
l_ctx = pd.melt(ctx.reset_index(), id_vars='om', value_vars=ctx.columns, var_name='subtype', value_name='syn_count')
l_rb = pd.melt(rb.reset_index(), id_vars='om', value_vars=rb.columns,  var_name='subtype', value_name='vol')

# combined has fields for 'vol': raw volume, 'fvol': fractional volume, 'syn_count': connection count, 'term_count': terminal count 
combined = pd.concat([l_rb, l_rbfrac['fvol'], l_ctx['syn_count'], l_terms['term_count']], axis=1)
combined


# +
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]
multi = (data['syn_count']/data['term_count']).mean()
multi_sd = (data['syn_count']/data['term_count']).std(ddof=0)

display(len(data))
display(multi)
display(multi_sd)
# -

# ## Differences between short photoreceptor subtypes 

# Options for the type of volume/connectivity vars to plot and fit
x = 'fvol'  # or 'vol'
y = 'syn_count'  # or 'term_count'

# +

#combined = pd.concat([l_rb, l_ctx['syn_count']], axis=1)
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]
xmax = data[x].max()
ymax = data[y].max()

g = sns.JointGrid(x=x, y=y, data=data, height=8,
                  xlim=[0, xmax + (xmax*0.1)], ylim=[0, ymax + (ymax*0.1)])


spr_pairs = (('R1', 'R4'), ('R2' , 'R5'), ('R3', 'R6'))
pt = ('o', '+', '^')

for i, p in enumerate(spr_pairs):
    rows = (data['subtype'] == p[0]) | (data['subtype'] == p[1])
    
    c = cm[p[0]+p[1]]
    
    g.ax_joint.scatter(x=data.loc[rows, x], y=data.loc[rows, y], label=f'{p[0]}/{p[1]}', marker=pt[i], color=c, s=40)
    g.ax_joint.legend(loc='upper left')
    
    sns.kdeplot(data.loc[rows, x], legend=False, ax=g.ax_marg_x, color=c)
    sns.kdeplot(data.loc[rows, y], legend=False, ax=g.ax_marg_y, vertical=True, color=c)
if x == 'fvol':
    g.ax_joint.set_xlabel("Fraction of total rhabdom volume")
else:
    g.ax_joint.set_xlabel('Rhabdomere volume')
        
if y == 'syn_count':
    g.ax_joint.set_ylabel("Number of synaptic connections (outputs)")
else:
    g.ax_joint.set_ylabel('Number of presynaptic terminals')
    

#g.ax_joint.set_xlabel("Rhabdomere volume (\u03BC" + "$m^3$)")
#g.savefig("/mnt/home/nchua/Dropbox/200609_pr-v-cx.svg")
# -
# ## Correlation between rhabdomere volume and number of presynaptic terminals
# - [statsmodel OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
# - [why R-squared is so high when no intercept is fitted?](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-why-are-r2-and-f-so-large-for-models-without-a-constant/)


# +
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]

X = data[x] 
X = sm.add_constant(X)
Y = data[y]

model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]

X = data['fvol']
X = sm.add_constant(X)
Y = data['term_count']

model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]

X = data['vol']
X = sm.add_constant(X)
Y = data['syn_count']

model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
data = combined.loc[[i for i, row in combined.iterrows() if row['subtype'] not in ['R7', 'R7p', 'R8']]]

X = data['vol']
X = sm.add_constant(X)
Y = data['term_count']

model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())
# -

#

# ## Hypothesis test: synaptic outputs from R2 and R5 (central short PRs) and R1, R3, R4, and R6 (peripheral short PRs) 

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

# ## Hypothesis test: rhabdomere volume of R2 and R5 (central short PRs) and R1, R3, R4, and R6 (peripheral short PRs) 

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





