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

# # R1-6 output characteristics
#
#

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import itertools
#from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from vis.hex_lattice import hexplot
from vis.colour_palettes import subtype_cm
from vis.fig_tools import linear_cmap

import matplotlib as mpl
#mpl.rc('font', size=14)

# +
tp = '200914'
lamina_links = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')

subtypes = np.unique([*lamina_links["pre_type"], *lamina_links["post_type"]])
all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  
all_ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]
om_list = ommatidia = np.unique(lamina_links['pre_om'])

df_lamina = pd.DataFrame(index=om_list, columns=all_ctype_labels)
for om, row in df_lamina.iterrows():
    for c in all_ctype_labels:
        pre_t, post_t = c.split('->')
        # Cartridges on the posterior edge lack L4, so their counts for these connections are NaNed 
        if om in ['B0', 'E4', 'E5', 'E6', 'E7', 'D2'] and post_t == 'LMC_4':
            df_lamina.loc[om, c] = np.nan
        else:
            df_lamina.loc[om, c] = sum((lamina_links.pre_om == om) & (lamina_links.post_om == om) & 
                                       (lamina_links.pre_type == pre_t) & (lamina_links.post_type == post_t))


# +
svfs = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
all_vfs = [*svfs, 'R7', 'R8', 'R7p']
lmcs = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4', 'LMC_N']
#df_presites = pd.DataFrame(index=om_list, columns=all_vfs)
#df_outputs = pd.DataFrame(index=om_list, columns=all_vfs)

# long form dataframe with multiindex (om x svf subtype) might be easier to use
long = pd.DataFrame(index=pd.MultiIndex.from_product([om_list, all_vfs], names=['om', 'subtype']))

# Uses neuron name to query list of connections since pretype combines svfs pairs
for om in om_list:
    for s in all_vfs:
        neuron_name = f"om{om}_{s}"
        if om == 'C2':
            neuron_name += '_nc'
            
        links = lamina_links.loc[lamina_links['pre_neuron'] == neuron_name]
        lmc_links = links.loc[[i for i, v in links['post_type'].items() if v[0:3] == 'LMC']]
        c_ids = links['cx_id'].unique()
#         df_presites.loc[om, s] = len(c_ids)
#         df_outputs.loc[om, s] = len(links)
        
        long.loc[(om, s), 'presites'] = len(c_ids)
        long.loc[(om, s), 'syn'] = len(links)
        long.loc[(om, s), 'lmc_syn'] = len(lmc_links)
        for l in lmcs:
            long.loc[(om, s), f'->{l}'] = len(links.loc[[i for i, v in links['post_type'].items() if v == l]])
        if len(links) > 0 and s in svfs:
            long.loc[(om, s), 'syn_multi'] = len(links)/len(c_ids)
        elif len(links) > 0 and s not in svfs:
            long.loc[(om, s), 'syn_multi'] = len(links)/len(c_ids)
            display('Long photoreceptor has more than zero connections in the lamina. Check if this is real')
            display(links)
        else:
            # When links are 0 (lvfs), multiplicity is NaN
            long.loc[(om, s), 'syn_multi'] = np.nan


# +
# Volume data from excel provided by AM. 
# Contains both the volume of each PR's cell body, and the volume of just their rhabdomeres
xl_dir = '~/Data/cell_rbd_volume.xlsx'
full_df = pd.read_excel(xl_dir, header=[0, 1], index_col=0, nrows=9)

# Each om has two columns: cell body and rhabdomere vol 
rbd = dict.fromkeys(full_df.columns.levels[0])
soma = dict.fromkeys(full_df.columns.levels[0])

for om in full_df.columns.levels[0]:
    rbd[om] = full_df.loc[:, (om, 'rbd')]
    soma[om] = full_df.loc[:, (om, 'cell body')]
    
vol_df = pd.DataFrame(rbd)
# E5 was damaged in our specimen so Anastasia took measurements from an E5 in a different wasp
vol_df = vol_df.rename(mapper={"R7'": "R7p"}, axis=0).rename(mapper={"E5*": "E5"}, axis=1).T
soma_df = pd.DataFrame(soma)
soma_df = soma_df.rename(mapper={"R7'": "R7p"}, axis=0).rename(mapper={"E5*": "E5"}, axis=1).T
# append this to the long form dataframe
for om, row in vol_df.iterrows():
    total = row.sum()
    for s, v in row.items():
        long.loc[(om, s), 'rbd_vol'] = v
        long.loc[(om, s), 'rbd_frac'] = v/total
        long.loc[(om, s), 'soma_vol'] = soma_df.loc[om, s]
# -

# ## Summary of PR subtypes

display('Average across ommatidia', long.groupby('subtype').mean().round(decimals=1))
display('Standard deviation', long.groupby('subtype').std().round(decimals=1))

# ## Compare (R2, R5) and (R1, R3, R4, R6)
# - medial pair/lateral pairs (problem: for retinotopic space, medial means center of the face)
# - major/minor?

# +
r1r4r3r6_ind = (slice(None),['R1', 'R3', 'R4', 'R6'])
r2r5_ind = (slice(None),['R2', 'R5'])


svf_comp =  pd.DataFrame(index=pd.MultiIndex.from_product([['R2, R5', 'R1, R3, R4, R6'], ['Mean', 'SD']],  
                                                          names=['class', 'statistic']), columns=long.columns)
svf_comp.loc[('R2, R5', 'Mean'), :] =  long.loc[r2r5_ind, :].mean()
svf_comp.loc[('R2, R5', 'SD'), :] =  long.loc[r2r5_ind, :].std(ddof=0)
svf_comp.loc[('R1, R3, R4, R6', 'Mean'), :] =  long.loc[r1r4r3r6_ind, :].mean()
svf_comp.loc[('R1, R3, R4, R6', 'SD'), :] =  long.loc[r1r4r3r6_ind, :].std(ddof=0)

display(svf_comp.round(decimals=1))

# display(long.loc[r1r4r3r6_ind, :].mean())
# display(long.loc[r2r5_ind, :].mean())
# display(long.loc[r1r4r3r6_ind, :].std())
# display(long.loc[r2r5_ind, :].std())
# display(long.loc[r1r4r3r6_ind, :].mean() / long.loc[r2r5_ind, :].mean())
# display(long.loc[r2r5_ind, :].mean() / long.loc[r1r4r3r6_ind, :].mean())
# -

# ## Look at the ratio of these measures, averaged across ommatidia

# +
om_ratio = dict()
for om, rows in long.groupby('om'):
    om_ratio.update({om: 1-(rows.loc[r1r4r3r6_ind, :].mean()/rows.loc[r2r5_ind, :].mean())})
    
ratio_df = pd.DataFrame(om_ratio).T
display(ratio_df.mean())
display(ratio_df.std())
# -

# ## Does rhabdomere volume correlate with:
# - Number of photoreceptor terminals
# - Number of outputs synapses 
# - Number of Output synapses to LMCs 
# - Average synapse multiplicity?

# +
x_vars = ['rbd_vol', 'rbd_frac', 'soma_vol']
y_vars = ['presites', 'syn', 'lmc_syn', 'syn_multi']

g = sns.pairplot(long, x_vars=x_vars, y_vars=y_vars, kind='reg')
res_table = pd.DataFrame(index=pd.MultiIndex.from_product([x_vars, y_vars]), columns=['R-squared', 'p-value'])

for this_x, this_y in itertools.product(x_vars, y_vars):
    X = sm.add_constant(long[this_x])
    Y = long[this_y]

    model = sm.OLS(Y, X)
    results = model.fit()
    res_table.loc[(this_x, this_y), 'R-squared'] = results.rsquared
    res_table.loc[(this_x, this_y), 'p-value'] = results.f_pvalue
#     display((X, Y))
#     display(f"R-squared: {results.rsquared}")
#     display(f"p-value (t-stat): {results.pvalues}")
display(res_table)

# +
x_vars = ['rbd_vol', 'rbd_frac', 'soma_vol']
y_vars = ['->LMC_1', '->LMC_2', '->LMC_3', '->LMC_4']

g = sns.pairplot(long, x_vars=x_vars, y_vars=y_vars, kind='reg')
res_table = pd.DataFrame(index=pd.MultiIndex.from_product([x_vars, y_vars]), columns=['R-squared', 'p-value'])

for this_x, this_y in itertools.product(x_vars, y_vars):
    X = sm.add_constant(long[this_x])
    Y = long[this_y]

    model = sm.OLS(Y, X)
    results = model.fit()
    res_table.loc[(this_x, this_y), 'R-squared'] = results.rsquared
    res_table.loc[(this_x, this_y), 'p-value'] = results.f_pvalue
#     display((X, Y))
#     display(f"R-squared: {results.rsquared}")
#     display(f"p-value (t-stat): {results.pvalues}")
display(res_table)
# -

# ## Are these correlations present *within* SVF classes?  

# ### R1, R4, R3, and R6

# +
x_vars = ['rbd_vol', 'rbd_frac', 'soma_vol']
y_vars = ['presites', 'syn', 'lmc_syn', 'syn_multi']

data = long[long.index.get_level_values(1).isin(['R1', 'R3', 'R4', 'R6'])]
g = sns.pairplot(data, x_vars=x_vars, y_vars=y_vars, kind='reg')
res_table = pd.DataFrame(index=pd.MultiIndex.from_product([x_vars, y_vars]), columns=['R-squared', 'p-value'])

for this_x, this_y in itertools.product(x_vars, y_vars):
    X = sm.add_constant(data[this_x])
    Y = data[this_y]

    model = sm.OLS(Y, X)
    results = model.fit()
    res_table.loc[(this_x, this_y), 'R-squared'] = results.rsquared
    res_table.loc[(this_x, this_y), 'p-value'] = results.f_pvalue
#     display((X, Y))
#     display(f"R-squared: {results.rsquared}")
#     display(f"p-value (t-stat): {results.pvalues}")
display(res_table)
# -

# ### R2 and R5

# +
x_vars = ['rbd_vol', 'rbd_frac', 'soma_vol']
y_vars = ['presites', 'syn', 'lmc_syn', 'syn_multi']

data = long[long.index.get_level_values(1).isin(['R2', 'R5'])]
g = sns.pairplot(data, x_vars=x_vars, y_vars=y_vars, kind='reg')
res_table = pd.DataFrame(index=pd.MultiIndex.from_product([x_vars, y_vars]), columns=['R-squared', 'p-value'])

for this_x, this_y in itertools.product(x_vars, y_vars):
    X = sm.add_constant(data[this_x])
    Y = data[this_y]

    model = sm.OLS(Y, X)
    results = model.fit()
    res_table.loc[(this_x, this_y), 'R-squared'] = results.rsquared
    res_table.loc[(this_x, this_y), 'p-value'] = results.f_pvalue
#     display((X, Y))
#     display(f"R-squared: {results.rsquared}")
#     display(f"p-value (t-stat): {results.pvalues}")
display(res_table)
# -

sns.pairplot(long, x_vars=['rbd_vol', 'rbd_frac', 'soma_vol'], y_vars=['->LMC_1', '->LMC_2', '->LMC_3', '->LMC_4'])



# +
X = long['rbd_vol']
Y = long['presites']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
X = long['rbd_vol']
Y = long['syn']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
X = long['rdb_vol']
Y = long['lmc_syn']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
X = long['rbd_vol']
Y = long['syn_multi']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())
# -

# - Rhabdomere volume (both raw and fractional) is positively correlated with both number of SVF output terminals and synapses. Number of terminals is a slightly better predictor or rhabdomere volume, but the difference in R-squared is small. 
# - SVF synapse multiplicity (number of synapses per terminal) is relatively constant; i.e. the number of SVF output synapses scales predictably with number of terminals. Because of this, neither has a stronger correlation with rhabdomere volume 

# ## Does SVF volume correlate with number of outputs to the LMC subtypes?

# +
X = long['rbd_vol']
Y = long['->LMC_1']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
X = long['rbd_vol']
Y = long['->LMC_2']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())
# -



# +
X = long['rbd_vol']
Y = long['->LMC_3']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
X = long['rbd_vol']
Y = long['->LMC_4']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
X = long['rbd_vol']
Y = long['->LMC_N']
sns.jointplot(X, Y)

# OLS
X = sm.add_constant(X)
model = sm.OLS(Y, X)
results = model.fit()

display(results.summary())

# +
# av_multi = df_outputs/df_presites
# r1r4_m = [*av_multi['R1'].to_list(), *av_multi['R4'].to_list()]
# r2r5_m = [*av_multi['R2'].to_list(), *av_multi['R5'].to_list()]
# r3r6_m = [*av_multi['R3'].to_list(), *av_multi['R6'].to_list()]

# r1r4_ntb = [*df_presites['R1'].to_list(), *df_presites['R4'].to_list()]
# r2r5_ntb = [*df_presites['R2'].to_list(), *df_presites['R5'].to_list()]
# r3r6_ntb = [*df_presites['R3'].to_list(), *df_presites['R6'].to_list()]

# +
max_sites = long['presites'].max().max()

sns.jointplot(df_presites['R1'], df_presites['R4'], xlim=(0, max_sites), ylim=(0, max_sites))

sns.jointplot(df_presites['R2'], df_presites['R5'], xlim=(0, max_sites), ylim=(0, max_sites))

sns.jointplot(df_presites['R3'], df_presites['R6'], xlim=(0, max_sites), ylim=(0, max_sites))

# -

# ### R1R4 vs R3R6

# +
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[15, 15])
ax = axes.flatten()
plot_lin_fit(df_lamina['R1R4->LMC_1'],df_lamina['R3R6->LMC_1'], ax[0], plot_kwargs={'color': 'r'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[0].set_title('Number of inputs to L1')
ax[0].set_xlabel('R1R4')
ax[0].set_ylabel('R3R6')
plot_lin_fit(df_lamina['R1R4->LMC_2'],df_lamina['R3R6->LMC_2'], ax[1], plot_kwargs={'color': 'b'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[1].set_title('Number of inputs to L2')
ax[1].set_xlabel('R1R4')
ax[1].set_ylabel('R3R6')
plot_lin_fit(df_lamina['R1R4->LMC_3'],df_lamina['R3R6->LMC_3'], ax[2], plot_kwargs={'color': 'g'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[2].set_title('Number of inputs to L3')
ax[2].set_xlabel('R1R4')
ax[2].set_ylabel('R3R6')
plot_lin_fit(df_lamina['R1R4->LMC_4'].dropna(),df_lamina['R3R6->LMC_4'].dropna(), ax[3], plot_kwargs={'color': 'y'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[3].set_title('Number of inputs to L4')
ax[3].set_xlabel('R1R4')
ax[3].set_ylabel('R3R6')

#plot_lin_fit(df_lamina['R1R4->LMC_4'],df_lamina['R3R6->LMC_4'], ax, plot_kwargs={'color': 'r'}, scatter_kwargs={'color': 'r'})
plt.show()
# + {}
# fig, axes = plt.subplots(3, 1, figsize=[20, 30])
# ax = axes.flatten()
# cm = get_cmap('coolwarm')

# frac = (df_lamina.loc[:, 'R1R4->LMC_1']/df_lamina.loc[:, 'R3R6->LMC_1']) - 1.0
# nd = {om: {'colour': cm(frac[om])} for om in ommatidia}
# hexplot(node_data=nd, ax=ax[0])

# frac = (df_lamina.loc[:, 'R1R4->LMC_2']/df_lamina.loc[:, 'R3R6->LMC_2']) - 1.0
# nd = {om: {'colour': cm(frac[om])} for om in ommatidia}
# hexplot(node_data=nd, ax=ax[1])

# frac = (df_lamina.loc[:, 'R1R4->LMC_3']/df_lamina.loc[:, 'R3R6->LMC_3']) - 1.0
# nd = {om: {'colour': cm(frac[om])} for om in ommatidia}
# hexplot(node_data=nd, ax=ax[2])
# -

# ### R2R5 vs (R1R4 + R3R6)

# +
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[15, 15])
ax = axes.flatten()
plot_lin_fit(2 * df_lamina['R2R5->LMC_1'], df_lamina['R1R4->LMC_1'] + df_lamina['R3R6->LMC_1'], 
             ax[0], plot_kwargs={'color': 'r'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[0].set_title('Number of inputs to L1')
ax[0].set_xlabel('R2R5')
ax[0].set_ylabel('R1R4 + R3R6')
plot_lin_fit(2 * df_lamina['R2R5->LMC_2'], df_lamina['R1R4->LMC_2'] + df_lamina['R3R6->LMC_2'], 
             ax[1], plot_kwargs={'color': 'b'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[1].set_title('Number of inputs to L2')
ax[1].set_xlabel('R2R5')
ax[1].set_ylabel('R1R4 + R3R6')
plot_lin_fit(2 * df_lamina['R2R5->LMC_3'], df_lamina['R1R4->LMC_3'] + df_lamina['R3R6->LMC_3'], 
             ax[2], plot_kwargs={'color': 'g'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[2].set_title('Number of inputs to L3')
ax[2].set_xlabel('R2R5')
ax[2].set_ylabel('R1R4 + R3R6')
plot_lin_fit(2 * df_lamina['R2R5->LMC_4'].dropna(), df_lamina['R1R4->LMC_4'].dropna() + df_lamina['R3R6->LMC_4'].dropna(), 
             ax[3], plot_kwargs={'color': 'y'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[3].set_title('Number of inputs to L4')
ax[3].set_xlabel('R2R5')
ax[3].set_ylabel('R1R4 + R3R6')

plt.show()
# -

fig, ax = plt.subplots(1)
plot_lin_fit(df_lamina['centri->LMC_1'],df_lamina['R3R6->LMC_1'], ax)
plt.show()

# +
fig, ax = plt.subplots(1)
plot_lin_fit(df_lamina['centri->LMC_3'], df_lamina['R1R4->LMC_3'] + df_lamina['R3R6->LMC_3'], ax)
plt.show()

frac = (df_lamina['centri->LMC_3']/ df_lamina['R1R4->LMC_3']).sort_values()
display(frac)
# -

display(df_lamina['centri->LMC_3'])
display(df_lamina['centri->LMC_1'])



# Consistent vs. Inconsistent 
# - Frequency at which the connection is observed
#
# High vs. Low
# - Mean contact count 

# +
def n_morethan_0(df):
    n_observed = []
    for c in df.columns:
        n_observed.append((df[c] > 0.0).sum())
    return pd.Series(n_observed, index=df.columns)

n_ob = n_morethan_0(df_lamina)
p_ob = n_ob / n_ob.max()
fig, ax = plt.subplots(1)
ax.set_xlabel("% ommatidia connection observed > 0")
ax.set_ylabel("Number of connection types")

ax = p_ob.plot.hist()
#ax = sns.distplot(p_ob, bins=50, kde=False)
#ax = sns.distplot(p_ob, kde=False)
plt.show()
# -

p_ob = n_ob / n_ob.max()
ob_df = pd.DataFrame(data={'mean': df_lamina.mean(),
                          'percent_om_obs': p_ob}, index=df_lamina.columns)

print(f"Connections that are observed less than 100% but have contact count mean more than 1.0")
x = ob_df.loc[(df_lamina.mean() > 1.0) & (ob_df['percent_om_obs'] < 1.0)].sort_values(by='mean', ascending=False)
display(x)
#len(df_lamina.columns[(df_lamina.mean() > 1.0) & (ob_df['percent_om_obs'] < 1.0)])

dat = df_lamina['R2R5->LMC_4'].plot.hist(bins=50)

fig, ax = plt.subplots(len(x.index), 1)
for i, c in enumerate(x.index):
    print(c)
    ax[i].hist(df_lamina[c])
plt.show()

print(f"Connections that are observed consistently (100%) but have contact count mean less than 1.0")
display(df_lamina.mean()[(df_lamina.mean() < 1.0) & (ob_df['percent_om_obs'] == 1.0)])
len(df_lamina.columns[(df_lamina.mean() < 1.0) & (ob_df['percent_om_obs'] == 1.0)])

# +
# Interested in connections with mean close to 0, bin all high counts together
upper_bin = 10.0
x = [upper_bin if mn >= upper_bin else mn for mn in df_lamina.mean()]

fig, ax = plt.subplots(1, figsize=[15, 10])
ax.set_title("Mean number of contacts for each possible connection type\n" + 
             f"Total subtype permutations: {len(df_lamina.columns)}\n" + 
            f"Mean >= {upper_bin} combined in the last bin")

ax.set_xlabel("Mean number of contacts")
ax.set_ylabel("Number of connection types")

ax.hist(x, bins=100)
plt.show()
#display(x)

# +
# Interested in connections with mean close to 0, bin all high counts together
upper_bin = 10.0
x = [upper_bin if mn >= upper_bin else mn for mn in df_lamina.median()]

fig, ax = plt.subplots(1, figsize=[15, 10])
ax.set_title("MEDIAN number of contacts for each possible connection type\n" + 
             f"Total subtype permutations: {len(df_lamina.columns)}\n" + 
            f"MEDIAN >= {upper_bin} combined in the last bin")

ax.set_xlabel("MEDIAN number of contacts")
ax.set_ylabel("Number of connection types")

ax.hist(x, bins=100)
plt.show()
#display(x)

# +
upper_bin = 100
x = df_lamina.values.flatten()
x = x[~df_lamina.isna().values.flatten()]  # Discard L4 NaNs 

x = [int(upper_bin) if ob >= upper_bin else int(ob) for ob in x]

fig, ax = plt.subplots(1, figsize=[15, 10])
ax.set_title("All observed contact counts between every neuron\n" + 
             f"N obervations = {len(x)}\n" + 
            f"Counts >= {upper_bin} combined in the last bin")

ax.set_xlabel("Number of contacts")
ax.set_ylabel("Number of observations")

ax.hist(x, bins=100)
plt.show()
#display(x)

# +
low_means = df_lamina.mean()[df_lamina.mean() < 1.0]

display(df_lamina.mean()[df_lamina.mean() < 1.0].sort_values(ascending=False))
# -



