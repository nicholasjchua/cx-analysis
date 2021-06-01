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

# # Errors and Outliers
# We screen our connectome for the following: 
# 1. Number of inputs from homologous pairs of short photoreceptors is relatively uniform (e.g. R1->L2 ~= R4->L2)
# 2. Number of inputs from R1R4 is typically very close to that of R3R6 (e.g. R1R4->L1 ~= R3R6->L1)
# 3. 
# - Identifying discrepencies between cartridges that could be the result of reconstruction error

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression

from vis.hex_lattice import hexplot
from vis.colour_palettes import subtype_cm
from vis.fig_tools import linear_cmap


import matplotlib as mpl
mpl.rc('font', size=14)

# +
lamina_links = pd.read_pickle('~/Data/200128_lamina/200128_linkdf.pickle')
subtypes = np.unique([*lamina_links["pre_type"], *lamina_links["post_type"]])

all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  
all_ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]
ommatidia = ommatidia = np.unique(lamina_links['pre_om'])

df_lamina = pd.DataFrame(index=ommatidia, columns=all_ctype_labels)
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
def lin_model_intercept0(x, y):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    return LinearRegression(fit_intercept=False).fit(x, y)

def plot_lin_fit(x, y, ax, plot_kwargs={}, scatter_kwargs={}):
    xpoints = np.arange(0, max(x.max(), y.max())).reshape(-1, 1)
    model = lin_model_intercept0(x, y)
    fitprops = f"R^2: {model.score(x.to_numpy().reshape(-1, 1), y): .2f}, coef: {model.coef_[0]: .2f}"

    ax.plot(xpoints, model.predict(xpoints), label=plot_kwargs.get('label', '') + ' \n' + fitprops, **plot_kwargs)
    ax.scatter(x,y, label=None, **scatter_kwargs)
    ax.legend()
    


# -

svfs = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6']
df_presites = pd.DataFrame(index=ommatidia, columns=svfs)
df_outputs = pd.DataFrame(index=ommatidia, columns=svfs)
for om in ommatidia:
    for s in svfs:
        neuron_name = f"om{om}_{s}"
        links = lamina_links.loc[lamina_links['pre_neuron'] == neuron_name]
        c_ids = links['cx_id'].unique()
        df_presites.loc[om, s] = len(c_ids)
        df_outputs.loc[om, s] = len(links)


fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=[40, 20])
plot_lin_fit(df_presites['R1'], df_presites['R4'], ax=ax[0])
ax[0].set_title('Number of presynaptic terminals, R1 vs R4')
ax[0].set_xlabel('R1')
ax[0].set_ylabel('R4')
plot_lin_fit(df_presites['R2'], df_presites['R5'], ax=ax[1])
ax[1].set_title('Number of presynaptic terminals, R2 vs R5')
ax[1].set_xlabel('R2')
ax[1].set_ylabel('R5')
plot_lin_fit(df_presites['R3'], df_presites['R6'], ax=ax[2])
ax[2].set_title('Number of presynaptic terminals, R3 vs R6')
ax[2].set_xlabel('R3')
ax[2].set_ylabel('R6')


# +
max_sites = df_presites.max().max()

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
# -



# +

sns.jointplot(x='R1R4->LMC_1',y='R3R6->LMC_1', data=df_lamina)
# -

sns.jointplot(df_lamina['R1R4->LMC_2'],df_lamina['R3R6->LMC_2'], color='k')

sns.jointplot(df_lamina['R1R4->LMC_3'],df_lamina['R3R6->LMC_3'], color='k')

# +
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
plot_lin_fit(df_lamina['R2R5->LMC_1'], df_lamina['R1R4->LMC_1'] + df_lamina['R3R6->LMC_1'], 
             ax[0], plot_kwargs={'color': 'r'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[0].set_title('Number of inputs to L1')
ax[0].set_xlabel('R2R5')
ax[0].set_ylabel('R1R4 + R3R6')
plot_lin_fit(df_lamina['R2R5->LMC_2'], df_lamina['R1R4->LMC_2'] + df_lamina['R3R6->LMC_2'], 
             ax[1], plot_kwargs={'color': 'b'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[1].set_title('Number of inputs to L2')
ax[1].set_xlabel('R2R5')
ax[1].set_ylabel('R1R4 + R3R6')
plot_lin_fit(df_lamina['R2R5->LMC_3'], df_lamina['R1R4->LMC_3'] + df_lamina['R3R6->LMC_3'], 
             ax[2], plot_kwargs={'color': 'g'}, scatter_kwargs={'c': 'k', 'marker':'x'})
ax[2].set_title('Number of inputs to L3')
ax[2].set_xlabel('R2R5')
ax[2].set_ylabel('R1R4 + R3R6')
plot_lin_fit(df_lamina['R2R5->LMC_4'].dropna(), df_lamina['R1R4->LMC_4'].dropna() + df_lamina['R3R6->LMC_4'].dropna(), 
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

# ## Source of methodological error
#
# ### Connectome reconstruction
# 1. Labelling of synaptic contacts  
#     i. Identification of presynaptic terminals   
#     ii. Identification of postsynaptic partners   
# 2. Skeletal morphology:  
#    i. Misattribution error  
#    ii. Arbor fragments (failure to attribute an arbor to an identified neuron) 
# 3. Misc.  
#    i. Subtype categorisation/ambiguity  
#    ii. Inhomogeneity of image dataset  

# +
unknown_partners = dict.fromkeys(ommatidia)
total_contacts = dict.fromkeys(ommatidia)
total_tbars = dict.fromkeys(ommatidia)
for om, rows in lamina_links.groupby('pre_om'):

    unknown_partners[om] = (rows['post_type'] == 'UNKNOWN').sum()
    total_contacts[om] = len(rows)
    total_tbars[om] = len(rows['cx_id'].unique())
    
print({unknown_partners})

# -


