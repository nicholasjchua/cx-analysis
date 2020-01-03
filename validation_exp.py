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

# # The contribution of reconstruction error to the variability observed between lamina circuits
# - How do differences in skeletal reconstruction and synaptic partner identification contribute to variability of our connectivity data?
# - To control for stochastic/functional differences between lamina circuits, annotators independently reconstructed the same lamina cartridge (C2). These 'validation reconstructions' represent a state prior to a cartridge's 'review process', which is employed for each lamina cartridge presented in our connectome.  

# %load_ext autoreload
# %autoreload 0

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from src.fig_utils import hex_to_rgb
from sklearn.linear_model import LinearRegression

from IPython.display import display, Math, Latex
# -

print(hex_to_rgb("ffffff"))

# +
val_data_path = '~/Data/191219_exp2/200102_linkdf.pickle'

lamina_links = pd.read_pickle('~/Data/191219_exp2/191204_lamina_link_df.pkl')
val_links = pd.read_pickle(val_data_path)

subtypes = np.unique([*val_links["pre_type"], *val_links["post_type"]])
annotators = np.unique(val_links["pre_om"])
ommatidia = np.unique(lamina_links['pre_om'])

# all possible ('pre', 'post') permutations
all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  
all_ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]
# Colors to distinguish data from the connectome (grey) and data from the validation experiment (red)
cm = {'lam': "#383a3f", 
     'val': "#bf3b46"}

# -

# ### Preprocess connectivity data
# - ignores interommatidial connections
# - For posterior edges cartridges (that lack their own L4s), connections postsynaptic to L4 will be filled in with np.nan instead of zero: i.e. they are not included in mean/var calculations for that connection type. C2 is not a posterior edge cartridge

# +
# cx counts from lamina connectome
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

### A1 temporarily removed (it has no connections?)
df_lamina = df_lamina.drop(str('A1'), axis=0)

# cx counts from validation experiment
df_val = pd.DataFrame(index=annotators, columns=all_ctype_labels)
for annot, row in df_val.iterrows():
    for c in all_ctype_labels:
        pre_t, post_t = c.split('->')
        df_val.loc[annot, c] = sum((val_links.pre_om == annot) & (val_links.post_om == annot) & 
                                   (val_links.pre_type == pre_t) & (val_links.post_type == post_t))
# -

# ### Defining connection types that are consistently observed for all ommatidia in our lamina connectome
# Alternatives:
# 1. Average count above a certain threshold
# 2. Every observation exceeds a threshold 

# +
sig_thresh = 1.0

######### 1. Threshold: mean count must exceed threshold
# df_val = df_val.loc[:, (df_lamina.mean() > sig_thresh)]
# df_lamina = df_lamina.loc[:, (df_lamina.mean() > sig_thresh)]


######### 2. Threshold: all observations of that ctype must exceed thereshold
#display(df_lamina.filter(like='A1', axis='index'))

# need to change L4 nans so they won't be considered as < thresh when computing condition
df_lamina[df_lamina.isna()] = 1000.0 
condition = (df_lamina >= int(sig_thresh)).all()
df_lamina[df_lamina == 1000.0] = np.nan
df_val = df_val.loc[:, condition]
df_lamina = df_lamina.loc[:, condition]

sig_cts = df_val.columns
display(sig_cts)

assert(len(df_val.columns) == len(df_lamina.columns)) 
print(f"\n{len(sig_cts)} significant connection types (out of a total possible {len(all_ctypes)} pre/post permutations)")
# -

# ### Comparing the variance of different connection types

# +
lamina_var = df_lamina.var().sort_values()
display(lamina_var)

n_ex = 3  # show n ctypes with the largest and smallest variance in our connectome
low_cts = lamina_var.head(n_ex).index
high_cts = lamina_var.tail(n_ex).index

fig, axes = plt.subplots(n_ex, 1)
for i in range(0, n_ex):
    tmp_low = pd.DataFrame(data=[df_lamina[low_cts[i]], df_val[low_cts[i]]], columns=['Lamina connectome', 'Validation experiment'])
    sns.boxplot(x=tmp_low.columns, data=tmp_low,  
                palette={'Lamina connectome': cm['lam'], 'Validation experiment': cm['val']}, ax=axes[i])


# -

def fano(df):
    return (df.std() ** 2)/df.mean()
'''
def coef_var(df):
    return df.std()/df.mean()
'''

# ### Fano factor: a measure of dispersion for a set of observations
# - For a given connection type (e.g. LMC_2 -> LMC_1), we have a set of observed connection counts dispersed around a mean value
# - Because mean counts can differ significantly between types, we use Fano factor to describe a unitary deviation, scaled by the mean, that can be compared across connection types or aggregated to obtain an overall measure of dispersion in our data
#
# $$D = \frac{\sigma^2}{\mu}$$
#
# NOTE: Mean is not robust to outliers. Counts that are typically close to 0, but much higher because of a mistake in a particular observation, will have a very large fano factor. Ask mitya about using median for such a small n in our validation exp
#
# Outliers are examined during a step of peer-review. When a connection count is much larger than the average or if a previously unobserved connection type is present, it is almost always due to a mistake in the skeletal reconstruction. 

# +
fig, ax = plt.subplots(1, figsize=[15, 15])
lamina_fano = fano(df_lamina).dropna().T
val_fano = fano(df_val).dropna().T

max_fano = max([lamina_fano.max(), val_fano.max()])
interval = np.arange(0, max_fano + (10 - max_fano % 10), 0.5)  # round up to nearest 10

sns.distplot(lamina_fano, bins = interval,
             ax=ax, color=cm['lam'], label=f'Lamina Data (n={len(ommatidia)})', kde=False)
sns.distplot(val_fano, bins = interval,
             ax=ax, color=cm['val'], label='C2 Validation Tracing (n=4)', kde=False)

ax.set_title("Fano factor of connection counts")
ax.set_xlabel("Fano factor")
ax.set_ylabel("Percentage of connection types")
ax.legend()

# +
lamina_var = df_lamina.std().dropna().T ** 2
val_var = df_val.std().T ** 2

fig, ax = plt.subplots(1, figsize=[15, 15])

sns.distplot(lamina_var, ax=ax, color=cm['lam'], label=f'Lamina Data (n={len(ommatidia)})')
sns.distplot(val_var, ax=ax, color=cm['val'], label='C2 Validation Tracing (n=4)')

ax.set_title("Variability of connection counts")
ax.set_xlabel("Variance")
ax.set_ylabel("Percentage of connection types")
ax.legend()


# +
def lin_model_intercept0(x, y):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    return LinearRegression(fit_intercept=False).fit(x, y)

lamina_model = lin_model_intercept0(df_lamina.mean(), df_lamina.std()**2)
val_model = lin_model_intercept0(df_val.mean(), df_val.std()**2)

# +

fig, ax = plt.subplots(1, figsize=[15, 15])
xticks = np.arange(0, max((df_lamina.mean().max(), df_val.mean().max()))).reshape(-1, 1)

ax.set_xlim(0, max((df_lamina.mean().max(), df_val.mean().max())))
ax.set_ylim(0, max((df_lamina.std().max(), df_val.std().max())) ** 2)

ax.plot(xticks, lamina_model.predict(xticks), color='g', label="Lamina connectome")
ax.scatter(df_lamina.mean(), df_lamina.std()**2, color='g')
ax.plot(xticks, val_model.predict(xticks), color='r', label="Validation experiment")
ax.scatter(df_val.mean(), df_val.std()**2, color='r')
ax.plot(xticks, xticks*1.18, '--', label='')

ax.legend()
ax.set_title("Relationship of each connection type's mean and variance")
ax.set_xlabel('Mean Count')
ax.set_ylabel('Variance')
# -





