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

# # Quantifying the variability of annotator reconstructions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression

# +
data_path = '~/Data/191219_exp2/200102_linkdf.pickle'
sig_thresh = 1.0

linkdf = pd.read_pickle(data_path)
data_long = pd.read_pickle('~/Data/191219_exp2/191204_lamina_link_df.pkl')

#display(data_long.loc[data_long['pre_om'] != data_long['post_om']])

subtypes = np.unique([*linkdf["pre_type"], *linkdf["post_type"]])
annotators = np.unique(linkdf["pre_om"])
ommatidia = np.unique(data_long['pre_om'])
all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  # all possible ('pre', 'post') connection types
assert((len(subtypes) ** 2) == len(all_ctypes))
ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]

# -

# cx counts from validation experiment
df_val = pd.DataFrame(index=annotators, columns=ctype_labels)
for annot, row in df_val.iterrows():
    for c in ctype_labels:
        pre_t, post_t = c.split('->')
        df_val.loc[annot, c] = sum((linkdf.pre_om == annot) & (linkdf.post_om == annot) & 
                                   (linkdf.pre_type == pre_t) & (linkdf.post_type == post_t))
# cx counts from lamina connectome
df_lamina = pd.DataFrame(index=ommatidia, columns=ctype_labels)
for om, row in df_lamina.iterrows():
    for c in ctype_labels:
        pre_t, post_t = c.split('->')
        # Cartridges on the posterior edge lack L4, so their counts for these connections are NaNed 
        if om in ['B0', 'E4', 'E5', 'E6', 'E7', 'D2'] and post_t == 'LMC_4':
            df_lamina.loc[om, c] = np.nan
        else:
            df_lamina.loc[om, c] = sum((data_long.pre_om == om) & (data_long.post_om == om) & 
                                       (data_long.pre_type == pre_t) & (data_long.post_type == post_t))
        

# +
# Define consensus connections (mean count in lamina connectome > thresh) 
df_val = df_val.loc[:, (df_lamina.mean() > sig_thresh)]
df_lamina = df_lamina.loc[:, (df_lamina.mean() > sig_thresh)]

print(df_val.columns)
print(df_lamina.columns)


# -

# Fano factor of each connection count = var/mean
def fano(df):
    return (df.std() ** 2)/df.mean()


# +
fig, ax = plt.subplots(1, figsize=[15, 15])
lamina_fano = fano(df_lamina).dropna().T
val_fano = fano(df_val).dropna().T

sns.distplot(lamina_fano, bins = np.arange(0, 40, 0.5),
             ax=ax, color='g', label=f'Lamina Data (n={len(ommatidia)})', kde=False)
sns.distplot(val_fano, bins = np.arange(0, 40, 0.5),
             ax=ax, color='r', label='C2 Validation Tracing (n=4)', kde=False)

ax.set_title("Fano factor of connection counts")
ax.set_xlabel("Fano factor")
ax.set_ylabel("Percentage of connection types")
ax.legend()

# +
lamina_var = df_lamina.std().dropna().T ** 2
val_var = df_val.std().dropna().T ** 2

fig, ax = plt.subplots(1, figsize=[15, 15])

sns.distplot(lamina_var, ax=ax, color='g', label=f'Lamina Data (n={len(ommatidia)})')
sns.distplot(val_var, ax=ax, color='r', label='C2 Validation Tracing (n=4)')

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
ax.plot(xticks, xticks*1.18, '--')

ax.legend()
ax.set_title('Variance of connection counts as a function of the mean')
ax.set_xlabel('Mean Count')
ax.set_ylabel('Variance')

# +
display(df_lamina.std().sort_values()**2)
display(df_val.std().sort_values()**2)


# -


