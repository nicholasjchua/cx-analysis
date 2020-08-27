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

# # Lamina Monopolar Cell Connectivity Patterns
# Exploratory data analysis: what variables seperate the different classes of LMCs we identified?

# +
import numpy as np
import pandas as pd
from typing import Tuple, Union, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
import itertools

from vis.colour_palettes import subtype_cm

# +
tp = '200218'
linkdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')
cxdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_cxdf.pickle')

subtypes = np.unique([*linkdf["pre_type"], *linkdf["post_type"]])
lmcs = [s for s in subtypes if s[0] is 'L']

all_ctypes = [p for p in itertools.product(subtypes, subtypes)]  
all_ctype_labels = [f"{pre}->{post}" for pre, post in all_ctypes]
ommatidia = np.unique(linkdf['pre_om'])

# df of interommatidial connections
criteria = ((linkdf['pre_om'] != linkdf['post_om']) & (linkdf['post_om'] != 'UNKNOWN'))
interdf = linkdf.loc[criteria]
# lists of ommatidia with or without l4s
with_l4 = np.unique([row['post_om'] for i, row in linkdf.iterrows() if row['post_type'] == 'LMC_4'])
no_l4 = [om for om in ommatidia if om not in with_l4]

# +
### Assemble dataframe ###
# LMC outputs
out_counts = []
for om, r in cxdf.groupby('om'):
    for pre, rr in r.groupby('pre_type'):
        if (om in no_l4) and (pre == 'LMC_4'):
            continue
        else:
            out_counts.append({'om': om, 
                               'type': pre, 
                               'output_count': rr['n_connect'].sum()})
# data contains cols for number of overall outputs and number of inputs from each subtype
data = pd.DataFrame(out_counts)
for s in subtypes:
    data[s] = np.zeros(len(data.index))
    
inter_in = np.zeros(len(data.index), dtype=int)
for i, row in data.iterrows():
    # Home inputs
    criteria = (cxdf['om'] == row['om']) & (cxdf['post_type'] == row['type'])
    inputs = cxdf.loc[criteria, ('pre_type', 'n_connect')].set_index('pre_type').to_dict()
    data.loc[i, 3:-1] = pd.Series(inputs['n_connect'])
    # Inputs to the lmcs from outside home
    criteria = ((interdf['post_om'] == row['om']) & (interdf['post_type'] == row['type']))
    inter_in[i] = len(interdf.loc[criteria])
        
data['inter_in'] = inter_in
#display(data.loc[data["type"] == 'LMC_4'])



# +
fig = plt.figure(figsize=[10, 10])
ax = fig.gca(projection='3d')
ax.set_xlabel('Presynaptic contacts')
ax.set_zlabel('Inputs from R1R4 + R3R6')
ax.set_ylabel('Inputs from neighboring cartridges')

c = subtype_cm()

m = {'LMC_N': 'D', 'LMC_1': 's', 'LMC_2': '^', 'LMC_3': 'o', 'LMC_4': 'P'}

for pre, rows in data.groupby('type'):
    if pre not in lmcs:
        continue
    else:
        
        infrac = rows['R1R4'] + rows['R3R6']
        ax.scatter(rows['output_count'], rows['inter_in'], infrac, 
                   label=f"{pre}, n = {len(rows)}", marker=m[pre], s=50, 
                   c=c[pre], alpha=0.5, depthshade=False)
        #ax.scatter(rows['R1R4'] + rows['R3R6'], rows['inter_in'], rows['output_count'], label=pre)
ax.legend()

ax.invert_yaxis()
ax.view_init(elev=5)
plt.show()

# +
fig, ax = plt.subplots(1, figsize=[15, 10])
ax.set_title('LMC ')
ax.set_xlabel('Inputs from R1R4 + R3R6')
ax.set_ylabel('Number of cartridges')

for pre in lmcs:
    x = data.loc[data['type'] == pre, ('R1R4', 'R3R6')].sum(axis=1)
    sns.distplot(x, ax=ax, color=c[pre])
# -

fig, ax = plt.subplots(1, figsize=[10, 8])
ax.set_title('L1 and L3')
ax.set_xlabel('Inputs from R1R4 + R3R6')
ax.set_ylabel('Number of cartridges')
for pre in ['LMC_1', 'LMC_3']:
    
    x = data.loc[data['type'] == pre, ('R1R4', 'R3R6')].sum(axis=1)
    sns.distplot(x, ax=ax, color=c[pre], kde=False,
                hist_kws={'label': pre})
ax.legend()

fig, ax = plt.subplots(1, figsize=[10, 8])
ax.set_title('L1 and L3')
ax.set_xlabel('Inputs from R1R4 + R3R6')
ax.set_ylabel('Number of cartridges')
for pre in ['LMC_1', 'LMC_3']:
    
    x = data.loc[data['type'] == pre, ('centri')]
    display(x)
    sns.distplot(x, ax=ax, color=c[pre], kde=False,
                hist_kws={'label': pre})
ax.legend()

# +
# PCA all LMCs by their INPUT VECTORS
pca = PCA(n_components=2)
lmc_ins = data.loc[[i for i, t in enumerate(data['type']) if t in lmcs], 
                   ('om', 'type', *subtypes)].drop('UNKNOWN', axis=1).reset_index()

X = lmc_ins.iloc[:, 3:]
display(X.columns)
X_r = pca.fit(X).transform(X)

fig = plt.figure(figsize=[10, 10])
ax = fig.gca()
ax.set_title("LMC inputs\n" + 
             f"explained variance ratio (first two PCs): {pca.explained_variance_ratio_}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

sts = []
for i, row in lmc_ins.iterrows():
    ax.scatter(X_r[i, 0], X_r[i, 1], color=c[row['type']], label=row['type'], marker=m[row['type']])
    sts.append(row['type'])
legend_elements = []

for s in sorted(np.unique(sts)):
    legend_elements.append(Line2D([0], [0], marker=m[s], color=c[s], label=s))
ax.legend(handles=legend_elements)
plt.show()

display(pca.components_)


# +
# PCA L1 and L3 by their INPUT VECTORS
pca = PCA(n_components=2)
lmc_ins = data.loc[[i for i, t in enumerate(data['type']) if t in ('LMC_1', 'LMC_3')], 
                   ('om', 'type', *subtypes)].drop('UNKNOWN', axis=1).reset_index()

X = lmc_ins.iloc[:, 3:]
X_r = pca.fit(X).transform(X)

fig = plt.figure(figsize=[10, 10])
ax = fig.gca()
ax.set_title("L1 and L3 inputs\n" + 
             f"explained variance ratio (first two PCs): {pca.explained_variance_ratio_}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
sts = []
for i, row in lmc_ins.iterrows():
    ax.scatter(X_r[i, 0], X_r[i, 1], color=c[row['type']], label=row['type'], marker=m[row['type']])
    sts.append(row['type'])
legend_elements = []
for s in sorted(np.unique(sts)):
    legend_elements.append(Line2D([0], [0], marker=m[s], color=c[s], label=s))
ax.legend(handles=legend_elements)
plt.show()

display(pca.components_)

# +
# PCA L1 and L3 by their INPUT VECTORS
pca = PCA(n_components=2)
lmc_ins = data.loc[[i for i, t in enumerate(data['type']) if t in ('LMC_1', 'LMC_2', 'LMC_3')], 
                   ('om', 'type', *subtypes)].drop('UNKNOWN', axis=1).reset_index()

X = lmc_ins.iloc[:, 3:]
X_r = pca.fit(X).transform(X)

fig = plt.figure(figsize=[10, 10])
ax = fig.gca()
ax.set_title("L1 and L3 inputs vectors\n" + 
             f"explained variance ratio (first two PCs): {pca.explained_variance_ratio_}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
for i, row in lmc_ins.iterrows():
    ax.scatter(X_r[i, 0], X_r[i, 1], color=c[row['type']], label=row['type'], marker=m[row['type']])

plt.show()

display(pca.get_params())

# +
# PCA all LMCs by their SHORT PHOTORECEPTOR INPUT VECTORS
pca = PCA(n_components=2)
lmc_ins = data.loc[[i for i, t in enumerate(data['type']) if t in lmcs], 
                   ('om', 'type', *subtypes)].drop('UNKNOWN', axis=1).reset_index()

X = lmc_ins.loc[:, ('R1R4', 'R2R5', 'R3R6')]
X_r = pca.fit(X).transform(X)

fig = plt.figure(figsize=[10, 10])
ax = fig.gca()
ax.set_title("->LMC short photoreceptor inputs\n" + 
             f"explained variance ratio (first two PCs): {pca.explained_variance_ratio_}")
for i, row in lmc_ins.iterrows():
    ax.scatter(X_r[i, 0], X_r[i, 1], color=c[row['type']], label=row['type'], marker=m[row['type']])

plt.show()

display(X_r.shape)

# +
# PCA all LMCs by their INPUT VECTORS
pca = PCA(n_components=3)

X = lmc_ins.iloc[:, 3:]
X_r = pca.fit(X).transform(X)

fig = plt.figure(figsize=[10, 10])
ax = fig.gca(projection='3d')
for i, row in lmc_ins.iterrows():
    ax.scatter(X_r[i, 0], X_r[i, 1], X_r[i, 2], color=c[row['type']], label=row['type'])
plt.show()
ax.legend()
#display(X_r)

# +
fig = plt.figure(figsize=[10, 10])
ax = fig.gca(projection='3d')
ax.set_xlabel('Presynaptic contacts')
ax.set_zlabel('Inputs from R1R4 + R3R6')
ax.set_ylabel('Inputs from neighboring cartridges')

for pre, rows in data.groupby('type'):
    if pre not in lmcs:
        continue
    else:
        #infrac = (rows['R1R4'] + rows['R3R6']) / (rows.filter(items=subtypes, axis=1).sum(axis=1))
        #infrac = (rows['R1R4'] + rows['R3R6']) / (rows['R1R4'] + rows['R2R5'] + rows['R3R6'])
        #infrac = rows['R1R4'] + rows['R3R6']
        infrac = rows['R1R4'] + rows['R3R6'] - rows['R2R5']
        ax.scatter(rows['output_count'], rows['inter_in'], infrac, 
                   label=f"{pre}, n = {len(rows)}", marker=m[pre], s=80, depthshade=True)
        #ax.scatter(rows['R1R4'] + rows['R3R6'], rows['inter_in'], rows['output_count'], label=pre)
ax.legend()

ax.invert_yaxis()
ax.view_init()
plt.show()

# +
fig = plt.figure(figsize=[10, 10])
ax = fig.gca(projection='3d')
ax.set_xlabel('Presynaptic contacts')
ax.set_ylabel('Inputs from neighboring cartridges')
ax.set_zlabel('R1R4 + R3R6 inputs / all inputs')

for pre, rows in data.groupby('type'):
    if pre not in lmcs:
        continue
    else:
        #infrac = (rows['R1R4'] + rows['R3R6']) / (rows.filter(items=subtypes, axis=1).sum(axis=1))
        #infrac = (rows['R1R4'] + rows['R3R6']) / (rows['R1R4'] + rows['R2R5'] + rows['R3R6'])
        infrac = rows['R2R5'] / (rows['R1R4'] + rows['R3R6']) 
        #infrac = rows['R1R4'] + rows['R3R6']
        #s = rows.filter(items=subtypes, axis=1).sum(axis=1) + 50
        ax.scatter(rows['output_count'], rows['inter_in'], infrac, label=pre, s=100, depthshade=True)
        #ax.scatter(rows['R1R4'] + rows['R3R6'], rows['inter_in'], rows['output_count'], label=pre)
ax.legend()

ax.invert_yaxis()
ax.view_init(elev=10)
plt.show()

# +
fig, ax = plt.subplots(1, figsize=[15, 15])
ax.set_xlabel('Number of presynaptic outputs')
ax.set_ylabel('Number of inputs from centri')

for pre, rows in data.groupby('type'):
    if pre not in lmcs:
        continue
    else:
        ax.scatter(x=rows['output_count'], y=rows['R2R5'])

        
# -


