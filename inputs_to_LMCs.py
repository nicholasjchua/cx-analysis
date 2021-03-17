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
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression

from src.dataframe_tools import assemble_cxvectors
from vis.hex_lattice import hexplot
from vis.fig_tools import linear_cmap, subtype_cm

# +
plt.rcdefaults()
plt.style.use('vis/lamina.mplstyle')
cm = subtype_cm() # a dict containing the color for each lamina subtype

savefigs = False
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
mean_thresh = 0.0
#cxvecs = cxvecs.loc[:, cxvecs.mean() >= mean_thresh]
display(f"Number of connections observed (where mean count >= {mean_thresh}): {len(cxvecs.columns)}")

only_home_connections = True
if only_home_connections:
    cxvecs = cxvecs.loc[:, [i for i in cxvecs.columns if ('eLMC_4' not in i) and ('eLMC_2' not in i)]]
display(f"Number of connections after removing inter-ommatidial: {len(cxvecs.columns)}")

# +
svfs = ['R1R4', 'R2R5', 'R3R6']
lmcs = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4', 'LMC_N']
feedforward = [pre + '->' + post for pre, post in itertools.product(svfs, lmcs)]


cp_ratio = pd.DataFrame()
svf_lmc = cxvecs.filter(items=feedforward)
combined_svf = pd.DataFrame()

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
ax.set_ylabel(r'# inputs from $\frac{R2+R5}{R1+R3+R4+R6}$')

if savefigs:
    fig.savefig(savepath + '/LMC_r1-6_bias.svg')
    fig.savefig(savepath + '/LMC_r1-6_bias.png')


# -

# ### LMC subtypes sample from R1-6 at different proportions
# **Boxplot depicting the ratio of photoreceptor inputs received by L1-4.** L1 and L2 receive approximately the same proportion of inputs from R2 and R5 as they do from each of the other four short photoreceptors (R1, R3, R4, and R6). L3 and L4 show a bias for R2 and R5 inputs. Boxes depict the interquartile range observed among lamina cartridges. Whiskers and point beyond describe data points beyond the interquartile range. 

post_types = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4']
pairwise_results = pd.DataFrame(index=post_types, columns=post_types)
for v1, v2 in itertools.permutations(post_types, 2):
    if v1 == v2:
        pairwise_results.loc[v1, v2] = np.nan()
        continue
    else:
        s, p, = mannwhitneyu(cp_ratio[v1], cp_ratio[v2], alternative='greater')
        print(f"### Null: {v1} does not sample from a larger proportion of R2 and R5 compared to {v2} ###")
        print(f"Test statistic (Mann-Whitney U): {s}, p-value: {p: .2e}")
        pairwise_results.loc[v1, v2] = p
        if p >= 0.01:
            print("Fail to reject null (P >= 0.01)")
        else:
            print(f"Reject null : {v1} more selective than {v2}")
        print('\n')
print("Pairwise p-values (ratio of row subtype more than column subtype)")
pd.set_option("display.precision", 2)
display(pairwise_results)



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




