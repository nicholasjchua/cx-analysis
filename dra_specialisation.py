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

# # Dorsal Rim Specialisation

# +
# # %load_ext autoreload
# # %autoreload 2

# +
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import itertools
from sklearn.linear_model import LinearRegression
#from statsmodels.stats.weightstats import ttest_ind
from scipy.stats import mannwhitneyu
import statsmodels.api as sm

from src.utils import index_by_om
from vis.colour_palettes import subtype_cm
from vis.fig_tools import linear_cmap
from vis.hex_lattice import hexplot

import matplotlib as mpl
mpl.rc('font', size=14)


# +
# Each ommatidia's onnection counts
data_dir = '~/Data/2002_lamina'
cxdf = pd.read_pickle(os.path.join(os.path.expanduser(data_dir), '200128_cxdf.pickle'))
widedf = index_by_om(cxdf)  # pivot so that each row is an ommatidium with columns for each possible connection count

# Anastasia's measurements for each Rhabdom's twist
twistdf = pd.read_csv('~/Data/lamina_additional_data/1911_am_rhab_twist.csv', index_col=0)

# Retinotopic regions
om_list = [str(o) for o in widedf.index]
dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
ndra_om = [str(o) for o in widedf.index if o not in dra_om]

cm = subtype_cm()
# -

# ## Lamina connectome reveals different R7 wiring patterns in the dorsal area of the retinotopic field

# +
# Retinotopic map
fig, ax = plt.subplots(1, 2, figsize=[20, 15])

r7_inputs = widedf.filter(items=['centri->R7', 'LMC_2->R7', 'R2R5->R7']).sum(axis=1)
r7cm = linear_cmap(n_vals=r7_inputs.max() - r7_inputs.min(), max_colour=cm['R7'])
node_data = {k: {'colour': r7cm(v/r7_inputs.max()),
                'label': str(int(v))} for k, v in r7_inputs.items()}
hexplot(node_data=node_data, ax=ax[0])
ax[0].set_title("Number of inputs to R7")

r7p_inputs = widedf.filter(items=['centri->R7p', 'LMC_2->R7p', 'R2R5->R7p']).sum(axis=1)
r7pcm = linear_cmap(n_vals = r7p_inputs.max() - r7p_inputs.min(), max_colour=cm['R7p'])
node_data = {k: {'colour': r7pcm(v/r7p_inputs.max()),
                'label': str(int(v))} for k, v in r7p_inputs.items()}
ax[1].set_title("Number of inputs to R7'")

hexplot(node_data=node_data, ax=ax[1])
plt.show()
# -

# ## DRA ommatidia receive significantly more inputs to R7 than Non-DRA ommatidia
# Two sample Mann-Whitney U test (one tailed)
#
# $$H_{0}: P(x_{i} > y_{j}) <= 1/2$$
#
# $$H_{1}: P(x_{i} > y_{j}) > 1/2$$
#
# Where x is the number of R7 inputs observed in DRA ommatidium i, and y is the number of R7 inputs in NDRA ommatidium j

s, p, = mannwhitneyu(r7_inputs[dra_om], r7_inputs[ndra_om], alternative='greater')
print("###### RESULTS ######")
print(f"Test statistic: {s}, p-value: {p: .6f}")
if p > 0.001:
    print("Fail to reject null")
else:
    print("Reject null: DRA R7 receive more inputs")

# ## Angular displacement of rhabdom microvilli is significantly smaller in DRA ommatidia 
# Two sample Mann-Whitney U test (one tailed)
#
# $$H_{0}: P(x_{i} < y_{j}) <= 1/2$$
#
# $$H_{1}: P(x_{i} < y_{j}) > 1/2$$
#
# Where x is the distal - proximal angular difference of the rhabdom of ommatidium i, and y is the distal - proximal angular difference of the rhabdom of ommatidium j

s, p = mannwhitneyu(twistdf.loc[dra_om], twistdf.loc[ndra_om], alternative='less')
print("###### RESULTS ######")
print(f"Test statistic: {s}, p-value: {p: .6f}")
if p > 0.001:
    print("Fail to reject null")
else:
    print("Reject null: DRA rhabdoms twist at a smaller degree")

# +
fig, ax = plt.subplots(1, 2, figsize=[20, 15])
ax[0].margins(x=0.2)
ax[0].set_yticks(ticks=np.arange(0, r7p_inputs.max(), 5), minor=True)
ax[0].set_xlabel("Postsynaptic Subtype")
ax[0].set_ylabel("Number of Connections")

for o in om_list:
    if o in dra_om:
        linestyle = '--'
    else:
        linestyle = '-'
        
    ax[0].plot(['Inputs to R7', 'Inputs to R7p'], [r7_inputs[o], r7p_inputs[o]], color=cm['R7p'], linestyle=linestyle)
    ax[0].scatter(['Inputs to R7', 'Inputs to R7p'], [r7_inputs[o], r7p_inputs[o]], color='k')
lines = [Line2D([0], [0], color=cm['R7p'], linestyle='--'),
        Line2D([0], [0], color=cm['R7p'], linestyle='-')]
ax[0].legend(lines, ['DRA ommatidia', 'NDRA ommatidia'])

# ~~~~~~~~~~~~~~~

data = [twistdf.loc[ndra_om].to_numpy(), twistdf.loc[dra_om].to_numpy()]
ax[1].set_title("Distal-proximal microvilli orientatation difference")

bp = ax[1].boxplot(data, 0, 'kD', 0, patch_artist=True)
ax[1].set_yticklabels(["NDRA", "DRA"])
ax[1].set_xlabel("Angular difference (degrees)")
bp["boxes"][0].set_facecolor(cm['R7'])
bp["boxes"][1].set_facecolor(cm['R7'])

plt.show()

# +


sns.distplot(r7_inputs)
sns.distplot(r7p_inputs)
# -



# ## Relationship between connection counts and optical measurements

'''
def lin_model_intercept0(x, y):
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y)
    return LinearRegression(fit_intercept=False).fit(x, y)

fig, ax = plt.subplots(1)
print([len(widedf['centri->R7']), len(twistdf.loc[widedf.index])])
x = widedf['centri->R7']
y = twistdf.loc[widedf.index]   # TODO: change col name

xticks = np.arange(0, x.max()).reshape(-1, 1)
r7_v_twist = lin_model_intercept0(x, y)
ax.plot(xticks, r7_v_twist.predict(xticks))
ax.set_title(f"R^2 = {r7_v_twist.score(widedf['centri->R7'].to_numpy().reshape(-1, 1), twistdf.loc[om_list])}")
'''

# +
fig, ax = plt.subplots(1, figsize=[15, 15])
ax.set_xlabel('Number of R7 inputs')
ax.set_ylabel('Rhabdom angular difference (degrees)')

#display(widedf.filter(regex='->R7$'))
total_r7 = widedf.filter(regex='->R7$').sum(axis=1)
total_r7 = total_r7.loc[total_r7.index != 'C6']

x = np.asarray(total_r7).reshape(-1, 1)
y = np.asarray(twistdf.loc[total_r7.index])
model = LinearRegression().fit(x, y)
xticks = np.arange(0, total_r7.max()).reshape(-1, 1)

ax.plot(xticks, model.predict(xticks), label=f"R^2 = {model.score(x, y): .3f}")
ax.scatter(x, y)
ax.legend()

# -






