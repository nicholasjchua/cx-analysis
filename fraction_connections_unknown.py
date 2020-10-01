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

# # Fraction of connections unidentified
# - Queries our lamina connectome for a list of all connections (also called 'links' in catmaid) to the number of those with unidentified postsynaptic partners. 
# - The list of unknowns is broken down by ommatidia and displayed in a retinotopic heatmap. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vis.hex_lattice import hexplot
from vis.fig_tools import hex_to_rgb, linear_cmap

tp = '200914'
data_path = f"~/Data/{tp}_lamina/{tp}"
#cx = pd.read_pickle(data_path + "_cxdf.pickle")
links = pd.read_pickle(data_path + "_linkdf.pickle")

# ## Number of synaptic inputs received by each subtype

input_counts = links.groupby('post_type')['link_id'].nunique()  
display(counts)

# +
p_unknown = dict()
n_unknown = dict()
n_syn = dict()
n_term = dict()

for om, l in links.groupby('pre_om'):
    unknown_count = (l['post_type'] == 'UNKNOWN').sum()
    percent = float(unknown_count)/float(len(l['link_id']))

    # make sure each link ID is unique (preprocessing should get rid of pre/post duplicates)
    assert(len(l['link_id'].unique()) == len(l['link_id']))
    n_term[om] = l['cx_id'].nunique()
    n_syn[om] = l['link_id'].nunique()
    p_unknown[om] = percent
    n_unknown[om] = unknown_count

# -

om_summary = pd.DataFrame(data={'n_syn': n_syn, 
                                'n_unknown':n_unknown, 
                                'n_term': n_term})
om_summary['n_known'] = om_summary['n_syn'] - om_summary['n_unknown']
om_summary

# +
fig, ax = plt.subplots(1, figsize=[8, 10])
cm = linear_cmap(n_vals=100, max_colour='#a83232', min_colour='#ffffff')
high = om_summary['n_syn'].max()

hex_data = {om: {'colour': cm(v/high),
                 'label': f'{v: .0f}'} for om, v in om_summary['n_syn'].items()}
ax.set_title(f"Total number of synaptic contacts\n" + 
            f"mean = {om_summary['n_syn'].mean(): .1f}\n" + 
            f"SD = {om_summary['n_syn'].std(ddof=0): .1f}")

hexplot(hex_data, ax=ax)
plt.show()


# +
fig, ax = plt.subplots(1, figsize=[8, 10])
cm = linear_cmap(n_vals=100, max_colour='#a83232', min_colour='#ffffff')
high = om_summary['n_known'].max()

hex_data = {om: {'colour': cm(v/high),
                 'label': f'{v: .0f}'} for om, v in om_summary['n_known'].items()}
ax.set_title(f"Total number of IDENTIFIED synaptic contacts\n" + 
            f"mean = {om_summary['n_known'].mean(): .1f}\n" + 
            f"SD = {om_summary['n_known'].std(ddof=0): .1f}")

hexplot(hex_data, ax=ax)
plt.show()


# +
fig, ax = plt.subplots(1, figsize=[8, 10])
cm = linear_cmap(n_vals=100, max_colour='#a83232', min_colour='#ffffff')
high = om_summary['n_term'].max()

hex_data = {om: {'colour': cm(v/high),
                 'label': f'{v: .0f}'} for om, v in om_summary['n_term'].items()}
ax.set_title(f"Total number of presynaptic terminals\n" + 
            f"mean = {om_summary['n_term'].mean(): .1f}\n" + 
            f"SD = {om_summary['n_term'].std(ddof=0): .1f}")

hexplot(hex_data, ax=ax)
plt.show()

# +
fig, ax = plt.subplots(1, figsize=[8, 10])
high = max(list(p_unknown.values()))

hex_data = {om: {'colour': cm(v/high),
                 'label': f'{v: .2f}'} for om, v in p_unknown.items()}

ax.set_title(f"Percentage of synaptic contacts with an unidentified partner\n" + 
            f"mean = {np.mean(list(p_unknown.values())): .2f}\n" + 
            f"SD = {np.std(list(p_unknown.values())): .2f}")
hexplot(hex_data, ax=ax)
plt.show()

# -


