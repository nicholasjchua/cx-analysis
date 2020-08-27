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

import pandas as pd
import matplotlib.pyplot as plt
from vis.hex_lattice import hexplot
from vis.fig_tools import hex_to_rgb

tp = '200507'
data_path = f"~/Data/{tp}_lamina/{tp}"
#cx = pd.read_pickle(data_path + "_cxdf.pickle")
links = pd.read_pickle(data_path + "_linkdf.pickle")

counts = links.groupby('post_type')['link_id'].nunique()  
display(counts)

p_by_om = dict()
c_by_om = dict()

for om, l in links.groupby('pre_om'):
    count = (l['post_type'] == 'UNKNOWN').sum()
    percent = count/len(l)  # unknowns / total
    
    p_by_om[om] = percent
    c_by_om[om] = count

# +
cm = linear_cmap(n_vals=100, max_colour='#a83232', min_colour='#ffffff')
high = max(list(p_by_om.values()))

hex_data = {om: {'colour': cm(v/high),
                 'label': f'{v: .3f}'} for om, v in p_by_om.items()}

# +
fig, ax = plt.subplots(1, figsize=[8, 10])

hexplot(hex_data, ax=ax)
plt.show()

# -


