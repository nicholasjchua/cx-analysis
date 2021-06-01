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

# # Collateral connectivity between ommatidia
# WORK IN PROGRESS
# - Quantifies the number of connections in the lamina connectome between neurons associated with different ommatidia (collateral connections)
# - Displays the interommatidial flow of information facilitated by this collateral connectivity
#
# This analysis focuses on L4 and L2, which were the only lamina subtypes exhibiting clear collateral arbors. These collaterals are also found to be exclusively postsynaptic, i.e. they ***receive*** inputs from external ommatidia. 

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vis.hex_lattice import hexplot
from vis.colour_palettes import subtype_cm
from vis.fig_tools import linear_cmap
# -

tp = '200507'
data_path = f"~/Data/{tp}_lamina/{tp}"
cx = pd.read_pickle(data_path + "_cxdf.pickle")
links = pd.read_pickle(data_path + "_linkdf.pickle")

# +
collateral_links = []
home_links = []
om_list = []

for i, l in links.iterrows():
    if (l.post_om != 'UNKNOWN'):
        om_list.append(l.pre_om)
        
    if (l.pre_om != l.post_om) and (l.post_om != 'UNKNOWN'):
        collateral_links.append(l)
    elif (l.pre_om == l.post_om) and (l.post_om != 'UNKNOWN'):
        home_links.append(l)
    else:
        continue
    
om_list = list(sorted(set(om_list)))
collaterals = pd.DataFrame(collateral_links)
homes = pd.DataFrame(home_links)

display(f"Total contacts: {len(links)}")
display(f"Total identified contacts: {len(homes) + len(collaterals)}")
display(f"% UNKNOWN: {int(sum(links.post_om == 'UNKNOWN'))/int(len(links)) * 100.0}")
display(f"Number of interom contacts: {len(collaterals)}")
display(f"Number of home contacts: {len(homes)}")
display(f"Interom % of total: {float(len(collaterals))/float(len(homes) + len(collaterals))}")

# +
l4_co = collaterals.loc[collaterals['post_type'] == 'LMC_4']
fig, ax = plt.subplots(1)

display(l4_co.groupby('pre_type')['link_id'].nunique())
ax.hist(l4_co.groupby('post_om')['pre_type'].nunique())

# +
data = []

for post_om, links in l4_co.groupby('post_om'):
    for pre_om, l in links.groupby('pre_om'):
#         display(f"{pre_om} to {post_om}")
#         display(f"Number of contacts: {len(l)}")
#         display(f"Line width: {len(l)}")
        data.append({"pre_om": pre_om,
                     "post_om": post_om, 
                     "n_contacts": len(l)})

        
l4_inter_counts = pd.DataFrame(data, columns=['pre_om', 'post_om', 'n_contacts'])
#display(l4_inter_counts)

l4_inter_counts['line_width'] = ((l4_inter_counts['n_contacts']/l4_inter_counts['n_contacts'].max()) * 4) + 2
l4_inter_counts['line_width'] = l4_inter_counts['line_width'].round(decimals=1)
display(l4_inter_counts)

# +
to_l4 = dict.fromkeys(om_list, 0)
to_l2 = dict.fromkeys(om_list, 0)
to_all = dict.fromkeys(om_list, 0)

weirdos = []

for i, c in collaterals.iterrows():
        om = c.post_om
        to_all[om] += 1
        if c.post_type == 'LMC_2':
            to_l2[om] += 1
        elif c.post_type == 'LMC_4':
            to_l4[om] += 1
        else:
            weirdos.append(c)

# -

weirdos

# +
fig, ax = plt.subplots(1, 3, figsize=[30, 15])
cm = linear_cmap(n_vals=100, max_colour='k')

l2_dat = {om: {'colour': cm(v/max(to_l2.values())), 
               'label': int(v)} for om, v in to_l2.items()}
l4_dat = {om: {'colour': cm(v/max(to_l4.values())), 
               'label': int(v)} for om, v in to_l4.items()}
all_dat = {om: {'colour': cm(v/max(to_all.values())), 
               'label': int(v)} for om, v in to_all.items()}

hexplot(node_data=l2_dat, ax=ax[0])
ax[0].set_title('Number of interommatidial input to L2 (recipient om)')
hexplot(node_data=l4_dat, ax=ax[1])
ax[1].set_title('Number of interommatidial inputs to L4 (recipient om)')
hexplot(node_data=all_dat, ax=ax[2])
ax[2].set_title('Number of interommatidial inputs to ALL(recipient om)')

plt.show()


# +
out_to_l2 = dict.fromkeys(om_list, 0)
out_to_l4 = dict.fromkeys(om_list, 0)
out_to_all = dict.fromkeys(om_list, 0)
out_weirdos = []

for i, c in collaterals.iterrows():
        om = c.pre_om
        out_to_all[om] += 1
        if c.post_type == 'LMC_2':
            out_to_l2[om] += 1
        elif c.post_type == 'LMC_4':
            out_to_l4[om] += 1
        else:
            out_weirdos.append(c)

# +
fig, ax = plt.subplots(1, 3, figsize=[30, 15])
cm = linear_cmap(n_vals=100, max_colour='k')

l2_dat = {om: {'colour': cm(v/max(out_to_l2.values())), 
               'label': int(v)} for om, v in out_to_l2.items()}
l4_dat = {om: {'colour': cm(v/max(out_to_l4.values())), 
               'label': int(v)} for om, v in out_to_l4.items()}
all_dat = {om: {'colour': cm(v/max(out_to_all.values())), 
                'label': int(v)} for om, v in out_to_all.items()}
hexplot(node_data=l2_dat, ax=ax[0])
ax[0].set_title('Number of interommatidial inputs to neighbor L2 (Provider om)')
hexplot(node_data=l4_dat, ax=ax[1])
ax[1].set_title('Number of interommatidial inputs to neighbor L4 (Provider om)')
hexplot(node_data=all_dat, ax=ax[2])
ax[2].set_title("Number of interommatidial inputs to neighbor (Provider om)")

plt.show()

# -



# +
to_l4_home = dict.fromkeys(om_list, 0)
to_l2_home = dict.fromkeys(om_list, 0)
to_all_home = dict.fromkeys(om_list, 0)

for i, h in homes.iterrows():
        om = h.post_om
        to_all_home[om] += 1
        if c.post_type == 'LMC_2':
            to_l2_home[om] += 1
        elif c.post_type == 'LMC_4':
            to_l4_home[om] += 1
        else:
            continue


# +
fig, ax = plt.subplots(1, 3, figsize=[30, 15])
cm = linear_cmap(n_vals=100, max_colour='k')

l2_dat = {om: {'colour': cm(v/(max(to_l2_home.values()) + 1)), 
               'label': int(v)} for om, v in to_l2_home.items()}
l4_dat = {om: {'colour': cm(v/(max(to_l4_home.values()) + 1)), 
               'label': int(v)} for om, v in to_l4_home.items()}
all_dat = {om: {'colour': cm(v/(max(to_all_home.values()) + 1)), 
               'label': int(v)} for om, v in to_all_home.items()}

hexplot(node_data=l2_dat, ax=ax[0])
ax[0].set_title('Number of home inputs to L2 (recipient om)')
hexplot(node_data=l4_dat, ax=ax[1])
ax[1].set_title('Number of home inputs to L4 (recipient om)')
hexplot(node_data=all_dat, ax=ax[2])
ax[2].set_title('Number of home inputs to ALL(recipient om)')

plt.show()

# -




