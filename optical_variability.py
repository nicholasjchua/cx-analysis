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

# # Optical variability between ommatidia
# - View ommatidia optical features measured by AM and team 
# - Perform clustering on these features 

# +
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import index_by_om
from src.dataframe_tools import assemble_cxvectors
from vis.colour_palettes import subtype_cm
from vis.fig_tools import linear_cmap
from vis.hex_lattice import hexplot
# -

# R7 and R7p rhabdomere twist data
sevens_df = pd.read_csv('~/Data/r7r7p_microvilli_sd.csv', index_col=0).T
#display(df)
# Connectivity data designations
dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
ndra_om = [str(o) for o in sevens_df.index if o not in dra_om]



# +
fig, ax = plt.subplots(1, 2, figsize=(24, 10))
max_val =  df.max().max()
#cm = subtype_cm()
lincm = linear_cmap(n_vals=100, max_colour='r')

r7_twist = dict()
r7p_twist = dict()
for om, vals in df.iterrows():
    r7_twist[om] = {'colour': lincm(vals['R7 SD']/max_val),
                   'label': vals['R7 SD']}
    r7p_twist[om] = {'colour': lincm(vals["R7' SD"]/max_val),
                   'label': vals["R7' SD"]}

hexplot(node_data=r7_twist, ax=ax[0])
hexplot(node_data=r7p_twist, ax=ax[1])

ax[0].set_title('R7 Rhabdomere twist \n(standard deviation of microvilli angle)')
ax[1].set_title("R7' Rhabdomere twist \n(standard deviation of microvilli angle)")

# +
fig, ax = plt.subplots(1, 2, figsize=(24, 10))
max_val =  df.max().max()
min_val = df.min().min()
#cm = subtype_cm()
lincm = linear_cmap(n_vals=100, max_colour='r')

r7_twist = dict()
r7p_twist = dict()
for om, vals in df.iterrows():
    r7_twist[om] = {'colour': lincm((vals['R7 SD'] - min_val)/max_val),
                   'label': vals['R7 SD']}
    r7p_twist[om] = {'colour': lincm((vals["R7' SD"] - min_val)/max_val),
                   'label': vals["R7' SD"]}

hexplot(node_data=r7_twist, ax=ax[0])
hexplot(node_data=r7p_twist, ax=ax[1])

ax[0].set_title('R7 Rhabdomere twist \n(standard deviation of microvilli angle)')
ax[1].set_title("R7' Rhabdomere twist \n(standard deviation of microvilli angle)")
# -

optics_df = pd.read_excel('~/Data/data for ligh prop.xlsx', index_col=0)
optics_df = optics_df.iloc[:29] # remove last two lines of excel file
#display(optics_df)

# +
r1 = optics_df['outer curvature']
r2 = -1 * optics_df['inner curvature']
t = optics_df['lense thickness']
A = optics_df['facet diameter (stack)']
Drh = optics_df['D rhabdom dist.']

# Refractive indices from Apis mellifera (Varela & Wiitanen 1970)
# n = 1 # air
nl = 1.452 # lens
nc = 1.348 # cone
# Power = P1 + P2 + P3 (thick lens formula)
p1 = (nl - 1.0)/r1  # interface air->lens 
p2 = (nc - nl)/r2 # interface lens->cone
p3 = (-t/nl)*(p1*p2)
p = p1 + p2 + p3
# Focal length of lens and image (n/p)
f = 1.0/p
fi = nc/p
# F-number (ratio of lens diameter to focal length)
FN = A/f
# Acceptance angle of rhabdom
# distal rhabdomere diameter / focal length
aa = Drh/f
# Half-width of airy disk
lam = 0.5 # assuming green light with wavelength=0.5 microns
hw = lam/A

optics_df['f'] = f 
optics_df['fi'] = fi
optics_df['power'] = p
optics_df['F-number'] = FN
optics_df['acceptance_angle'] = aa
optics_df['half-width'] = hw

#display(optics_df.iloc[:, -6:])

#### SAVE CALCULATIONS ####
#optics_df.to_pickle('~/Data/200713_optics_calcs.pkl')

# +
fig, ax = plt.subplots(3, 2, figsize=(24, 33))
axes = ax.flatten()
labels = ['focal length of lens (\u03BCm)', 'focal length in image plane (\u03BCm)',
         'lens power', 'F-number', 'acceptance angle (radians)',
         'Airy disk half-width (\u03BCm)']

i = 0
for param, vals in optics_df.iloc[:, -6:].iteritems():
    max_val = vals.max()
    min_val = vals.min()
    node_data = {om: {'label': np.round(v, decimals=2),
                     'colour': lincm((v-min_val)/max_val)} for om, v in vals.items()}
    hexplot(node_data, ax=axes[i])
    axes[i].set_title(f"{labels[i]}")
    # TO CHECK IF LABELS AND PARAMS LINE UP
    #axes[i].set_title(f"{labels[i]}\n{param}")
    i += 1

# +
fig, ax = plt.subplots(3, 2, figsize=(24, 33))
axes = ax.flatten()
labels = ['focal length of lens (\u03BCm)', 'focal length in image plane (\u03BCm)',
         'lens power', 'F-number', 'acceptance angle (degrees)',
         'Airy disk half-width (\u03BCm)']

i = 0
for param, vals in optics_df.iloc[:, -6:].iteritems():
    max_val = vals.max()
    node_data = {om: {'label': np.round(v, decimals=2),
                     'colour': lincm(v/max_val)} for om, v in vals.items()}
    hexplot(node_data, ax=axes[i])
    axes[i].set_title(f"{labels[i]}")
    # TO CHECK IF LABELS AND PARAMS LINE UP
    #axes[i].set_title(f"{labels[i]}\n{param}")
    i += 1
    
    

# +
tp = '200507'
linkdf = pd.read_pickle(f'~/Data/{tp}_lamina/{tp}_linkdf.pickle')

cxvecs = assemble_cxvectors(linkdf)
thresh = cxvecs.mean()>1.0
cxvecs = cxvecs.loc[:, thresh]
homevecs = cxvecs.loc[:, [i for i in cxvecs.columns if '->e' not in i]]

display(homevecs)
# -

# combined connectivity and optics into one df 
combined = homevecs.join(optics_df)
assert(all(combined.notna())) # should not contain any NaNs
#display(combined)

# +

print(f"{len(homevecs.columns)} connection types")

all_corr = combined.corr()
cx_v_optics = all_corr.filter(items=homevecs.columns, axis=0).filter(items=optics_df.columns, axis=1)
#display(cx_v_optics)

clus = sns.clustermap(cx_v_optics, 
                      linewidth=0.1,
                      figsize=[11, 11], metric='cosine', 
                      cmap='vlag', vmax=1.0, vmin=-1.0)

#row_colors = [cm[x.split('->')[0]] for x in cx_corr.index]
# -

fig, ax = plt.subplots(1)
r7_totals = 
sns.regplot()
