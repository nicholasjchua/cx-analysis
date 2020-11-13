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

# # Optical variability of ommatidia
# - View ommatidia measurements from AM and team 
# - Calculate optical parameters from physical measurements
# - Perform clustering on these physical features, compare with lamina circuit clustering

# +
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.utils import index_by_om
from src.dataframe_tools import assemble_cxvectors
from vis.fig_tools import linear_cmap, subtype_cm
from vis.hex_lattice import hexplot
# -

optics_df = pd.read_excel('~/Data/data for ligh prop.xlsx', index_col=0)
optics_df = optics_df.iloc[:29] # remove last two lines of excel file
#display(optics_df)

dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
ndra_om = [str(o) for o in optics_df.index if o not in dra_om]

# ## Lens calculations
# ### Measurements (all $\mu{m}$)
# - Lens: Outer radius of curvature, $r_1$
# - Lens: Inner radius of curvature, $r_2$
# - Lens: thickness, $t$
# - Lens: diameter, $D$
# - Rhabdom: distal diameter, $D_r$
# - Measurements also taken for the length and distal width of the crystalline cone, and the proximal diameter of the rhabdom (not used for the following calculations) 
#
# ### Calculations
# - We used the refractive indices of the honey bee lens ($n_l$) and crystalline cone ($n_c$), from Valera & Wiitanen (1970)
# - Optical power ($\mu{m}^{-1}$) of the diopteric apparatus (lens + crystalline cone), $P$, calculated using the thick lens formula (Fundamentals of Optics, Jenkins & White, p.84, 2001): 
# $$P = P_1 + P_2 - \frac{t}{n_l} P_1 P_2$$
#     - $P_1 = \frac{n_l - 1.0}{r_1}$, power from outer lens surface
#     - $P_2 = \frac{n_c - n_l}{r_2}$, power from inner lens surface  
# - Focal length of the object ($\mu{m}$): $f = 1/P$
# - Focal length of the image ($\mu{m}$): $f' = {n_c}/P$
# - F-number: $F = f/D$
# - Acceptance angle of the ommatidium (radians), $\Delta\rho_s$, is approximated by diffraction at the lens,  $\Delta\rho_l$ and the geometry of the distal rhabdom tip and the lens, $\Delta\rho_r$ (Snyder, 1979):
# $$\Delta\rho_s = \sqrt{{\Delta\rho_l}^2 + {\Delta\rho_r}^2}$$
#     - $\Delta\rho_l = \lambda/D$, where we set $\lambda$ to 0.5 $\mu{m}$ (green monochromatic light)
#     - $\Delta\rho_r = D_{r}/f$
#



# +
r1 = optics_df['outer curvature']
r2 = -1 * optics_df['inner curvature']
t = optics_df['lense thickness']
D = optics_df['facet diameter (stack)']
Dr = optics_df['D rhabdom dist.']

# Refractive indices from Apis mellifera (Varela & Wiitanen 1970)
# n = 1 # air
nl = 1.452 # lens
nc = 1.348 # cone
# lens power IN MICROMETERS
p1 = (nl - 1.0)/r1 # interface air->lens 
p2 = (nc - nl)/r2 # interface lens->cone
p3 = (t/nl)*p1*p2  # thickness 'correction'
p = p1 + p2 - p3
# Focal length object and image (n/p)
f = 1.0/p
fi = nc/p
# F-number (ratio of lens diameter to focal length)
F = D/f
# Acceptance angle
rho_l = 0.5/D # angular sens. due to diffraction at the lens, lambda=0.5
rho_r = Dr/f # angular sens. due to geometry of rhabdom tip
# 'simple' acceptance angle formula 
rho = (rho_l ** (2.0) + rho_r ** (2.0)) ** (0.5)

optics_df['f'] = f 
optics_df['f-image'] = fi
optics_df['P'] = p
optics_df['F-number'] = F
optics_df['diffraction-rho'] = rho_l
optics_df['geometric-rho'] = rho_r
optics_df['rho'] = rho
# -

# - Diffraction by the facet lens depends on light wavelength, lens diameter, and its focal distance
# - Male blowfly, Calliphora (Stavenga 1990): D=20-40 $\mu{m}$; $f/D$ remains relatively constant despite range of D, giving $F = 2.0{\pm}0.2$ (based on optical measurements); 

optics_df['mesh_diameter'] = 

display(optics_df.loc[:,'facet diameter (stack)'])

# +
fig, ax = plt.subplots(1, 2, figsize=[20, 10])

sns.regplot(x=optics_df['cone length (from the tip)'] + optics_df['lense thickness'], 
            y='f', data=optics_df, ax=ax[0])
ax[0].set_title('focal length of lens vs length from cornea to rhabdom')
ax[0].set_xlabel('lens thickness + cone thickness')
ax[0].set_ylabel('focal length')

# +
# f_pos = optics_df['cone length (from the tip)'] + optics_df['lense thickness'] - optics_df['f']
# fig, ax = plt.subplots(1, figsize = [10, 10])
# lincm = linear_cmap(n_vals=100, max_colour='r', min_colour='b', mp_colour=(1, 1, 1))
# abs_max = f_pos.abs().max() 
# display(abs_max)

# node_data = {om: {'colour': lincm((x-f_pos.min())/abs_max),
#                  'label': f"{x: .2f}"} for om, x in f_pos.items()}

# hexplot(node_data, ax=ax)

# +
fig, ax = plt.subplots(7, 2, figsize=(24, 77))
axes = ax.flatten()
#labels = ['focal length of lens (\u03BCm)', 'focal length in image plane (\u03BCm)',
         #'lens power', 'F-number', 'acceptance angle (radians)',
         #'Airy disk half-width (\u03BCm)', '']



i = 0
for param, vals in optics_df.iteritems():
    max_val = vals.max()
    min_val = vals.min()
    node_data = {om: {'label': np.round(v, decimals=2),
                     'colour': lincm((v-min_val)/max_val)} for om, v in vals.items()}
    hexplot(node_data, ax=axes[i])
    axes[i].set_title(f"{param}\n" + 
                      f"Mean: {vals.mean(): .2f}\n" + 
                      f"SD: {vals.std(): .2f}")
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
