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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # The relationship betwwen optical variability and lamina circuitry
# - Why does the *Megaphragma* eye have two distinct sets of ommatidia? How might visual information from these regions differ?
# - Which aspects of the lamina circuit reflect optical variability in the retina? What does this say about these types of connections?
#
# Prior observations:
# - $D_{Rh} / f$ is relatively constant among ommatidia
# - Based on their lens optics, dorsal ommatidia  
# - Dorsal ommatidia have wider acceptance angle, but more diffracted optics
# - 
# - Variance of R1-6 t-bar counts within subtype pairs e.g. (R2, R5) could be explained by variance in rhabdom volume, $D_{rh}/f$, $D/f$, as proxies for light admittance or photon capture

# %load_ext autoreload
# %autoreload 2

# +
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import statsmodels.api as sm

from src.utils import index_by_om
from src.dataframe_tools import assemble_cxvectors, extract_connector_table
from vis.fig_tools import linear_cmap, subtype_cm
from vis.hex_lattice import hexplot_TEST, hexplot

plt.rcdefaults()
plt.style.use('vis/lamina.mplstyle')

cm = subtype_cm()
# -

### SAVE FIGS? ###
save_figs=False
##################
#if save_figs:
#    fig.savefig('/mnt/home/nchua/Dropbox/lamina_figures/FNAME.svg')
#    fig.savefig('/mnt/home/nchua/Dropbox/lamina_figures/FNAME.png')

# ### Optical measurements

optics_df = pd.read_excel('~/Data/data for ligh prop.xlsx', index_col=0)
optics_df = optics_df.iloc[:29] # remove last two lines of excel file

# +
# Add pre-computed rhabdom lengths (from rhabdomere_lengths.ipynb)
rb_len = pd.read_pickle('~/Dropbox/Data/201123_rh_len_df.pickle')
for i, v in rb_len.iterrows():
    optics_df.loc[i, 'rhabdom_len'] = float(v)

# Rhabdomere vols from Anastasia: trying be consistent rb = rhabdomere, Rbd = Rhabdom
rb = pd.read_csv('~/Data/lamina_additional_data/ret_cell_vol.csv').set_index('rtype').T
rb.index.name = 'om'
rb = rb.loc[sorted(rb.index), sorted(rb.columns)]
Rbd_frac = (rb.T/rb.sum(axis=1)).T.rename(mapper={'vol': 'fvol'}, axis=1)

# +
pr_types = rb.columns
subtypes = np.unique(link_df['post_type'])
ommatidia = np.unique(link_df['pre_om'])

dra_om = ['A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E7']
ndra_om = [o for o in ommatidia if o not in dra_om]


# -

# Specifies colors for DRA and non-DRA ommatidia
def om_colors(om_list):
    c_list = []
    for o in ommatidia:
        if str(o) in dra_om:
            c_list.append('darkviolet')
        else:
            c_list.append('darkgreen')
    return c_list


# ### Lamina connections

# long-form of each synaptic link (each connection)
tp = '200914'
data_path = f'~/Data/{tp}_lamina/{tp}_linkdf.pickle'
link_df = pd.read_pickle(data_path)

# Wide-form data frame with cols for each type of connection 
cxvecs = assemble_cxvectors(link_df)
# thresh = 3.0
# cxvecs = cxvecs.loc[:, cxvecs.mean() > thresh].fillna(0)  # filter out connections with mean less than 1
# cxvecs = cxvecs.rename_axis(index='om')

# Table with rows for each connector (presynaptic terminal)
ct_df = extract_connector_table(link_df) # DataFrame of connectors (presyn terminals)

# +
# Summarize connector table
n_terminals = {om: dict.fromkeys(rtypes) for om in ommatidia}
n_contacts = {om: dict.fromkeys(rtypes) for om in ommatidia}
n_outs = df['pre_neuron'].value_counts().to_dict() # count of links associated with every neuron

# Filter out non-short PR contacts/terminals
# TODO: helper function to add column for sub sub type (e.g. 'R1' instead of R1R4)
# This can be done from link_df, TODO: MAKE MORE GENERAL AND PUT IN DATAFRAME TOOLS
for pre_name, these_rows in ct_df.groupby('pre_neuron'):
    # using our neuron name pattern to get ommatidium/rtypes of indv photoreceptors
    if pre_name[0:2] == 'om' and pre_name[5] == 'R':  
        om = pre_name[2: 4]
        r = pre_name.split('_')[1]
        assert(len(r) in (2, 3))
        n_terminals[om][r] = len(these_rows)
        n_contacts[om][r] = n_outs.get(pre_name, 0)
    else:
        continue

terms = pd.DataFrame(n_terminals).fillna(0).astype(int).T
cxs = pd.DataFrame(n_contacts).fillna(0).astype(int).T

terms.index.name = 'om'
cxs.index.name = 'om'
# + {}
rtypes = cxvecs.index
subtypes = np.unique(df['post_type'])
ommatidia = np.unique(df['pre_om'])

ct_df = extract_connector_table(link_df) # DataFrame of connectors (presyn terminals)
# -

# ## Lens calculations
# ### Measurements (all $\mu{m}$)
# - Lens: Outer radius of curvature, $r_1$
# - Lens: Inner radius of curvature, $r_2$
# - Lens: thickness, $t$
# - Lens: diameter, $D$
# - Rhabdom: distal diameter, $D_{Rh}$
# - Rhabdom: length, $l_{Rh}$
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
# - The acceptance angle of an ommatidium, $\Delta\rho$, is proscribed by diffraction effects at the lens,  $\Delta\rho_l$, and the geometry of the distal rhabdom tip and the lens, $\Delta\rho_{Rh}$ (Snyder, 1979):
# $$\Delta\rho = \sqrt{{\Delta\rho_l}^2 + {\Delta\rho_{Rh}}^2}$$
#     - $\Delta\rho_l = \lambda/D$, where we set $\lambda$ = 0.5 $\mu{m}$ (green monochromatic light)
#     - $\Delta\rho_{Rh} = D_{Rh}/f$
# - Optical sensitivity to an extended broadband-spectrum source, $S$ ($\mu{m}^2{sr}$), approximated by:  
# $$S = (\frac{\pi}{4})^2 D^2 {\Delta\rho}^2 \frac{k{l}_{Rh}}{2.3 + k{l}_{Rh}}$$
#     - See [Fredriksen and Warrant, 2008](https://dx.doi.org/10.1098%2Frsbl.2008.0467); [Kirschfeld, 1974](https://doi.org/10.1515/znc-1974-9-1023)
#     - k is the peak absorbtion (length) coefficient of the visual pigment, taken as 0.0067 $\mu{m}^-1$ (Warrant et al., 2004)
#     - Notes: facet area is $\frac{\pi{D}^2}{4}$, the fraction of incident light absorbed is $\frac{k{l}_{Rh}}{2.3 + k{l}_{Rh}}$
#
#

# +
r1 = optics_df['outer curvature']
r2 = -1 * optics_df['inner curvature']
t = optics_df['lense thickness']
D = optics_df['facet diameter (stack)']
Dr = optics_df['D rhabdom dist.']
lr = optics_df['rhabdom_len']

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
rho_rh = Dr/f # angular sens. due to geometry of rhabdom tip
# 'simple' acceptance angle formula 
rho = (rho_l ** (2.0) + rho_rh ** (2.0)) ** (0.5)

# Optical sensitivity
k = 0.0067 # peak absorbtion coefficient (wrt rhabdom length) from Warrant et al., 2004
kl = k * lr
S = ((np.pi / 4.0)**2.0) * (D ** 2.0) * (rho ** 2.0) * (kl / (2.3 + kl))

optics_df['f'] = f.astype('float') 
optics_df['f-image'] = fi.astype('float') 
optics_df['P'] = p.astype('float') 
optics_df['F-number'] = F.astype('float') 
optics_df['rho_l'] = np.degrees(rho_l.astype('float') )
optics_df['rho_rh'] = np.degrees(rho_rh.astype('float') )
optics_df['rho'] = np.degrees(rho.astype('float') )
optics_df['S'] = S.astype('float')  # steradians. convert to deg?

# +
#dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
dra_om = ['A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E7']
ndra_om = [str(o) for o in optics_df.index if o not in dra_om]
regional_summary = pd.DataFrame([optics_df.mean(), optics_df.std(ddof=0), 
                                 optics_df.loc[ndra_om].mean(), optics_df.loc[ndra_om].std(ddof=0), 
                                 optics_df.loc[dra_om].mean(), optics_df.loc[dra_om].std(ddof=0)], 
                                index=['All_mean', 'All_SD', 'NDRA_mean', 'NDRA_SD', 'DRA_mean', 'DRA_SD'])

display(regional_summary.T.round(decimals=2))


# -

terms

optics_df


