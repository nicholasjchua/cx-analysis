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

# # Microvilli orientation
# - Visualize Anastasia's microvilli orientation data (more detailed excel sent in Aug 2020)
# - Determine if microvilli twist is more or less constant in the DRA
# - Determine if candidates for polarization detection have more or less orthogonally oriented microvilli in the DRA

# +
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

from vis.hex_lattice import hexplot
from vis.fig_tools import linear_cmap
# -

xl_dir = '~/Data/200824_microvilli.xlsx'
full_df = pd.read_excel(xl_dir, sheet_name='microvilli angle', header=None, index_col=None)
#display(full_df)

# +
# The excel file contains a table for each ommatidium, the formatting is very consistent
twist = []
# TODO: extract other data
om_df = []
cell_df = []

for i in range(0, 29):  # ommatidia
    # excel has 16 rows for each ommatidia, but only 13 have data
    this_range = full_df.loc[i * 16: i * 16 + 13].reset_index(drop=True)
    this_om = this_range.iloc[0, 0]
    if this_om == 'C5':  # C5 has an extra row that replaces one of the blank ones at the end
        this_range = full_df.loc[i * 16: i * 16 + 14].reset_index(drop=True)
        rows = 11
    else:
        rows = 10
    # these cells contain the z-index each measurement was taken
    z_st_cols = this_range.iloc[3, 13:]  

    assert(len(this_om) == 2)  # check om name
    assert(len(z_st_cols) == 9)  # check that there are 9 photoreceptors
    
    for ii, this_st in this_range.iloc[0, 4:13].items():  # subtypes 
        if '(' in this_st:  # some have the old subtype nomenclature in ()
            this_st = this_st.split('(')[0]
        this_st = this_st.strip().upper()
        # Should R7' be changed to R7p? 
        z_col = z_st_cols[z_st_cols == this_st.lower()].index[0]
        
        # Add negative sign to indices of stacks that start proximal so the direction is distal->proximal
        if this_range.iloc[2, 4] < this_range.iloc[3, 4]: # start/end index for R1
            z_inds = this_range.iloc[4:, z_col]
        else:
            z_inds = this_range.iloc[4:, z_col] * -1.0
        
        twist.append(pd.DataFrame({'om': [this_om]*rows, 
                                     'subtype': [this_st]*rows, 
                                     'z-index': z_inds, 
                                     'angle': this_range.iloc[4:, ii]}))
        
twist_df = pd.concat(twist, ignore_index=True)
twist_df = twist_df.astype({'z-index': float, # because there are NaNs
                           'angle': float})
#display((twist_df.loc[1900, 'z-index']))
# -


all_om = sorted(twist_df['om'].unique())
dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']  # as defined by connectivity clustering
ndra_om = [str(o) for o in all_om if o not in dra_om]
subtypes = sorted(twist_df['subtype'].unique())

# ## Compute angular differences between measurements
# - Data is collected on ImageJ, which records the smaller (< 180) angle between two lines drawn by the user
# - Anastasia added a negative sign to angles measured clockwise from the reference angle
# - Because of this, we cannot use angle1 - angle2 as the difference
# - First, the angle1 - angle2 differences are converted into minimal positive (counter-clockwise) angles by taking % 360 , "_diff"
# - Secondly, for counter clockwise angles more than +180, we take the smaller angle in the CCW direction by subtracting 360, "diff". This retains the sign convention of negative -> CW twist 

# +


for i, om_rows in twist_df.groupby('om'):
    for ii, rows in om_rows.groupby('subtype'):
        n = 0
        for iii, row in rows.sort_values('z-index').iterrows():
            twist_df.loc[iii, 'n'] = int(n)
            if n == 0:  # first measurement has diff of 0
                twist_df.loc[iii, '_diff'] = np.nan
                twist_df.loc[iii, 'diff'] = np.nan
                twist_df.loc[iii, 'cumulative'] = 0.0
                twist_df.loc[iii, '_angle'] = twist_df.loc[iii, 'angle']
                twist_df.loc[iii, 'interval_len'] = np.nan # 0.0
                twist_df.loc[iii, 'interval_z'] = np.nan
                previous = (iii, twist_df.loc[iii, 'angle'])
            elif math.isnan(twist_df.loc[iii, 'angle']):
                #print(f"NaN found for {i}_{ii} measurement: {n}")
                twist_df.loc[iii, '_diff'] = np.nan
                twist_df.loc[iii, 'diff'] = np.nan
                twist_df.loc[iii, 'cumulative'] = np.nan
                twist_df.loc[iii, '_angle'] = np.nan
                twist_df.loc[iii, 'interval_len'] = np.nan
                twist_df.loc[iii, 'interval_z'] = np.nan
                # previous = measurement before the NaN, though n still increases by 1
            else:
                twist_df.loc[iii, '_diff'] = (twist_df.loc[iii, 'angle'] - previous[1]) % 360.0
                twist_df.loc[iii, 'diff'] = twist_df.loc[iii, '_diff'] - 360.0 * (twist_df.loc[iii, '_diff'] > 180.0)
                twist_df.loc[iii, 'cumulative'] = twist_df.loc[previous[0], 'cumulative'] + twist_df.loc[iii, 'diff']
                twist_df.loc[iii, '_angle'] = twist_df.loc[previous[0], '_angle'] + twist_df.loc[iii, 'diff']
                twist_df.loc[iii, 'interval_len'] = (twist_df.loc[iii, 'z-index'] - twist_df.loc[previous[0], 'z-index']) * 8.0 / 1000.0
                twist_df.loc[iii, 'interval_z'] = twist_df.loc[iii, 'z-index'] - twist_df.loc[previous[0], 'z-index']
                # because 1 px = 8/1000 microns
                previous = (iii, twist_df.loc[iii, 'angle'])
            n += 1
twist_df['diff_per_micron'] = twist_df['diff'] / twist_df['interval_len']
#twist_df['_angle'] = twist_df['_angle'] % 180.0
            
#twist_df['diff'] = twist_df['_diff'] - 360*(twist_df['_diff']>180)

# display(twist_df.loc[(twist_df['om']=='A0') & (twist_df['subtype'] == 'R5'), 
#                     ('om', 'subtype', 'z-index', 'angle', '_diff', 'diff', 'cumulative')].sort_values('z-index'))

# Example D6_R1 shows how NaNs are used
display(twist_df.loc[(twist_df['om']=='D6') & (twist_df['subtype'] == 'R1'), :].sort_values('z-index'))
display(twist_df.loc[(twist_df['om']=='A4') & (twist_df['subtype'] == 'R8'), :].sort_values('z-index'))
display(twist_df.loc[(twist_df['om']=='B1') & (twist_df['subtype'] == 'R7'), :].sort_values('z-index'))

display(twist_df.loc[:, 'cumulative'].max())
display(twist_df.loc[:, 'diff'].max())
# -

# ## 

#twist_df['cosine'] = np.cos(twist_df['angle']*(np.pi/180.0))
twist_df['cosine_sq'] = np.cos(twist_df['angle']*(np.pi/180.0)) ** 2
twist_df['CCW_angle'] = [(360.0 + x) if x < 0 else x for x in twist_df['angle']]

# +
# fig, ax = plt.subplots(15, 2, figsize=[25, 80], sharey=True)
# axes = ax.flatten()
# i = 0
# for om, rows in twist_df.groupby('om'):
    
#     #display(rows)
#     sns.lineplot(x='z-index', y='cosine_sq', hue='subtype', data=rows, markers=True, ax=axes[i])
#     axes[i].set_title(f"Ommatidium: {om}")
#     i += 1
    
# axes[-1].remove()
# #fig.savefig("/mnt/home/nchua/Dropbox/200902_microvilli_raw_all.pdf", bbox_inches='tight')

# +
fig, ax = plt.subplots(15, 2, figsize=[25, 80], sharey=True)
axes = ax.flatten()
i = 0
for om, rows in twist_df.groupby('om'):
    rows = rows.loc[[i for i, v in rows['subtype'].items() if int(v[1]) > 6]] 
    #display(rows)
    sns.lineplot(x='z-index', y='cumulative', hue='subtype', data=rows, markers=True, ax=axes[i])
    axes[i].set_title(f"Ommatidium: {om}")
    axes[i].set_ylabel("Cumulative angular displacement\n(degrees)")
    i += 1
    
axes[-1].remove()
#fig.savefig("/mnt/home/nchua/Dropbox/200902_microvilli_raw_lvf.pdf", bbox_inches='tight')
# + {}
fig, ax = plt.subplots(15, 2, figsize=[25, 80], sharey=True)
axes = ax.flatten()
i = 0
for om, rows in twist_df.groupby('om'):
    rows = rows.loc[[i for i, v in rows['subtype'].items() if int(v[1]) == 7]] 
    #display(rows)
    sns.lineplot(x='z-index', y='_angle', hue='subtype', data=rows, markers=True, ax=axes[i])
    #sns.lineplot(x='z-index', y=)
    axes[i].set_title(f"Ommatidium: {om}")
    axes[i].set_ylabel("Cumulative angular displacement\n(degrees)")
    i += 1
    
axes[-1].remove()
#fig.savefig("/mnt/home/nchua/Dropbox/200902_microvilli_raw_lvf.pdf", bbox_inches='tight')
# + {}
abs_diff = twist_df.copy()
#abs_diff['diff'] = np.abs(abs_diff['diff'])
abs_diff['abs_diff_per_micron'] = np.abs(abs_diff['diff_per_micron'])
abs_diff['previous_z'] = abs_diff['z-index'] - abs_diff['interval_z']

display(abs_diff['abs_diff_per_micron'].median())
# -


abs_diff.sort_values('abs_diff_per_micron', ascending=False).loc[abs_diff['subtype']=='R7', 
                    ['om', 'subtype','previous_z', 'z-index', 'diff', 'diff_per_micron']].head(20)


abs_diff.sort_values('abs_diff_per_micron', ascending=False).loc[abs_diff['subtype']=="R7'", 
                    ['om', 'subtype','previous_z', 'z-index', 'diff', 'diff_per_micron']].head(20)


# +
display(twist_df.loc[twist_df['diff_per_micron'].abs() == twist_df['diff_per_micron'].abs().max()])

fig, ax = plt.subplots(1)
ax = sns.distplot(twist_df['diff_per_micron'].abs().dropna(), kde=False)
ax.set_xlabel('Angular displacement between measurements\n(degrees/micron)')
ax.set_ylabel('Frequency observed')


# +
# fig, ax = plt.subplots(15, 2, figsize=[25, 80], sharey=True)
# axes = ax.flatten()
# i = 0
# for om, rows in twist_df.groupby('om'):
#     rows = rows.loc[[i for i, v in rows['subtype'].items() if int(v[1]) < 7]] 
#     #display(rows)
#     sns.lineplot(x='z-index', y='cosine_sq', hue='subtype', data=rows, markers=True, ax=axes[i])
#     axes[i].set_title(f"Ommatidium: {om}")
#     i += 1
    
# axes[-1].remove()
# #fig.savefig("/mnt/home/nchua/Dropbox/200902_microvilli_raw_svf.pdf", bbox_inches='tight')

# +
cols = pd.MultiIndex.from_product([['mean_diff', 'mean_diff_per_micron', 'SD_diff', 'mean_cosine_sq', 'r_mean_cosine_sq', 'SD_cosine_sq', 'max_displacement', 'SD_displacement', 'length', 'n_measure'], subtypes], names=['measure', 'subtype'])
twist_results = pd.DataFrame(columns = cols, index=all_om)

rh_length = pd.Series(index=all_om, dtype=float)

display(twist_df.loc[[bool([twist_df['om'] == 'A1']) & bool([twist_df['subtype'] == 'R7'])], 'cosine_sq'].mean())

for this_st, st_rows in twist_df.groupby('subtype'):
    for this_om, rows in st_rows.groupby('om'):
        # TODO: get rid of first difference when calculating mean? 
        
        #twist_results.loc[this_om, ('length', this_st)] = (rows['z-index'].max() - rows['z-index'].min()) * 8.0 /1000.0 # 1 px = 8/1000 microns
        
        twist_results.loc[this_om, ('mean_diff', this_st)] = rows['diff'].mean()
        twist_results.loc[this_om, ('SD_diff', this_st)] = rows['diff'].std()
        twist_results.loc[this_om, ('mean_diff_per_micron', this_st)] =  rows['diff_per_micron'].mean()
        
        twist_results.loc[this_om, ('mean_cosine_sq', this_st)] = rows['cosine_sq'].mean()
        
        twist_results.loc[this_om, ('SD_cosine_sq', this_st)] = rows['cosine_sq'].std()
        
        twist_results.loc[this_om, ('max_displacement', this_st)] = rows['cumulative'].max()
        twist_results.loc[this_om, ('SD_displacement', this_st)] = rows['cumulative'].std()
        #twist_results.loc[this_om, ('max_displacement_per_micron', this_st)] = rows['cumulative'].max() / twist_results.loc[this_om, ('length', this_st)]
        

#twist_results.loc[:, ('r_mean_cosine_sq')] = twist_results.loc[:, ('mean_cosine_sq', 'R7')] + 
#twist_results.loc[:, 'mean_cosine_sq']
#- twist_results.loc[:, ('mean_cosine_sq', 'R7')]
#display(twist_results.loc[:, ('mean_cosine_sq', 'R7')])
#display(twist_results.loc[:, ['mean_cosine_sq', 'r_mean_cosine_sq']])
# display(twist_results['length'])
# display(twist_results['mean_diff_per_micron'])

# +
fig, axes = plt.subplots(3, 3, figsize=[35, 35])
# cm = linear_cmap(n_vals=100, max_colour=(1.0, 1.0, 1.0), min_colour='r')
cm = linear_cmap(n_vals=100, max_colour='b')
overall_max = abs(twist_results.loc[:, 'mean_diff']).max().max()

for ax, this_st in zip(axes.flatten(), subtypes):
    node_data = dict.fromkeys(all_om)
    this_max = abs(twist_results.loc[:, ('mean_diff', this_st)]).max()
    for om in all_om:
        x = abs(twist_results.loc[om, ('mean_diff', this_st)])
        node_data[om] = {'label': f"{x: .2f}",
                        'colour': cm(x/overall_max)}
    ax.set_title(f"{this_st} Mean angular displacement/micron")
    hexplot(node_data=node_data, ax=ax)

# +
fig, axes = plt.subplots(3, 3, figsize=[35, 35])
# cm = linear_cmap(n_vals=100, max_colour=(1.0, 1.0, 1.0), min_colour='r')
cm = linear_cmap(n_vals=100, max_colour='b')
overall_max = abs(twist_results.loc[:, 'mean_diff_per_micron']).max().max()

for ax, this_st in zip(axes.flatten(), subtypes):
    node_data = dict.fromkeys(all_om)
    this_max = abs(twist_results.loc[:, ('mean_diff_per_micron', this_st)]).max()
    for om in all_om:
        x = abs(twist_results.loc[om, ('mean_diff_per_micron', this_st)])
        node_data[om] = {'label': f"{x: .2f}",
                        'colour': cm(x/overall_max)}
    ax.set_title(f"{this_st} Mean angular displacement/micron")
    hexplot(node_data=node_data, ax=ax)

# +
fig, axes = plt.subplots(3, 3, figsize=[35, 35])
# cm = linear_cmap(n_vals=100, max_colour=(1.0, 1.0, 1.0), min_colour='r')
cm = linear_cmap(n_vals=100, max_colour='b')
overall_max = abs(twist_results.loc[:, 'SD_displacement']).max().max()

for ax, this_st in zip(axes.flatten(), subtypes):
    node_data = dict.fromkeys(all_om)
    this_max = abs(twist_results.loc[:, ('SD_displacement', this_st)]).max()
    for om in all_om:
        x = abs(twist_results.loc[om, ('SD_displacement', this_st)])
        node_data[om] = {'label': f"{x: .2f}",
                        'colour': cm(x/overall_max)}
    ax.set_title(f"{this_st} angular SD")
    hexplot(node_data=node_data, ax=ax)

# +


fig, axes = plt.subplots(3, 3, figsize=[35, 35])
# cm = linear_cmap(n_vals=100, max_colour=(1.0, 1.0, 1.0), min_colour='r')
cm = linear_cmap(n_vals=100, max_colour='r')

for ax, this_st in zip(axes.flatten(), subtypes):
    node_data = dict.fromkeys(all_om)
    this_max = twist_results.loc[:, ('mean_cosine_sq', this_st)].max()
    for om in all_om:
        data = twist_results.loc[om, ('mean_cosine_sq', this_st)]
        node_data[om] = {'label': f"{data: .2f}",
                        'colour': cm(data)}
    ax.set_title(f"{this_st} Mean of cosine squared angle")
    hexplot(node_data=node_data, ax=ax)
# -

s, p, = mannwhitneyu(r7_sd[dra_om], r7_sd[ndra_om], alternative='less')
print("###### RESULTS ######")
print(f"Test statistic: {s}, p-value: {p: .6f}")
if p > 0.01:
    print("Fail to reject null")
else:
    print("Reject null: DRA R7 receive more inputs")

s, p, = mannwhitneyu(r7p_sd[dra_om], r7p_sd[ndra_om], alternative='less')
print("###### RESULTS ######")
print(f"Test statistic: {s}, p-value: {p: .6f}")
if p > 0.01:
    print("Fail to reject null")
else:
    print("Reject null: DRA R7 receive more inputs")


