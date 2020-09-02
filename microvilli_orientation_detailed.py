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
# - Visualize Anastasia's microvilli orientation data for R7 and R7'
# - NOTE: this was written for the data AM sent in April 2020: this spreadsheet only included R7 and R7', and lacked the z index in which each measurement was taken
# - Replaced by notebook written for the new excel file AM sent in August 2020 that includes measurements for all photoreceptors along with the z index each measurement was taken

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# +
xl_dir = '~/Data/200824_microvilli.xlsx'

full_df = pd.read_excel(xl_dir, sheet_name='microvilli angle', header=None, index_col=None)

# +
twist = []
# TODO: extract other data
om_df = []
cell_df = []

for i in range(0, 29):  # each table in the sheet corresponds to an ommatidia
    # excel has 16 rows of each ommatidia, but only 13 have data
    this_range = full_df.loc[i * 16: i * 16 + 13].reset_index(drop=True)
    this_om = this_range.iloc[0, 0]
    
    if this_om == 'C5':  # C5 has 11 rows 
        this_range = full_df.loc[i * 16: i * 16 + 14].reset_index(drop=True)
        rows = 11
    else:
        rows = 10
    z_st_cols = this_range.iloc[3, 13:]

    assert(len(this_om) == 2)
    assert(len(z_st_cols) == 9)
    
    for ii, this_st in this_range.iloc[0, 4:13].items():  # subtypes 
        #display(i, subtype)
        if '(' in this_st:  # some have the old subtype nomenclature in ()
            this_st = this_st.split('(')[0]
        this_st = this_st.strip().upper()
        # Should R7' be changed to R7p? 

        z_col = z_st_cols[z_st_cols == this_st.lower()].index[0]
        twist.append(pd.DataFrame({'om': [this_om]*rows, 
                                     'subtype': [this_st]*rows, 
                                     'z-index': this_range.iloc[4:, z_col], 
                                     'angle': this_range.iloc[4:, ii]}))
        
twist_df = pd.concat(twist, ignore_index=True)
twist_df = twist_df.astype({'z-index': float,
                           'angle': float})
#display(twist_df.loc[twist_df.om == 'C5'], twist_df.tail(30))
# -


all_om = sorted(twist_df['om'].unique())
dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
ndra_om = [str(o) for o in all_om if o not in dra_om]
subtypes = sorted(twist_df['subtype'].unique())

# +
fig, ax = plt.subplots(15, 2, figsize=[25, 80], sharey=True)
axes = ax.flatten()
i = 0
for om, rows in twist_df.groupby('om'):
    
    #display(rows)
    sns.lineplot(x='z-index', y='angle', hue='subtype', data=rows, markers=True, ax=axes[i])
    axes[i].set_title(f"Ommatidium: {om}")
    i += 1
    
axes[-1].remove()
fig.savefig("/mnt/home/nchua/Dropbox/200902_microvilli_raw_all.pdf", bbox_inches='tight')

# +
fig, ax = plt.subplots(15, 2, figsize=[25, 80], sharey=True)
axes = ax.flatten()
i = 0
for om, rows in twist_df.groupby('om'):
    rows = rows.loc[[i for i, v in rows['subtype'].items() if int(v[1]) > 6]] 
    #display(rows)
    sns.lineplot(x='z-index', y='angle', hue='subtype', data=rows, markers=True, ax=axes[i])
    axes[i].set_title(f"Ommatidium: {om}")
    i += 1
    
axes[-1].remove()
fig.savefig("/mnt/home/nchua/Dropbox/200902_microvilli_raw_lvf.pdf", bbox_inches='tight')

# +
fig, ax = plt.subplots(15, 2, figsize=[25, 80], sharey=True)
axes = ax.flatten()
i = 0
for om, rows in twist_df.groupby('om'):
    rows = rows.loc[[i for i, v in rows['subtype'].items() if int(v[1]) < 7]] 
    #display(rows)
    sns.lineplot(x='z-index', y='angle', hue='subtype', data=rows, markers=True, ax=axes[i])
    axes[i].set_title(f"Ommatidium: {om}")
    i += 1
    
axes[-1].remove()
fig.savefig("/mnt/home/nchua/Dropbox/200902_microvilli_raw_svf.pdf", bbox_inches='tight')
# -

cols = pd.MultiIndex.from_product([['SD of angle', 'max displacement', ]])
twist_results = pd.DataFrame(columns = pd.Multi)

# +
r7_sd = r7.std()
r7p_sd = r7p.std()

display(r7_sd.loc[[*dra_om, *ndra_om]])
display(r7p_sd.loc[[*dra_om, *ndra_om]])

display(f"DRA mean standard deviation - R7 = {r7_sd.loc[dra_om].mean(): .2f} degrees, R7' = {r7p_sd.loc[dra_om].mean(): .2f}")
display(f"NON-DRA mean standard deviation - R7 = {r7_sd.loc[ndra_om].mean(): .2f} degrees, R7' = {r7p_sd.loc[ndra_om].mean(): .2f}")
display(f"All om mean standard deviation - R7 = {r7_sd.mean(): .2f} degrees, R7' = {r7p_sd.mean(): .2f}")
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

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[10, 12])

ax[0, 0].plot(r7.filter(items=ndra_om))
ax[0, 0].legend(r7.filter(items=ndra_om).columns)
ax[0, 1].plot(r7p.filter(items=ndra_om))
ax[0, 1].legend(r7p.filter(items=ndra_om).columns)

ax[1, 0].plot(r7.filter(items=dra_om))
ax[1, 0].legend(r7.filter(items=dra_om).columns)
ax[1, 1].plot(r7p.filter(items=dra_om))
ax[1, 1].legend(r7p.filter(items=dra_om).columns)
plt.show()

# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[15, 12])

fig.suptitle('Absolute microvilli twist relative to first (most distal) measurement')
ax[0, 0].plot(r7_dev.filter(items=ndra_om))
ax[0, 0].legend(r7_dev.filter(items=ndra_om).columns)
ax[0, 0].set_title('R7 Non-DRA')
ax[0, 1].plot(r7p_dev.filter(items=ndra_om))
ax[0, 1].legend(r7p_dev.filter(items=ndra_om).columns)
ax[0, 1].set_title("R7' Non-DRA")

ax[1, 0].plot(r7_dev.filter(items=dra_om))
ax[1, 0].legend(r7_dev.filter(items=dra_om).columns)
ax[1, 0].set_title('R7 DRA')
ax[1, 1].plot(r7p_dev.filter(items=dra_om))
ax[1, 1].legend(r7p_dev.filter(items=dra_om).columns)
ax[1, 1].set_title("R7' DRA")

ax[1, 0].set_xlabel('Length interval')
ax[1, 1].set_xlabel('Length interval')

ax[0, 0].set_ylabel('Degrees')
ax[1, 0].set_ylabel('Degrees')

plt.show()
# -

fig, ax = plt.subplots(29, 2, sharex=True, sharey=True, figsize=[15, 100])
for i, om in enumerate(sorted(r7_dev.columns)):
#     if om in dra_om:
#         c = 'darkviolet'
#     else:
#         c = 'k'
    c='k'
    
    ax[i, 0].plot(r7_dev[om], c=c, label=f"{om} R7")
    ax[i, 0].plot(r7_dev.mean(axis=1), ls='--', label="Average R7")
    ax[i, 0].set_title(f"{om} R7")
    ax[i, 0].legend()
    ax[i, 1].plot(r7p_dev[om], c=c, label=f"{om} R7'")
    ax[i, 1].plot(r7p_dev.mean(axis=1), ls='--', label="Average R7'")
    ax[i, 1].set_title(f"{om} R7'")
    ax[i, 1].legend()


fig, ax = plt.subplots(29, sharex=True, figsize=[8, 120])
for i, om in enumerate(sorted(r7_dev.columns)):
    if om in dra_om:
        c = 'darkviolet'
    else:
        c = 'k'
    
    ax[i].plot(r7[om], c='b', label=f"{om} R7")
    #ax[i].plot(r7.mean(axis=1), ls='--', label="Average R7")
    ax[i].plot(r7p[om], c='r', label=f"{om} R7'")
    #ax[i].plot(r7p.mean(axis=1), ls='--', label="Average R7'")
    ax[i].set_title(f"om {om} - Anastasia (raw angles)")
    ax[i].legend()


# +
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=[15, 12])

ax[0, 0].plot(r7_dev.filter(items=ndra_om).mean(axis=1), c='k')
ax[0, 0].set_title("Average R7 Non-DRA")
ax[0, 1].plot(r7p_dev.filter(items=ndra_om).mean(axis=1), c='k')
ax[0, 1].set_title("Average R7' Non-DRA")

ax[0, 0].plot(r7_dev.mean(axis=1), ls='--', c='c')
ax[0, 1].plot(r7p_dev.mean(axis=1), ls='--', c='c')

ax[1, 0].plot(r7_dev.filter(items=dra_om).mean(axis=1), c='darkviolet')
ax[1, 0].set_title("Average R7 DRA")
ax[1, 1].plot(r7p_dev.filter(items=dra_om).mean(axis=1), c='darkviolet')
ax[1, 1].set_title("Average R7' DRA")

ax[1, 0].plot(r7_dev.mean(axis=1), ls='--', c='c')
ax[1, 1].plot(r7p_dev.mean(axis=1), ls='--', c='c')

plt.show()
# -

r7_total = np.abs(r7.loc[1] - r7.loc[10])
r7p_total = np.abs(r7p.loc[1] - r7p.loc[10])

r7_total


