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

# # Microvilli orientation
# - Visualize Anastasia's microvilli orientation data for R7 and R7'
# - NOTE: this was written for the data AM sent in April 2020: this spreadsheet only included R7 and R7', and lacked the z index in which each measurement was taken
# - Replaced by notebook written for the new excel file AM sent in August 2020 that includes measurements for all photoreceptors along with the z index each measurement was taken

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# +


dra_om = ['A4', 'A5', 'B5', 'B6', 'C5', 'C6', 'D6', 'D7', 'E6', 'E7']
ndra_om = [str(o) for o in r7.columns if o not in dra_om]

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


