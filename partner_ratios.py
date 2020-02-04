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

# +
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.rc('font', size=14)

# +
data_path = '~/Data/200131_lamina/200131_linkdf.pickle'
df = pd.read_pickle(data_path)

df['post_type'].describe()
subtypes = np.unique(df['post_type'])


# +
def extract_connector_table(df, pre_type):
    ll = df.loc[df['pre_type'] == pre_type]
    cx_counts = dict()
    for this_cx, l in ll.groupby('cx_id'):
        cx_counts[this_cx] = {s: 0 for s in subtypes}
        cx_counts[this_cx].update(l['post_type'].value_counts())
    cx_df = pd.DataFrame(cx_counts).T
    return cx_df

    
def plot_cx_dist4(cx_df, pre_type, post1):
    scatter_kws= {
            'alpha': 0.05,
            's': 200,
            'marker': 'x',
    }
        
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=[40, 10], subplot_kw={'xlim': [-.5, 6], 'ylim': [-.5, 6]})
    fig.suptitle(f'{pre_type} Presynaptic Terminals (n = {len(cx_df)} terminals)')
    
    post_sts = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4', 'centri']
    post2s = [s for s in post_sts if s != post1]
    print(post2s)
    sns.regplot(cx_df[post1], cx_df[post2s[0]], ax=ax[0], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    sns.regplot(cx_df[post1], cx_df[post2s[1]], ax=ax[1], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    sns.regplot(cx_df[post1], cx_df[post2s[2]], ax=ax[2], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    sns.regplot(cx_df[post1], cx_df[post2s[3]], ax=ax[3], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    plt.show()
    
def plot_cx_dist(df, pre_type, post1, post2):
    scatter_kws= {
            'alpha': 0.05,
            's': 200,
            'marker': 'x',
    }
    pre_df = extract_connector_table(df, pre_type)    
    j = sns.jointplot(x=pre_df[post1], y=pre_df[post2], kind='kde', marginal_kws={'bw': 0.5})# xlim=[-.5, 8], ylim=[-.5, 8])
    
def plot_cx_heat(cx_df, pre_type, post1):
    scatter_kws= {
            'alpha': 0.05,
            's': 200,
            'marker': 'x',
    }
        
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=[40, 10])#, subplot_kw={'xlim': [0, 6], 'ylim': [0, 6]})
    fig.suptitle(f'{pre_type} Presynaptic Terminals (n = {len(cx_df)} terminals)')
    
    post_sts = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4', 'centri']
    post2s = [s for s in post_sts if s != post1]
    print(post2s)
    sns.regplot(cx_df[post1], cx_df[post2s[0]], ax=ax[0], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    sns.regplot(cx_df[post1], cx_df[post2s[1]], ax=ax[1], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    sns.regplot(cx_df[post1], cx_df[post2s[2]], ax=ax[2], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    sns.regplot(cx_df[post1], cx_df[post2s[3]], ax=ax[3], scatter_kws=scatter_kws)#, x_jitter=0.1, y_jitter=0.1)
    plt.show()
    
def sample_contacts(df, pre_type, n: int):
    contacts = df.loc[df['pre_type'] == pre_type]
    if n > len(contacts):
        raise Exception("Sample size more than total")
    else:
        return contacts.sample(n, replace=True)

def sample_expected_ratio(df: pd.DataFrame, pre_type: str, post_type: str, n: int):
    success = sum(sample_contacts(df, pre_type, n)['post_type'] == post_type)
    return success/float(n)

def sample_ratios(df: pd.DataFrame, pre_type, post_type, n, n_trials):
    ratios = np.zeros(n_trials, dtype=float)
    for i, val in enumerate(ratios):
        ratios[i] = sample_expected_ratio(df, pre_type, post_type, n)
    return ratios
    



# +
lmcs = ['LMC_1', 'LMC_2', 'LMC_3', 'LMC_4']
svfs = ['R1R4', 'R2R5', 'R3R6']

dist_args = {'n': 10,
            'n_trials': 1000}
fig, axes = plt.subplots(4, 3, figsize=[20, 15])
display(axes.shape)

for i, post in enumerate(lmcs):
    for ii, pre in enumerate(svfs):


        r = sample_ratios(df, pre, post, **dist_args)
        sns.distplot(r, kde=False, ax=axes[i, ii])
        axes[i, ii].set_title(f'{pre} -> {post}')

plt.show()




# +


r1r4_cx = extract_connector_table(df, 'R1R4')
r2r5_cx = extract_connector_table(df, 'R2R5')
r3r6_cx = extract_connector_table(df, 'R3R6')
#display(r2r5_cx)
plot_cx_dist(r1r4_cx, 'R1R4', 'LMC_1')
plot_cx_dist(r2r5_cx, 'R2R5', 'LMC_1')
plot_cx_dist(r3r6_cx, 'R3R6', 'LMC_1')


#r2r5_cx.groupby('LMC_2')
# -

plot_cx_dist(r1r4_cx, 'R1R4', 'LMC_2')
plot_cx_dist(r2r5_cx, 'R2R5', 'LMC_2')
plot_cx_dist(r3r6_cx, 'R3R6', 'LMC_2')

plot_cx_dist(r1r4_cx, 'R1R4', 'LMC_3')
plot_cx_dist(r2r5_cx, 'R2R5', 'LMC_3')
plot_cx_dist(r3r6_cx, 'R3R6', 'LMC_3')

centri_cx = extract_connector_table(df, 'centri')

centri_cx = extract_connector_table(df, 'centri')

centri_cx = extract_connector_table(df, 'centri')
l2_cx = extract_connector_table(df, 'LMC_2')
plot_cx_dist(r2r5_cx, 'R2R5', 'centri')


plot_cx_dist(centri_cx, 'centri', 'R7p')
plot_cx_dist(centri_cx, 'centri', 'R7')


def overview(df):
    pkws = {
        #'x_jitter': 0.5,
        #'y_jitter': 0.5,
        #'marker': '.',

        'scatter_kws': {
            'alpha': 0.1,
            's': 40,
            'marker': 's',
            #'xlim': [0, 8],
            #'ylim': [0, 8],
        },
    }
    '''
    grid = {'xlim': (0, 8),
           'ylim': (0, 8),
           'sharex': True,
           'sharey': True}
    '''
    pp = sns.pairplot(df, kind='reg', plot_kws=pkws)


overview(centri_cx.filter(items=['LMC_1', 'LMC_2', 'R7', 'R7p']))






