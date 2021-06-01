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

import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# +
data_path = '~/Data/200115_lamina/200115_linkdf.pickle'
df = pd.read_pickle(data_path)

df['post_type'].describe()

# +
cx_data = dict.fromkeys(df[])

for pre_type, i in df.groupby('pre_type'):
    this_cx, ii in df.groupby()
# -


