import pandas as pd
import numpy as np
from itertools import product
from typing import Tuple

#import src.connectome as cxt
from src.skeleton import Skeleton
"""
dataframe_org.py
Methods to extract and save summary data from a Connectome
"""

def assemble_linkdf(C) -> pd.DataFrame:
    """
    link_df contains a row for each synaptic contact made between neurons in the Connectome
    :param C: Connectome
    :return link_df: DataFrame
    """
    df_rows = []
    skel_data = C.skel_data

    for pre_id, pre_sk in skel_data.items():
        assert (type(pre_sk) is Skeleton)
        out_links = pre_sk.out_links  # list containing a Dict for each synaptic link
        for l in out_links:
            post_id = l.get('post_skel')
            post_sk = C.skel_data.get(post_id, None)

            if post_sk is None:  # unidentified neurites (aka fragments)
                post_name = str(post_id)
                post_type = 'UNKNOWN'
                post_om = 'UNKNOWN'
            else:
                post_name = post_sk.name
                post_type = post_sk.subtype
                post_om = post_sk.group
            # TODO pd.Category this?
            df_rows.append({'pre_neuron': pre_sk.name,
                            'pre_type': pre_sk.subtype,
                            'pre_om': pre_sk.group,
                            'pre_skel': pre_id,
                            'post_neuron': post_name,
                            'post_type': post_type,
                            'post_om': post_om,
                            'post_skel': post_id,
                            'link_id': l.get('link_id'),
                            'cx_id': l.get('cx_id')})

    df = pd.DataFrame(data=df_rows, columns=['link_id', 'cx_id',
                                             'pre_neuron', 'pre_om', 'pre_type', 'pre_skel',
                                             'post_neuron', 'post_om', 'post_type', 'post_skel'])
    return df


def assemble_cxdf(C, linkdf) -> Tuple:
    """
    Longform DataFrame that has a row for each group of neurons/each connection type
    requires link_df
    :param C: Connectome
    :return cxdf, inter, unknowns:
    """
    cx_types = [f"{pre}->{post}"
                for pre, post in product(C.cfg.subtypes, C.cfg.subtypes)]

    om_list = sorted([str(k) for k in C.grouping.keys()])

    counts = np.zeros((len(om_list), len(cx_types)), dtype=int)
    inter = []
    unknowns = []

    for ind, row in linkdf.iterrows():
        this_pre, this_post = (row['pre_type'], row['post_type'])
        if this_pre.upper() == 'UNKNOWN' or this_post.upper() == 'UNKNOWN':
            unknowns.append(row)
        elif row['pre_om'] != row['post_om']:
            inter.append(row)
        else:
            j = cx_types.index(f"{this_pre}->{this_post}")
            i = om_list.index(row['pre_om'])
            counts[i, j] += 1

    om_order = np.array([[om] * len(cx_types) for om in om_list]).reshape(-1)
    cx_order = np.tile(cx_types, len(om_list))
    print(f"om_order: {om_order.shape}, cx_order: {cx_order.shape}")
    pre_order, post_order = np.array([cx.split("->") for cx in cx_order]).T
    print(f"pre_order: {pre_order.shape}, post_order: {post_order.shape}")

    df = pd.DataFrame({'om': pd.Categorical(om_order),
                       'cx_type': pd.Categorical(cx_order),
                       'pre_type': pd.Categorical(pre_order),
                       'post_type': pd.Categorical(post_order),
                       'n_connect': np.ravel(counts)})
    df.loc[df['n_connect'] < 0, 'n_connect'] = np.nan

    return df, inter, unknowns
