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
    (CURRENTLY EXCLUDES INTEROM)
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


def extract_connector_table(link_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract synaptic terminals from link dataframe
    TODO: split into two dataframes (one of summary, the other for cx's partner subtype breakdown)
    :param link_df:
    :return:
    """
    cx_data = dict.fromkeys(np.unique(link_df['cx_id']))
    p_counts = dict.fromkeys(np.unique(link_df['cx_id']))
    subtypes = sorted(np.unique(link_df['post_type']))

    for cx, links in link_df.groupby('cx_id'):
        cx_data[cx] = dict()
        # if something goes wrong here, li
        # presynaptic info
        cx_data[cx].update({'pre_om': np.unique(links['pre_om'])[0],
                         'pre_type': np.unique(links['pre_type'])[0],
                         'pre_neuron': np.unique(links['pre_neuron'])[0]})
        # Number of partners belonging to each subtype
        type_freq = links['post_type'].value_counts().to_dict()
        cx_data[cx].update({str(s): type_freq.get(s, 0) for s in subtypes})

        # Ommatidia of post partners (should be the same)
        partner_oms = np.unique(links['post_om'])
        partner_oms = partner_oms[partner_oms != 'UNKNOWN']
        if len(partner_oms) != 1:
            # raise Warning(f"Post-partners for connector {cx} belong to more than one ommatidia: {partner_oms}")
            cx_data[cx].update({'post_om': 'multi'})
        else:
            cx_data[cx].update({'post_om': str(partner_oms[0])})

    return pd.DataFrame(cx_data).T


def assemble_cxvectors(linkdf: pd.DataFrame, external: bool=True) -> pd.DataFrame:
    """
    assemble_cxvectors
    Get a cxvectors dataframe from linkdf. Each row is an ommatidium, columns are each 
    connection type found in linkdf. (like a bunch of flattened adj mats)
    If external = True, will count connections made between ommatidia
    (e.g. as 'eR1R4->L4' for R1R4 inputs external to the L4's home cartridge, and 'R1R4->eL4' in the home
    cartridge of the R1R4). This way, connections between ommatidia are counted twice, once for the cartridge
    receiving the input, and once for the cartridge giving the input to an external cell
    """
    # filter out orphan connections
    linkdf = linkdf.loc[((linkdf['pre_om'] != 'UNKNOWN') & (linkdf['post_om'] != 'UNKNOWN'))]
    oms = np.unique(linkdf['pre_om'])
    subtypes = np.unique([*linkdf['pre_type'], *linkdf['post_type']])
    ctypes = [f'{pre}->{post}' for pre, post in [p for p in product(subtypes, subtypes)]]
    # initialize df with all counts = 0
    df = pd.DataFrame(np.zeros((len(oms), len(ctypes)), dtype=int), index=oms, columns=ctypes)
    
    # if you want external connections to be listed in row of the pre neuron's home ommatidium, use
    # groupby 'pre_om' (R1R4->eL4). If you want them listed in the post neuron's home ommatidium, use 
    # groupby 'post_om' (eR1R4->L4)
    for om, rows in linkdf.groupby('pre_om'):
        for i, link in rows.iterrows():
            
            pre_type = link['pre_type']
            post_type = link['post_type']
            # local connections
            if link['pre_om'] == link['post_om']:
                df.loc[om, f'{pre_type}->{post_type}'] += 1
            # external connections 
            elif external:  # if pre_om != post_om
                if f'{pre_type}->e{post_type}' not in df.columns:
                    # make new ctype column when this external connection is first seen
                    df[f'{pre_type}->e{post_type}'] = np.zeros(len(oms), dtype=int)
                df.loc[om, f'{pre_type}->e{post_type}'] += 1
            else:
                continue # skip external connections if asked to
                
    return df    
                        
            
    