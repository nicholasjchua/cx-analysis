import numpy as np
import pandas as pd


def extract_connector_table(link_df) -> pd.DataFrame:
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


