import argparse
import os, sys
import glob
from typing import Dict, List, Tuple
import itertools
from config.cfg_parser import parse_config_file

from numpy.matlib import repmat
from tqdm import tqdm

"""
fetch_lamina_connectome.py
Fetch and organise data from CATMAID
"""

def main():

    cfg = parse_config_file() # see config/default_cfg.json for defaults

    # Predefined neuron categories
    subtypes = cfg['subtypes']
    cx_types = [(u, v) for u, v in itertools.permutations(subtypes, 2)]  # all possible pre -> post subtype pairs

    neuron_data, category_map, excluded_terminals = gather_data(cfg)
    adj_mat, synapse_data, skel_ref = connectivity_data(neuron_data, category_map, cfg,
                                                        filter_connectors=excluded_terminals)


def gather_data(cfg: Dict) -> Tuple:
    """
    Get all neuron data in a hash indexed by skel_id. Also generates a hash of skel_ids belonging to the
    :param token: User's Catmaid access token
    :param p_id: Project ID
    :param annot: str/id of the annotation associated with the skeletons to be analyzed
    :param param: Analysis parameters, uses 'restrict_tag' and 'in_medulla' to restrict analysis to lamina only
    :returns neuron_data: Dict hashed by skel_id containing each neuron's annotation and partner data
    :returns ref:
    """
    # Catmaid Access
    token = cfg['user_token']
    p_id = cfg['project_id']
    annot = cfg['annot']

    skel_ids, neuron_names = skels_in_annot(token, p_id, annot)
    print(f"Found {len(skel_ids)} skeletons annotated with {annot}")

    neuron_data = {s: {"name": n,  # catmaid neuron name
                       "group": n[2:4],  # which ommatidia this is from, from neuron_name, not 'ommatidia_XY' annotation
                       "categories": [],  # neuron subtype
                       "partner_data": [],  # Outgoing connections, hashed by post-synaptic skel_id
                       "r_nodes": []}  # List of nodes that are in the restricted zone (beyond lamina in this case)
                   for s, n in zip(skel_ids, neuron_names)}

    r_connectors = set()  # will output a list of connector ids in the restriction zone

    for i, this_skel in enumerate(skel_ids):
        this_class = assign_category(token, p_id, this_skel, params['subtypes'], ignore_classless=True)
        neuron_data[this_skel]["categories"] = [this_class]
        # The traversal to search for restricted nodes is pretty intensive, only check subtypes in 'in_medulla'
        if param['restrict_tag'] is not None and this_class in param['in_medulla']:
            r_nodes = nodes_between_tags(token, p_id, this_skel, param['restrict_tag'], invert=True)
            neuron_data[this_skel]['r_nodes'] = r_nodes

        partner_data, these_r_connectors = get_postsynaptic_links(token, p_id, this_skel,
                                                                  r_nodes=neuron_data[this_skel]['r_nodes'])
        r_connectors.update(these_r_connectors)
        neuron_data[this_skel]["partner_data"] = partner_data

    link_df = link_data(token, p_id, neuron_data, params['subtypes'], r_connectors)
    print(link_df)

    cat2ids = dict()
    cat2ids["om"] = hash_by_om(neuron_data)
    cat2ids["subtype"] = hash_by_category(neuron_data)

    return neuron_data, cat2ids, list(r_connectors)

def connectivity_data(neuron_data: Dict, cat2ids: Dict, params: Dict, filter_connectors: List=[],
                      syn_adj_mat: bool=False) -> Tuple:
    """
    Extract presynaptic connection data (aka 'links') from neuron_data and sort it into an adjacency matrix

    TODO: Split this up into 'extract_links' and 'make_adj_mat'?
    :param neuron_data: Dict hashed by skel_id containing each neuron's annotation and partner data
    :param cat2ids: Dict hashed by our two categorical fields, 'om' and 'subtype'. Points to lists of skeleton ids
    :param params: Dict Stores analysis conditions
    :param filter_connectors: List of connector_ids to be filtered out from the dataset
    :param syn_adj_mat: Will return an additional adjacency matrix, except counts are expressed per synapse (i.e. each
    postsynaptic neuron can only make one contact with each synapse)
    """
    ref_mat = get_ref_mat(params['subtypes'], params['expected_n'], cat2ids)  # each skeleton's position on the adj_mat
    om_order = sorted(cat2ids["om"].keys())
    adj_mat = np.zeros((len(om_order), ref_mat.shape[1], ref_mat.shape[1], 2), dtype=int)
    output_data = dict()

    for i, om in enumerate(om_order):
        for j, pre_skel in enumerate(ref_mat[i]):
            if pre_skel == '-1':  # cartridges with missing neurons coded with -1 (only allowed for L4)
                print(f'PRESKEL is -1')
                adj_mat[i, j, :, 0] = -1
                adj_mat[i, j, :, 1] = -1
                continue
            else:
                # analyze the partner data for each skeleton in our adjacency matrix
                print(f'PRESKEL:   {pre_skel}')
                output_data[pre_skel] = analyze_synapses(neuron_data, pre_skel, cat2ids["om"],
                                                         params['subtypes'], filter_connectors)
            for k, post_skel in enumerate(ref_mat[i]):
                if post_skel == '-1':
                    adj_mat[i, j, k, 0] = -1
                    adj_mat[i, j, k, 1] = -1
                else:
                    tmp = neuron_data[pre_skel]["partner_data"].get(post_skel, [])
                    these_links = [link[1] for link in tmp if link[0] not in filter_connectors]

                    adj_mat[i, j, k, 0] = len(these_links)
                    adj_mat[i, j, k, 1] = len(set([data[0] for data in these_links]))

    if syn_adj_mat:
        return adj_mat[..., 0], adj_mat[..., 1], ref_mat, output_data
    else:  # only return the raw connection counts
        return adj_mat[..., 0], ref_mat, output_data

# DATA ORGANISATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_category(token: str, p_id: int, skel_id: str, cat_list: List, ignore_classless: bool=False) -> str:
    """
    Searches through the annotations of a skeleton to find a match with one of the categories in cat_list
    Raises an exception if multiple matches are found (i.e. neuron can only belong to one of the categories)
    Can ignore instances where none of the categories are found
    :param token: str, Catmaid API token
    :param p_id: int, Project ID
    :param skel_id: str, Skeleton ID
    :param cat_list: List, Contains str of annotations you want to use as categories
    :param ignore_classless: bool, if False (default), will raise an Exception if a neuron doesn't have any annotions
    in cat_list. If True, returns None instead
    :return: str, the skeleton's category, None if no category was found
    """
    categories = set(cat_list)
    annotations = set(annot_in_skel(token, p_id, skel_id))
    matches = list(categories & annotations)
    if len(matches) > 1:
        raise Exception(f"Skeleton {skel_id} can be assigned to more than one category: {matches}")
    elif len(matches) == 0:
        if ignore_classless:
            return ''
        else:
            raise Exception(f"Skeleton {skel_id} does not belong to any category in cat_list")
    else:
        return matches[0]





