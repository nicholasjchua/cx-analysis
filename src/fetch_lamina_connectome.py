#!/usr/bin/env python
import argparse
import os, sys
import glob
from typing import Dict, List, Tuple, Union
import itertools
from src.config import parse_cfg_file
from src.catmaid_queries import *

from numpy.matlib import repmat
from tqdm import tqdm

"""
fetch_lamina_connectome.py
Fetch and organise data from CATMAID
"""

def main():

    cfg = parse_cfg_file()  # see config/default_cfg.json for defaults
    ### MAIN FETCH ###

    skel_data, excluded_terminals = fetch_connectome(cfg)

    cx_types = [(u, v) for u, v in itertools.permutations(cfg['subtypes'], 2)]

def fetch_connectome(cfg: Dict) -> Tuple:
    """
    Parse data of skeletons annotated with the global annotation in analysis configs
    :param cfg: Analysis configs
    :returns skel_data: Dict hashed by skel_id containing name, ommatidia 'om', subtype 'st', outgoing connections 'out
    cx', restricted nodes and connectors 'r_nodes', 'r_connectors'
    :returns ref:
    """
    # Catmaid Access

    skel_ids, neuron_names = skels_in_annot(cfg['annot'], cfg)
    print(f"Found {len(skel_ids)} skeletons annotated with {cfg['annot']}")

    skel_data = {s: {'name': n,  # catmaid neuron name
                     'om': n[2:4],  # which ommatidia this is from, from neuron_name, not 'ommatidia_XY' annotation
                     'st': [],  # neuron subtype, assign_exclusive_subtype forces this to be len 1
                     'out_cx': [],  # Outgoing connectors
                     'r_nodes': [],  # List of nodes that are in the restricted zone (beyond lamina in this case
                     'r_cx': []}
                 for s, n in zip(skel_ids, neuron_names)}

    for i, this_skel in tqdm(enumerate(skel_ids)):
        # 1. Assign each a subtype 'st'
        this_subtype = assign_excl_category(this_skel, cfg, ignore_classless=True)
        skel_data[this_skel]["st"] = [this_subtype]
        # 2. If subtype knowed to traverse medulla, traverse its nodes, determine which nodes are outside lamina
        if cfg['end_tag'] is not '' and this_subtype in cfg['in_medulla']:
            r_nodes = nodes_between_tags(this_skel, cfg, invert=True)
            skel_data[this_skel]['r_nodes'] = r_nodes

        # these_out_cx, these_r_cx = cx_in_skel(this_skel, cfg, r_nodes=skel_data[this_skel]['r_nodes'])
        # skel_data[this_skel]['out_cx'].update('these_out_cx')
        # skel_data[this_skel]['r_cx'] = these_r_cx
        skel_data[this_skel]['out_cx'], skel_data[this_skel]['r_cx'] = out_cx_ids_in_skel(this_skel, cfg,
                                                                                          skel_data['r_nodes'])


    return skel_data

def local_adjacency_matrices(skel_data: Dict, cfg: Dict) -> Tuple:
    """

    """
    om_key, st_key = skel_id_keys(skel_data)  # Keys to the skel_ids associated w an ommatidia or subtype
    id_mat = get_skel_mat(cfg, om_key, st_key)  # each skeleton's position on the adj_mat
    om_order = sorted(om_key.keys())

    adj_mat = np.zeros((len(om_order), id_mat.shape[1], id_mat.shape[1]), dtype=int)
    unknown_post = np.zeros(id_mat.shape[0], dtype=int)


    for i, om in enumerate(om_order):
        home_skeletons = om_key[om]
        for j, pre_skel in enumerate(id_mat[i]):
            if pre_skel == '-1':  # cartridges with missing neurons coded with -1 (only allowed for L4)
                adj_mat[i, j, :] = -1
                continue
            else:
                # analyze the partner data for each skeleton in our adjacency matrix
                print(f'PRESKEL: {pre_skel}')
                out_cx = skel_data[pre_skel]['out_cx']
                if len(out_cx) < 1:  # not presynaptic
                    continue
                else:
                    for cx_id, links in out_cx:
                        for l in links:





    return adj_mat, unknown_post

# DATA ORGANISATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_excl_category(skel_id: str, excl_cats: List, cfg: Dict, ignore_classless: bool=False) -> str:
    """
    Searches through the annotations of a skeleton to find a match with one of the st in excl_list
    Raises an exception if multiple matches are found (i.e. neuron can only belong to one of the categories)
    Can ignore instances where none of the categories are found
    :param skel_id: str, Skeleton ID
    :param excl_cats: List, Contains str of annotations you want to use as categories
    :param cfg: Dict, analysis options
    :param ignore_classless: bool, if False (default), will raise an Exception if a neuron doesn't have any annotions
    in cat_list. If True, returns None instead
    :return: str, the skeleton's category, None if no category was found
    """
    categories = set(cfg['subtypes'])
    annotations = set(annot_in_skel(skel_id, cfg))

    intersect = list(categories & annotations)
    if len(intersect) > 1:
        raise Exception(f"Skeleton {skel_id} can be assigned to more than one category: {intersect}")
    elif len(intersect) == 0:
        if ignore_classless:
            return ''
        else:
            raise Exception(f"Skeleton {skel_id} does not belong to any category in cat_list")
    else:
        return intersect[0]

def skel_id_keys(skel_data: Dict)-> Tuple:

    om_hash = dict()
    st_hash = dict()

    for this_id, this_data in skel_data.items():
        # build {om:[skeleton ids]}
        if this_data["om"] is None:
            raise Exception(f"Skeleton {this_id} is not associated with any ommatidia")
        elif this_data["om"] is List:
            raise Exception(f"Skeleton {this_id} is associated with more than one ommatidia")
        else:
            om_hash.setdefault(this_data["group"], []).append(str(this_id))
        # build {st:[skeleton ids]}
        if this_data["st"][0] is None:
            raise Exception(f"Skeleton {this_id} is not associated with a subtype")
        elif isinstance(this_data["st"], list) and len(this_data["st"]) > 1:
            raise Exception(f"Skeleton {this_id} is associated with more than one subtype: {this_data['st']}")
        else:
            st_hash.setdefault(this_data["st"][0], []).append(str(this_id))

    return om_hash, st_hash

def get_skel_mat(cfg: Dict, om_hash: Dict, st_hash: Dict) -> np.array:
    """
    Get a 2D matrix of skeleton_ids according to their subtype and ommatidia assignment

    :param cfg: Dict, analysis configs
    :param om_hash: Dict, om -> [skel_ids]
    :param st_hash: Dict, subtype -> [skel_ids]
    :return:
    """
    om_list = sorted(om_hash.keys())
    ids = []

    for i, this_om in enumerate(om_list):
        skels_in_om = om_hash[this_om]
        tmp = []
        for ii, this_type in enumerate(cfg['subtypes']):
            filtered = [str(skel) for skel in skels_in_om if skel in st_hash.get(this_type, [])]
            if len(filtered) == abs(cfg['expected_n']):
                tmp = [*tmp, *filtered]
            # expected_n = -1 means that subtype is allowed to not exist
            elif len(filtered) == 0 and cfg['expected_n'] == -1:
                tmp.append('-1')
                raise Warning(f'Warning: No neuron of type {this_type} found in {this_om}')
            else:
                raise Exception(f"Unexpected number of neurons for om: {this_om} subtype: {this_type}."
                                f"Got the following ids: \n{filtered}")

        ids.append(tmp)
    ids = np.array(ids, dtype=str)
    return ids







