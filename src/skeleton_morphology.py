from typing import List, Dict, Tuple, Union
import numpy as np
import pandas as pd
import json
from src.node_ops import segment_skeleton, find_central_segment, measure_path_lengths, measure_seg_distances
from src.connectome import Connectome
"""
skeleton_morphology.py
Methods to compute and save morphology data. 
:Date: 13-Apr-2021
:Authors: Nicholas Chua 
"""

def run_morphology_analysis(C: Connectome, skel_ids: List[str], 
                            restrict_tags: Union[Tuple, str]=None, 
                            save_file: str=None, verbose: bool=False) -> Tuple[Dict]:
    """
    Run a series of morphology measurements for specified skeletons in a Connectome object
    
    Parameters
    ----------
    :param C: Connectome, object defined in src.connectome
    :param skel_ids: List[str], a list of skeleton IDs (str) to run morphology analysis
    :param restrict_tags: Union[Tuple, str], CATMAID tags pointing to start and end nodes to 
    constrain analysis. When one str is given, uses the root as the start point 
    (has only been tested for cases with one restrict node, 'lamina_end')
    :param save_path: str, optional, full file path to save json of data.
    :param verbose: bool, print information on each skeleton for debugging
    
    :return: tuple, containing four dictionaries: 
        segments,
        central_segs,
        seg_lengths,
        seg_distances
    """
    segments = dict()
    central_segs = dict()
    seg_lengths = dict()
    seg_distances = dict()
    
    for this_skel in skel_ids:
        this_skel = str(this_skel) # skel_id keys in C.skel_data are str
        if this_skel not in list(C.skel_data.keys()):
            raise Exception(f"Skeleton with ID {this_skel} not found in Connectome.skel_data")
        else:
            data = C.skel_data[this_skel]
            # this returns a dict where each entry is a list of contiguous node IDs
            segments[this_skel] = segment_skeleton(this_skel, 
                                                   cfg=C.cfg, 
                                                   node_data=data.skel_nodes, 
                                                   restrict_nodes=data.r_nodes, 
                                                   verbose=verbose)
            # List of nodes along the 'backbone'. This will break if end_tag is not str
            central_segs[this_skel] = find_central_segment(this_skel, 
                                                           end_tag=restrict_tags,
                                                           cfg=C.cfg, 
                                                           restrict_nodes=data.r_nodes, 
                                                           node_data=data.skel_nodes)
            # measure path lengths of each segment
            seg_lengths[this_skel] = measure_path_lengths(segments[this_skel], 
                                                        cfg=C.cfg, 
                                                        node_data=data.skel_nodes)
            # measure distance between the start and end of each segment
            seg_distances[this_skel] = measure_seg_distances(segments[this_skel], 
                                                             cfg=C.cfg, 
                                                             node_data=data.skel_nodes)
    # Save results as json
    if save_file is not None:
        if save_file[-5:] != '.json':
            save_file = save_file + '.json'
            
        results = {'segments': segments, 
                   'central_segs': central_segs, 
                   'seg_lengths': seg_lengths, 
                   'seg_distances': seg_distances}
        
        with open(save_file, 'x') as fh:
            json.dump(results, fh)
        print(f"Morphology data saved as {save_file}")
    
    return segments, central_segs, seg_lengths, seg_distances
    
    
def strahler_order(segments: Dict, node_data: List, r_nodes: List=None):
    """
    Compute the strahler order of each branch point in a skel segments dict
    """
    if r_nodes is not None:
        # the ids saved in C.skel_data.r_nodes is str
        r_nodes = [int(n) for n in r_nodes]
        node_data = [n for n in node_data if n not in r_nodes]
    
    root = find_root_node(node_data)
    leaves = find_leaf_nodes(node_data) 
        
    # First current branch is the last node in segments[root]
    results = _strahler(segments, segments[root][-1], node_data, leaves, results)
    return results
     
    
    #branch_order = 



def _strahl(segments: Dict, current_bp: int, node_data: List,  
            leaves: List, results: Dict):

    child_nodes = [n[0] for n in node_data if n[1] == int(current_bp)]
    child_bps = [segments[c] for c in children]
    child_orders = []
    for cbp in child_bps:
        if cbp in leaves:
            child_orders.append(1)
        else:
            #child_results = _strahl(
    
    results[current_segment] = max([_strahl(c) for c in child_branches])
    

    
def find_root_node(node_data: List) -> int:
    """
    Quick and easy way to find the root node from node_data
    doesn't require making a server query like catmaid_queries.get_root_id()
    :param node_data: 2D list, each line is a node, 
                      e.g. [2156, None, 4, 36192.0, 73024.0, 25376.0, 0.0, 5]
                           [node_id, parent_id, ?, x, y, z, ?, ?]
    :return: int, node_id of root
    """
    
    root = [int(n[0]) for n in node_data if n[1] is None] # no parent
    if len(root) != 1:  
        raise Exception(f"Found {len(root)} root nodes in node_list")
    else:
        return root[0]
    

def find_leaf_nodes(node_data: List) -> List[int]:
    """
    Quick and easy way to find the leaf nodes in node_data
    doesn't require making a server query
    :param node_data: 2D list, each line is a node, 
                      e.g. [2156, None, 4, 36192.0, 73024.0, 25376.0, 0.0, 5]
                           [node_id, parent_id, ?, x, y, z, ?, ?]
    :return: List[int], node_id of root
    """
    parent_nodes = np.array(node_data).T[1]
    return [int(n[0]) for n in node_data if n[0] not in parent_nodes]
    
    
def precomputed_children(node_data: List) -> Dict:
                
    the_map = dict()
    for this_node in np.array(node_data).T[0]:
        the_map[this_node] = [n[0] for n in node_data if n[1] == int(this_node)]
    
    return the_map
        
        