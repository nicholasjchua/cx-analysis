import numpy as np
from typing import Dict, Union, Tuple, List
from src.catmaid_queries import fetch_node_data, get_root_id, node_with_tag, skel_compact_detail
from src.config import Config
from scipy.spatial import distance


def segment_skeleton(skel_id: str, cfg: Config, nodes: List=None, restrict_tags=None) -> List:
    
    root_id = get_root_id(skel_id, cfg)

    if nodes is None:
        node_list = skel_compact_detail(skel_id, cfg)
    else:
        node_list = nodes
        
    if restrict_tags is not None:
        node_list = nodes_betwixt(skel_id, cfg, restrict_tags, node_list, invert=False)
    branches = dict()
    branches = find_branch_points(node_list, current=root_id, branches=branches, last_branch=root_id)
    
    return branches
    
    
        

def nodes_betwixt(skel_id: str, cfg: Config, restrict_tags: Union[str, Tuple], nodes: List=None,
                       invert: bool=True) -> Union[List[str], Tuple]:
    """
    Get a list of node_ids for nodes between two specified tags on a skeleton.
    TODO: allow this to take node_ids for start and end instead
    :param skel_id: Skeleton ID
    :param restrict_tags: str or tuple of two strings. Giving just one will define the segment as root -> <tag>
    :param nodes: (optional) list of node IDs so they aren't fetched again
    :param invert: If true, returns the nodes OUTSIDE the tagged segment.
    :return:
    """
    root_id = get_root_id(skel_id, cfg)

    if nodes is None:
        node_list = skel_compact_detail(skel_id, cfg)
    else:
        node_list = nodes

    if type(restrict_tags) is str:
        start = root_id
        end = node_with_tag(skel_id, root_id, restrict_tags, cfg)
    elif len(restrict_tags) == 1:
        start = root_id
        end = node_with_tag(skel_id, root_id, restrict_tags[0], cfg)
    elif len(restrict_tags) == 2:
        start = node_with_tag(skel_id, root_id, restrict_tags[0], cfg)
        end = node_with_tag(skel_id, root_id, restrict_tags[1], cfg)
    else:
        raise Exception("More than two restrict_tags given")

    dist = check_dist(start, end, cfg)
    nodes_within = traverse_nodes(node_list, int(start), int(end))

    if invert:  # TODO log number of nodes before/after restricting
        return [str(n[0]) for n in node_list if n[0] not in nodes_within]
    else:
        return [str(n) for n in nodes_within]


def dist_two_nodes(n1: str, n2: str, cfg: Config) -> float:
    coord1 = np.array(node_coords(n1, cfg), dtype=float)
    coord2 = np.array(node_coords(n2, cfg), dtype=float)

    dist = distance.euclidean(coord1, coord2)
    return dist


def check_dist(n1: str, n2: str, cfg: Config) -> float:
    """
    Raises an exception if two nodes are too close (i.e. lamina_end and root)
    :param n1: str, node_id of root
    :param n2:
    :param cfg:
    :return dist: float, if sufficiently far, returns the distance in nm
    """
    dist = dist_two_nodes(n1, n2, cfg)
    if np.abs(dist) < 1000.0:
        raise Exception(f'Nodes too close ({dist: .01f}nm)')
    else:
        return dist


def traverse_nodes(node_list: List, start: int=None, end: int=None):
    """
    Recursive walk through node_list from 'start' node, collecting node IDs in a list.
    Function will split when a branch is hit (i.e. node has two children)
    Ends when when an arbor terminates or if 'end' node is found
    :param node_list: List, of nodes to traverse
    :param start:
    :param end:
    :return:
    """
    current = start
    node_ids = []
    if current != end:
        node_ids.append(current)
        children = [n[0] for n in node_list if n[1] == current]
        if children == []:  # end of branch
            return node_ids
        else:
            for c in children:
                deeper_nodes = traverse_nodes(node_list, start=c, end=end)
                node_ids.extend(deeper_nodes)

    return node_ids


def find_branch_points(node_list: List, current: int=None, branches: Dict=None, last_branch=None):
    
    #these_branches = branches
    b = branches
    children = [n[0] for n in node_list if n[1] == current]
    #b.setdefault(last_branch, []).append(current)
    
    if len(children) == 0:  # End of branch
        return b
    elif len(children) > 1:  # Branch point
        b.update({current, [current]})
        for c in children:
            deeper_b = find_branch_points(node_list, current=c, branches=b, last_branch=current)
            b.update(deeper_b) 
        return b
    else: # 1 child, continue on 
        deeper_b = find_branch_points(node_list, current=children[0], branches=b, last_branch=last_branch)
        b.update(deeper_b) 
        return b


def node_coords(node_id: str, cfg:Config) -> Tuple:
    """
    Get the x, y, z coordinates of a node using fetch_node_data,
    TODO: check if nm or voxels
    :param node_id:
    :return x, y, z
    """
    x, y, z = fetch_node_data(node_id, cfg)[2: 5]
    return x, y, z

