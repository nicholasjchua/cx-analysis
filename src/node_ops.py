import numpy as np
from typing import Dict, Union, Tuple, List
from src.catmaid_queries import get_root_id, node_with_tag, skel_compact_detail
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
    branches = {int(root_id): [int(root_id)]}
    branches = find_branch_points(node_list, current=root_id, branches=branches, parent_branch=root_id)
    
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


def traverse_nodes(node_list: List, start: int, end: int):
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
        children = [n[0] for n in node_list if int(n[1]) == int(current)]
        if children == []:  # end of branch
            return node_ids
        else:
            for c in children:
                deeper_nodes = traverse_nodes(node_list, start=c, end=end)
                node_ids.extend(deeper_nodes)

    return node_ids


def find_branch_points(node_list: List, branches: Dict, current:int, parent_branch: int):
    """
    Get the node ID of branch points (and the IDs of intervening nodes in the segment)
    
    {branch ID: [list of node IDs]}
    """
    
    #these_branches = branches
    b = branches
    children = [n[0] for n in node_list if n[1] == int(current)]
    #b.setdefault(last_branch, []).append(current)
    
    if len(children) == 0:  # End of branch
        b[parent_branch].append(current)
        return b
    elif len(children) > 1:  # Branch point
        b.update({int(current): [int(current)]})
        for c in children:
            deeper_b = find_branch_points(node_list, current=int(c), 
                                          branches=b, parent_branch=int(current))
            b.update(deeper_b) 
        return b
    else: # 1 child, continue on 
        b[int(parent_branch)].append(current)
        deeper_b = find_branch_points(node_list, current=int(children[0]), 
                                      branches=b, parent_branch=int(parent_branch))
        b.update(deeper_b) 
        return b
    
def find_end_points(node_list: List) -> List:
    """
    Get IDs of end nodes (leaf nodes) of the skeleton
    """
    parent_nodes = np.array(node_list).T[1]
    end_ids = [n[0] for n in node_list if n[0] not in parent_nodes]
    
    return end_ids

def branch_lengths(node_list: List, branch_list: Dict, end_list: List, cfg: Config) -> Dict:
    
    length_data = dict.fromkeys(end_list)
    c_to_p = {n[0]: n[1] for n in node_list}
    
    for e in end_list:
        current = e
        d = 0.0
        while (current not in branch_list) and (c_to_p[current] is not None):
            parent = c_to_p[current]
            d += dist_two_nodes(current, parent, cfg)
            current = parent
        length_data[e] = d
    return length_data

# def node_coords(node_id: str, cfg:Config) -> Tuple:
#     """
#     Get the x, y, z coordinates of a node using fetch_node_data,
#     TODO: check if nm or voxels
#     :param node_id:
#     :return x, y, z
#     """
#     x, y, z = fetch_node_data(node_id, cfg)[2: 5]
#     return x, y, z

