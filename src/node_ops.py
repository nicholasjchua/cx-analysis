import numpy as np
from typing import Dict, Union, Tuple, List
from src.catmaid_queries import fetch_node_data, get_root_id, node_with_tag, skel_compact_detail
from src.config import Config
from scipy.spatial import distance


def nodes_betwixt(skel_id: str, cfg: Config, restrict_tag: Union[str, Tuple], nodes: List=None,
                       invert: bool=True) -> Union[List[str], Tuple]:
    """
    Get a list of node_ids for nodes between two specified tags on a skeleton.
    TODO: allow this to take node_ids for start and end instead
    :param skel_id: Skeleton ID
    :param restrict_tag: str or tuple of two strings. Giving just one will define the segment as root -> tag
    :param nodes: (optional) list of node IDs so they aren't fetched again
    :param invert: If true, returns the nodes OUTSIDE the tagged segment.
    :return:
    """
    root_id = get_root_id(cfg)

    if nodes is None:
        node_list = skel_compact_detail(skel_id, cfg)
    else:
        node_list = nodes

    if type(restrict_tag) is str:
        start = root_id
        end = node_with_tag(skel_id, root_id, restrict_tag, cfg)
    else:
        start = node_with_tag(skel_id, root_id, restrict_tag, cfg)
        end = node_with_tag(skel_id, root_id, restrict_tag, cfg)
    dist = check_dist(start, end, cfg)
    print(f'Nodes defining skeletal segment for {skel_id} are {dist} nm apart (as the crow flies)')

    nodes_within = traverse_nodes(node_list, int(start), int(end))
    # pprint(f"Total nodes: {len(node_list)}, Nodes between tags: {len(nodes_within)}")

    if invert:  # TODO log number of nodes before/after restricting
        return [str(n[0]) for n in node_list if n[0] not in nodes_within]
    else:
        return nodes_within


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
    Ends when when an arbor terminates of if 'end' node is found
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


def node_coords(node_id: str, cfg:Config) -> Tuple:
    """
    Get the x, y, z coordinates of a node using fetch_node_data,
    TODO: check if nm or voxels
    :param node_id:
    :return x, y, z
    """
    x, y, z = fetch_node_data(node_id, cfg)[2: 5]
    return x, y, z

