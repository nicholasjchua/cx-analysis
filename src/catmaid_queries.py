from typing import List, Tuple, Union, Dict, Sequence
import requests
import numpy as np
from scipy.spatial import distance
import networkx as nx
import itertools
from pprint import pprint
from glob import glob
import pandas as pd
from cartridge_metadata import lamina_subtypes


####################
# REQUEST WRAPPERS #
####################
def do_get(token: str, p_url: str, p_id: int, apipath: str) -> Tuple:
    """
    Wraps get requests. Performs specific action based on the operation specificed by apipath
    :param token: User's Catmaid access token
    :param p_url: URL of your Catmaid instance
    :param p_id: Project ID
    :param apipath: API path for a specific get operation, e.g. '/annotations/'
    :return response: True if the request was successful
    :return results: A json of the results if sucessful, a string if not.
    """
    path = p_url + "/" + str(p_id) + apipath

    result = requests.get(path, headers={'X-Authorization': 'Token ' + token})
    try:
        jresult = result.json()
    except ValueError:
        jresult = None
    if jresult is not None:
        if 'type' in jresult:  # API doesn't provide another way to get at a python-objects-structured parse
            if jresult['type'] == "Exception":
                print("exception info:")
                print(jresult['detail'])
                return False, "Something went wrong"
    if result.status_code == 200:
        if jresult is not None:
            return True, jresult
        else:
            raise Exception(f"Did not return json, but also not an error, text is {result.text}")
    else:
        return False, f"Something went wrong with {apipath}, return code was {result.status_code}"


def do_post(token: str, p_url: str, p_id: int, apipath: str, postdata: Dict) -> Tuple:
    """
    Wraps post requests. Performs specific action based on the operation specificed by apipath
    and the fields in postdata
    :param token: User's Catmaid access token
    :param p_url: URL of your Catmaid instance
    :param p_id: Project ID
    :param apipath: API path for a specific get operation, e.g.
    :param postdata: A Dict with K:Vs specified by particular post request
    :return response: True if the request was successful
    :return results: A json of the results if successful, a string if not.
    """
    path = p_url + "/" + str(p_id) + apipath

    result = requests.post(path, data=postdata, headers={'X-Authorization': 'Token ' + token})
    try:
        jresult = result.json()
    except ValueError:
        jresult = None
    if jresult is not None:
        if 'type' in jresult:
            if jresult['type'] == "Exception":
                print("exception info:")
                print(jresult['detail'])
                return False, "Something went wrong"
    if result.status_code == 200:
        if jresult is not None:
            return True, jresult
        else:
            raise Exception(f"Did not return json, but also not an error, text is {result.text}")
    else:
        return False, f"Something went wrong with {apipath}, return code was {result.status_code}"


def project_access(cfg: Dict) -> Tuple:

    return cfg['user_token'], cfg['project_id'], cfg['project_url']


# ANNOTATION QUERIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def skels_in_annot(annot: Union[int, str], cfg: Dict) -> Tuple[List[str], List[str]]:
    """
    Given an annotation ID, produce lists of skeleton IDs and neuron names

    Calls annot_to_annot_id() if you give it string annotations
    :param token: Catmaid API token
    :param p_id: Project
    :param annot: Annotation ID or name
    :return: skeleton_ids: List of skeleton_ids
    :return: neuron_names: List of neuron_names
    """
    token, p_id, project_url = project_access(cfg)
    op_path = "/annotations/query-targets"

    if type(annot) is str:
        annot_id = get_annot_id(token, p_id, annot)
    else:
        annot_id = annot

    post_data = {"annotated_with": annot_id,
                 "types": ["skeleton", "neuron"]}
    res_code, data = do_post(token, project_url, p_id, op_path, post_data)

    if len(data["entities"]) == 0:
        raise Exception(f"Entities annotated with annotation ID: {annot_id} not found")
    skeleton_ids = [str(entity.get("skeleton_ids")[0]) for entity in data["entities"]]
    neuron_names = [str(entity.get("name")) for entity in data["entities"]]

    if None in (skeleton_ids or neuron_names):
        raise Exception("Entities are missing fields")

    return skeleton_ids, neuron_names

def annot_in_skel(skel_id: str, cfg: Dict) -> List:
    """
    Fetch list of annotations associated with the skeleton ID, raises exception if no annotations found
    :param cfg: Dict, of analysis options
    :param skel_ids: str,the numerical skeleton ID
    :return: annot_list, List, of annotations
    """
    token, p_id, project_url = project_access(cfg)
    op_path = "/annotations/forskeletons"
    post_data = {"skeleton_ids": skel_id}
    res_code, data = do_post(token, project_url, p_id, op_path, post_data)

    annot_list = list(data["annotations"].values())
    if annot_list is []:
        raise Exception(f"{skel_id} has no annotations")
    else:
        return annot_list

def get_annot_id(annot: str, cfg: Dict) -> int:
    """
    Search project for an annotation, get its numerical ID
    :param token: Catmaid API token
    :param p_id: Project ID
    :param annot: str, Annotation
    :returns annot_id: int
    """
    token, p_id, project_url = project_access(cfg)
    op_path = "/annotations/"
    res_code, data = do_get(token, project_url, p_id, op_path)

    data = data['annotations']
    annot_id = None
    for this_annot in data:
        if this_annot['name'] == annot:
            annot_id = this_annot['id']
    if annot_id is None:
        raise Exception(f"The annotation: {annot} does not exist in project: {p_id}")
    else:
        return annot_id


# TREENODE QUERIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_root_id(skel_id: str, cfg: Dict) -> str:
    """
    Get the node ID corresponding to the root of a skeleton_id
    :param skel_id: str
    :return node_id: str
    """
    token, p_id, project_url = project_access(cfg)
    op_path = f"/skeletons/{skel_id}/root"
    res_code, root = do_get(token, project_url, p_id, op_path)
    node_id = root.get('root_id', None)
    if node_id is None:
        raise Exception(f"Root node not found for skeleton: {skel_id}")
    else:
        return str(node_id)


def fetch_node_data(node_id: str, cfg) -> List:
    """
    Get data associated with a node ID (TODO: what does this look like again?)
    :param node_id: str
    :param cfg:
    :return: List, of data corresponding to the node
    """
    token, p_id, project_url = project_access(cfg)
    op_path = f"/treenodes/{node_id}/compact-detail"
    res_code, node_data = do_get(token, project_url, p_id, op_path)

    if type(node_data) is list:
        return node_data
    else:
        raise Exception(f"Could not find node with ID: {node_id}")

def fetch_node_coords(token: str, p_id: int, node_id: str) -> Tuple:
    """
    Get the x, y, z coordinates of a node using fetch_node_data,
    TODO: check if nm or voxels
    :param node_id:
    :return x, y, z
    """
    x, y, z = fetch_node_data(token, p_id, node_id)[2: 5]
    return x, y, z


def first_node_with_tag(skel_id: str, root_id: str, tag_regex: str, cfg: Dict) -> str:
    """
    Returns the node_id of the first node in the skeleton tagged with 'tag_regex'
    TODO: change so that it returns list
    Note: the api call returns a list of nodes in ascending distance from root_id, this function returns the one
    nearest to root. The tag could also be a regular expression.
    :param skel_id: Skeleton ID
    :param root_id: ID of root node. Used to sort tagged nodes by distance
    :param tag_regex: Tag you want to query
    :return: node_id: str node ID of the tagged treenode
    """
    token, p_id, project_url = project_access(cfg)
    op_path = f"/skeletons/{skel_id}/find-labels"
    post_data = {"treenode_id": int(root_id),
                 "label_regex": str(tag_regex)}
    res_code, nodes = do_post(token, project_url, p_id, op_path, post_data)

    if len(nodes) == 0:
        raise Exception(f"Skeleton {skel_id} does not have a node tagged with {tag_regex}")
    else:
        node_id = str(nodes[0][0])
    # return nodes if you want all of them
    return node_id


# TREENODE OPERATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dist_two_nodes(token: str, p_id: int, node1: str, node2: str) -> float:
    coord1 = np.array(fetch_node_coords(token, p_id, node1), dtype=float)
    coord2 = np.array(fetch_node_coords(token, p_id, node2), dtype=float)

    dist = distance.euclidean(coord1, coord2)
    return dist



def nodes_between_tags(skel_id: str, cfg: Dict, restrict_tag: Union[str, Tuple]= '',
                       invert: bool=False) -> List[str]:
    """
    Get a list of node_ids for nodes between two specified tags on a skeleton.
    TODO: allow this to take node_ids for start and end instead
    :param skel_id: Skeleton ID
    :param restrict_tag: str or tuple of two strings. Giving just one will define the segment as root -> tag
    :param invert: If true, returns the nodes OUTSIDE the tagged segment.
    :return:
    """
    root_id = get_root_id(cfg)
    if type(restrict_tag) is str:
        start = root_id
        end = first_node_with_tag(skel_id, root_id, restrict_tag, cfg)
    else:
        start = first_node_with_tag(skel_id, root_id, restrict_tag, cfg)
        end = first_node_with_tag(skel_id, root_id, restrict_tag, cfg)
    dist = check_dist(start, end, cfg)
    print(f'Nodes defining skeletal segment for {skel_id} are {dist} nm apart (as the crow flies)')
    node_list = skel_compact_detail(skel_id, cfg)
    nodes_within = traverse_nodes(node_list, int(start), int(end))
    # pprint(f"Total nodes: {len(node_list)}, Nodes between tags: {len(nodes_within)}")
    if invert:
        restricted = [str(n[0]) for n in node_list if n[0] not in nodes_within]
        print(f"length total: {len(node_list)} length restricted: {len(restricted)}")
        return restricted
    else:
        return nodes_within

def check_dist(root_id: str, end_id: str, cfg: Dict):
    """
    If root is very close to the end tag, there was probably a mistake in CATMAID
    :param root_id: str, node_id of root
    :param end_id:
    :param cfg:
    :return dist: boolean, if sufficiently far, returns the distance in nm
    """
    dist = dist_two_nodes(root_id, end_id, cfg)
    print(dist)
    if np.abs(dist) < 1000.0:
        raise Exception('Node with restrict tag is suspiciously close to root node')
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


# SKELETON QUERIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def skel_compact_detail(skel_id: str, cfg: Dict) -> List:
    """
    Returns a list of treenodes for a given skeleton
    The 'post' version of this API call doesn't seem to work, so I don't know how to have 'with_connectors' etc. =True
    nodes is a list of lists: [[nodes], [connectors], {nodeID: [tags]}], but currently the latter two are empty
    Used by nodes_between_tags
    :param token: Catmaid API token
    :param p_id: Project ID
    :param skel_id: Skeleton ID
    :return: nodes[0], the first element which contains the node data
    """
    token, p_id, project_url = project_access(cfg)
    op_path = f"/skeletons/{skel_id}/compact-detail"
    #post_data = {"skeleton_id": [skel_id],
                 #"with_connectors": True}
    res_code, nodes = do_get(token, project_url, p_id, op_path)
    return nodes[0]


# CONNECTOR QUERIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_connector_data(skel_id: str, cfg: Dict, r_nodes: List=None) -> Dict:
    """
    Similar to above but returns the connectors associated with that neuron
    :param token:
    :param p_id:
    :param skel_id:
    :param r_str: Optional tag used to filter out links that are not in between the skeleton's root_node and the
    first node tagged with 'r_str'
    :return:
    """
    token, p_id, project_url = project_access(cfg)
    op_path = "/connectors/"
    post_data = {"skeleton_ids": skel_id,
                 "relation_type": "presynaptic_to", # doesn't seem to do anything (links w relation=15 still present)
                 "with_tags": True,
                 "with_partners": True}

    res_code, data = do_post(token, project_url, p_id, op_path, post_data)

    connector_ids = set()
    r_connectors = set()
    connector_data = dict()
    for c_id, p_data in data["partners"].items():
        for link in p_data:
            if link[3] != 16:
                continue
            else:
                if r_nodes is not None and str(link[1]) in r_nodes:
                    r_connectors.add(c_id)
                    continue
                else:
                    connector_ids.add(c_id)
                    connector_data.setdefault(c_id, []).append(link)
    print(f"connectors: {len(connector_ids)}, r connectors = {len(r_connectors)}")
    return connector_data

def query_connector_id(skel_id: str, cx_id: str, cfg: Dict):
    """
    Query single connector ID of a skeleton.

    :param token:
    :param p_id:
    :param skel_id:
    :param r_nodes:
    :return link_data: List, of dicts for each link: link_id, pre_skel, post_skel, post_node, cx_id, cx_x,
    cx_y, and cx_z.
    :return pre_to_check: List, of skeleton_ids that were pre-synaptic to a connector (FOR DEBUGGING)
    """
    '''
    output looks like this:
        {
      "connector_id": 28374,
      "x": 32221,
      "y": 73094.5,
      "z": 26843,
      "confidence": 5,
      "partners": [
        {
          "link_id": 260199,
          "partner_id": 333075,
          "confidence": 5,
          "skeleton_id": 221685,
          "relation_id": 16,
          "relation_name": "postsynaptic_to"
        },
        {
          "link_id": 219627,
          "partner_id": 329724,
          "confidence": 5,
          "skeleton_id": 221161,
          "relation_id": 16,
          "relation_name": "postsynaptic_to"
        }, ...
    '''
    token, p_id, project_url = project_access(cfg)
    op_path = f"/connectors/{cx_id}/"
    res_code, cx_data = do_get(token, project_url, p_id, op_path)

    assert(str(cx_data["connector_id"]) == cx_id)

    link_data = []
    pre_to_check = []  # to return list of skel_ids
    for l in cx_data["partners"]:
        assert(type(l) is dict)
        if l['relation_id'] == 16:
            # things that are the same for this connector
            link_data.append({'link_id': str(l['link_id']),
                              'pre_skel': skel_id,
                              'post_skel': str(l['skeleton_id']),
                              'post_node': str(l['partner_id']),  # need to confirm is this is treenode
                              'cx_id': cx_id,
                              'cx_x': cx_data['x'], 'cx_y': cx_data['y'], 'cx_z': cx_data['z']})
        elif l['relation_id'] == 15 and str(l['skeleton_id']) != skel_id:
            pre_to_check.append(str(l['skeleton_id']))
        elif l['relation_id'] == 15 and str(l['skeleton_id']) == skel_id:
            continue
        else:  # this is usually a mistake in Catmaid
            raise Exception(f"Found a link with an unknown relation ID in connector: {cx_id}")

    return link_data, pre_to_check
