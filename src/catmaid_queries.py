from typing import List, Tuple, Union, Dict, Sequence
import requests
import numpy as np
from scipy.spatial import distance
import networkx as nx
import itertools
from pprint import pprint
from src.config import Config
from glob import glob
import pandas as pd
#from cartridge_metadata import lamina_subtypes


####################
# REQUEST WRAPPERS #
####################
def do_get(apipath: str, cfg: Config) -> Tuple:
    """
    Wraps get requests. Performs specific action based on the operation specificed by apipath
    :param apipath: API path for a specific get operation, e.g. '/annotations/'
    :return response: True if the request was successful
    :return results: A json of the results if sucessful, a string if not.
    """
    p_url, token, p_id = cfg.cm_access()
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


def do_post(apipath: str, postdata: Dict, cfg: Config) -> Tuple:
    """
    Wraps post requests. Performs specific action based on the operation specificed by apipath
    and the fields in postdata

    :return response: True if the request was successful
    :return results: A json of the results if successful, a string if not.
    """
    p_url, token, p_id = cfg.cm_access()
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


# ANNOTATION QUERIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def skels_in_annot(annot: Union[int, str], cfg: Config) -> Tuple[List[str], List[str]]:
    """
    Given an annotation ID, produce lists of skeleton IDs and neuron names
    :param: annot: annotation to query neurons
    :param:cfg: Config, object stores analysis configs
    :return: neuron_names: Lists of (skel_id, neuron_name)
    """
    if type(annot) is str:
        annot_id = annot_to_id(annot, cfg)
    else:
        annot_id = annot

    op_path = "/annotations/query-targets"
    post_data = {"annotated_with": annot_id,
                 "types": ["skeleton", "neuron"]}
    res_code, data = do_post(op_path, post_data, cfg)

    if len(data["entities"]) == 0:
        raise Exception(f"Entities annotated with annotation ID: {annot_id} not found")
    skeleton_ids = [str(entity.get("skeleton_ids")[0]) for entity in data["entities"]]
    neuron_names = [str(entity.get("name")) for entity in data["entities"]]

    if None in (skeleton_ids or neuron_names):
        raise Exception("Entities are missing fields")

    return skeleton_ids, neuron_names


def annot_in_skel(skel_id: str, cfg: Config) -> List[str]:
    """
    Fetch list of annotations associated with the skeleton ID, raises exception if no annotations found
    :param cfg: Dict, of analysis options
    :param skel_ids: str,the numerical skeleton ID
    :return: annot_list, List, of annotations
    """
    op_path = "/annotations/forskeletons"
    post_data = {"skeleton_ids": skel_id}
    res_code, data = do_post(op_path, post_data, cfg)

    annot_list = list(data["annotations"].values())
    if annot_list is []:
        raise Exception(f"{skel_id} has no annotations")
    else:
        return annot_list

def annot_to_id(annot: str, cfg: Config) -> int:
    """
    Search project for an annotation, get its numerical ID
    :param token: Catmaid API token
    :param p_id: Project ID
    :param annot: str, Annotation
    :returns annot_id: int
    """

    op_path = "/annotations/"
    res_code, data = do_get(op_path, cfg)
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
def get_root_id(skel_id: str, cfg: Config) -> str:
    """
    Get the node ID corresponding to the root of a skeleton_id
    :param skel_id: str
    :return node_id: str
    """
    op_path = f"/skeletons/{skel_id}/root"
    res_code, root = do_get(op_path, cfg)
    node_id = root.get('root_id', None)
    if node_id is None:
        raise Exception(f"Root node not found for skeleton: {skel_id}")
    else:
        return str(node_id)


def fetch_node_data(node_id: str, cfg: Config) -> List:
    """
    Get data associated with a node ID
    :param node_id: str
    :param cfg:
    :return: List, of data corresponding to the node
    """
    op_path = f"/treenodes/{node_id}/compact-detail"
    res_code, node_data = do_get(op_path, cfg)
    # TODO what does this look like again?
    if type(node_data) is list:
        return node_data
    else:
        raise Exception(f"Could not find node with ID: {node_id}")




def node_with_tag(skel_id: str, root_id: str, tag_regex: str, cfg: Config, first: bool=True) -> Union[str, List]:
    """
    Returns the node_id of the first node in the skeleton tagged with 'tag_regex'

    Note: the api call returns a list of nodes in ascending distance from root_id, this function returns the one
    nearest to root. The tag could also be a regular expression.
    :param skel_id: Skeleton ID
    :param root_id: ID of root node. Used to sort tagged nodes by distance
    :param cfg: Config object
    :param tag_regex: Tag you want to query
    :return: node_id: str node ID of the tagged treenode. If first=False, returns a list of [nodeID,nodeData] in
    ascending order of distance from root.
    """
    # TODO Check the full output of this API call
    op_path = f"/skeletons/{skel_id}/find-labels"
    post_data = {"treenode_id": int(root_id),
                 "label_regex": str(tag_regex)}
    res_code, data = do_post(op_path, post_data, cfg)

    if len(data) == 0:
        raise Exception(f"Skeleton {skel_id} does not have a node tagged with {tag_regex}")
    elif first:
        return str(data[0][0])
    else:
        print("list of {len(data)} nodes and their node_data")
        return data





# SKELETON QUERIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def skel_compact_detail(skel_id: str, cfg: Config) -> List:
    """
    Returns a list of treenodes for a given skeleton
    The 'post' version of this API call doesn't seem to work, so I don't know how to have 'with_connectors' etc.
    nodes is a list of lists: [[nodes], [connectors], {nodeID: [tags]}], but currently the latter two are empty

    :param skel_id: Skeleton ID
    :param cfg: Config object
    :return: nodes[0], the first element which contains the node data
    """

    op_path = f"/skeletons/{skel_id}/compact-detail"

    res_code, nodes = do_get(op_path, cfg)
    return nodes[0]


# CONNECTOR QUERIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def cx_in_skel(skel_id: str, cfg: Dict, r_nodes: List=None) -> Tuple:
    """
    Get a Dict of all connectors PRESYNAPTICally associated with a neuron, and the associated link data

    :param token:
    :param p_id:
    :param skel_id:
    :param r_str: Optional tag used to filter out links that are not in between the skeleton's root_node and the
    first node tagged with 'r_str'
    :return connector_data: Dict, with entries for each of its presynaptic connector_ids: [link_data]
    """

    op_path = "/connectors/"
    post_data = {"skeleton_ids": skel_id,
                 "relation_type": "presynaptic_to", # doesn't seem to do anything (links w relation=15 still present)
                 "with_tags": True,
                 "with_partners": True}

    res_code, data = do_post(op_path, post_data, cfg)
    # data['partners'] is how you get the cx_id: [links] dictionary

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
                    connector_data.setdefault(c_id, []).append(link)

    print(f"connectors: {len(connector_ids)}, excluded connectors = {len(r_connectors)}")
    return connector_data, list(r_connectors)

def out_cx_ids_in_skel(skel_id: str, cfg: Dict, r_nodes: List=None) -> Tuple:
    """
    List of all outgoing connectors for a skeleton

    :param token:
    :param p_id:
    :param skel_id:
    :param r_str: Optional tag used to filter out links that are not in between the skeleton's root_node and the
    first node tagged with 'r_str'
    :return connector_data: Dict, with entries for each of its presynaptic connector_ids: [link_data]
    """
    token, p_id, project_url = project_access(cfg)
    op_path = "/connectors/"
    post_data = {"skeleton_ids": skel_id,
                 "relation_type": "presynaptic_to", # doesn't seem to do anything (links w relation=15 still present)
                 "with_tags": True,
                 "with_partners": True}

    res_code, data = do_post(token, project_url, p_id, op_path, post_data)
    # data['partners'] is how you get the cx_id: [links] dictionary
    pprint(data)

    cx_ids = set()
    cx_ids_excluded = set()
    for this_cx, p_data in data["partners"].items():
        for link in p_data:
            if link[3] != 16:
                continue
            else:
                if r_nodes is not None and str(link[1]) in r_nodes:
                    cx_ids_excluded.add(this_cx)
                    continue
                else:
                    cx_ids.add(this_cx)

    print(f"connectors: {len(connector_ids)}, excluded connectors = {len(r_connectors)}")
    return list(cx_ids), list(cx_ids_excluded)

def cx_data(skel_id: str, cx_id: str, cfg: Dict):
    """ CURRENTLY NOT USED
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
    res_code, data = do_get(token, project_url, p_id, op_path)

    assert(str(data["connector_id"]) == cx_id)

    link_data = []
    pre_to_check = []  # to return list of skel_ids
    for l in data["partners"]:
        assert(type(l) is dict)
        if l['relation_id'] == 16:
            # things that are the same for this connector
            link_data.append({'link_id': str(l['link_id']),
                              'pre_skel': skel_id,
                              'post_skel': str(l['skeleton_id']),
                              'post_node': str(l['partner_id']),  # need to confirm is this is treenode
                              'cx_id': cx_id,
                              'cx_x': data['x'], 'cx_y': data['y'], 'cx_z': data['z']})
        elif l['relation_id'] == 15 and str(l['skeleton_id']) != skel_id:
            pre_to_check.append(str(l['skeleton_id']))
        elif l['relation_id'] == 15 and str(l['skeleton_id']) == skel_id:
            continue
        else:  # this is usually a mistake in Catmaid
            raise Exception(f"Found a link with an unknown relation ID in connector: {cx_id}")

    return link_data, pre_to_check

### MOVE TO DATA ANALYSIS TOOLS
'''
def n_syn_between(skel_data: Dict, pre_id: str, post_id: str):


    pre_cx_data = skel_data[pre_id]['out_cx']

    if len(pre_cx_data) < 1:  # no outgoing connectors
        return count
    else:
        # Loop across connectors
        for cx_id, links in skel_data[pre_id]['out_cx'].items():
        # Loop across the links in each connector
            for l in links:
'''

