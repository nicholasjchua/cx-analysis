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
        raise Exception(f"The annotation: {annot} does not exist in project: {cfg.p_id}")
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
        print(f"{skel_id}: root = {root_id}, with tag = {data}")
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
def cx_in_skel(skel_id: str, cfg: Config, r_nodes: List) -> Tuple:
    """
    Get a Dict of all connectors PRESYNAPTICally associated with a neuron, and the associated link data
    """

    op_path = "/connectors/"
    post_data = {"skeleton_ids": skel_id,
                 "relation_type": "presynaptic_to", # doesn't seem to do anything (links w relation=15 still present)
                 "with_tags": True,
                 "with_partners": True}

    res_code, data = do_post(op_path, post_data, cfg)
    # data['partners'] is how you get the cx_id: [links] dictionary

    r_connectors = set()
    tmp_connector_data = dict()
    tmp_link_data = []
    for cx_id, p_data in data["partners"].items():
        for link in p_data:
            if link[3] == 15 and str(link[1]) in r_nodes:
                # print(f"Found a restricted connector {cx_id}")
                r_connectors.add(cx_id)
                continue
            elif link[3] == 16:
                '''
                if r_nodes is not None and str(link[1]) in r_nodes:
                    print(f"Found a restricted connector {cx_id}")
                    r_connectors.add(cx_id)
                    continue
                else:
                '''
                tmp_link_data.append({'link_id': str(link[0]),
                                      'pre_skel': skel_id,
                                      'post_skel': str(link[2]),
                                      'post_node': str(link[1]),
                                      'cx_id': cx_id})
                tmp_connector_data.setdefault(cx_id, []).append(link)
            else:
                continue
    # Filter out restricted ones
    if len(r_connectors) > 0:
        link_data = [l for l in tmp_link_data if l['cx_id'] not in r_connectors]
        connector_data = {c: data for c, data in tmp_connector_data.items() if c not in r_connectors}
    else:
        link_data = tmp_link_data
        connector_data = tmp_connector_data

    return connector_data, link_data, list(r_connectors)




