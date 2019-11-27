from typing import List, Dict, Iterable
from src.catmaid_queries import *
from src.utils import *
from tqdm import tqdm

class Connectome:

    def __init__(self, cfg):
        #if cfg.dir is not '' and glob(os.path.join(cfg.dir, '*preprocessed.pickle')):
            #self.neuron_data, self.neuron_names = load_preprocessed
        self.cfg = cfg
        self.skel_data, self.neuron_names = self.fetch_connectome(cfg)
        self.skel_data = parse_skeleton_data(self.skel_data, self.neuron_names, self.cfg)




class Neurite:

    def __init__(self, skel_id: str, name: str, cfg: Dict):
        # Unique
        self.skel_id: str = skel_id #kwargs.get(kwargs['skel_id'])
        self.name: str = name
        # Categorical
        self.cartridge: str = name[2:4]  # which ommatidia this is from, from neuron_name, not 'ommatidia_XY' annotation
        self.subtype = assign_exclusive_subtype(self.skel_id, cfg)  # check cfg to decide if categories are excl
        # nodes and connectors to be included or excluded from the analysis
        self.node_list, self.r_node_list = nodes_between_tags(skel_id, cfg, invert=True, both=True)
        self.out_cx, self.r_cx = cx_in_skel(skel_id, cfg, r_nodes=self.r_node_list)


def parse_skeleton_data(skel_ids, neuron_names, cfg: Dict) -> Dict:
    """
    Parse data of skeletons associated with Catmaid skeletal annotation defined in config file
    :param cfg: Analysis configs
    :returns skel_data: Dict hashed by skel_id containing name, ommatidia 'om', subtype 'st', outgoing connections 'out
    cx', restricted nodes and connectors 'r_nodes', 'r_connectors'
    :returns ref:
    """
    # Catmaid Access

    skel_ids, neuron_names = skels_in_annot(cfg['annot'], cfg)
    print(f"Found {len(skel_ids)} skeletons annotated with {cfg['annot']}")

    skel_data = dict.fromkeys(skel_ids)

    for s, n in tqdm(zip(skel_ids, neuron_names)):
        skel_data[s] = Neurite(s, n, cfg)
    return skel_data

