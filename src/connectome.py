from typing import List, Dict, Iterable
from src.catmaid_queries import *
from src.utils import *
from src.skeleton import Skeleton
from tqdm import tqdm
from src.config import Config

class Connectome:

    def __init__(self, cfg):
        #if cfg.dir is not '' and glob(os.path.join(cfg.dir, '*preprocessed.pickle')):
            #self.neuron_data, self.neuron_names = load_preprocessed
        self.cfg = cfg
        self.skel_data, \
        self.neurons_ids = self.fetch_skeletons()


    def fetch_skeletons(self) -> Tuple:
        """
        Parse data of skeletons associated with Catmaid skeletal annotation defined in config file
        :param cfg: Analysis configs
        :returns skel_data: Dict hashed by skel_id containing name, ommatidia 'om', subtype 'st', outgoing connections 'out
        cx', restricted nodes and connectors 'r_nodes', 'r_connectors'
        :returns ref:
        """
        # Catmaid Access

        skel_ids, neuron_names = skels_in_annot(self.cfg.annot, self.cfg)
        neurons_ids = zip(skel_ids, neuron_names)
        print(f"Found {len(skel_ids)} skeletons annotated with {self.cfg.annot}")

        skel_data = dict.fromkeys(skel_ids)

        for id, n in neurons_ids:
            skel_data[id] = Skeleton(id, n, self.cfg)
            print(skel_data[id].subtype)

        return skel_data, neurons_ids
