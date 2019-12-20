from typing import List, Dict, Iterable
from src.catmaid_queries import *
from src.utils import *
from src.skeleton import Skeleton


class Connectome:

    def __init__(self, cfg):

        self.cfg = cfg
        self.skel_data, \
            self.neurons_ids = self.fetch_skeletons()

    def fetch_skeletons(self) -> Tuple:
        """
        Parse skeletons associated with annotation defined in config file
        :returns skel_data: {id: skeleton_data}
        :returns neurons_ids: [(neuron_name, skel_id)]
        """
        # Catmaid Access

        skel_ids, neuron_names = skels_in_annot(self.cfg.annot, self.cfg)
        neuron_ids = zip(skel_ids, neuron_names)
        print(f"Found {len(skel_ids)} skeletons annotated with {self.cfg.annot}")

        skel_data = dict.fromkeys(skel_ids)

        for id, n in neuron_ids:
            skel_data[id] = Skeleton(id, n, self.cfg)
            print(skel_data[id].subtype)

        return skel_data, neuron_ids
