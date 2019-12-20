from src.catmaid_queries import *
from src.config import Config


class Skeleton:

    def __init__(self, skel_id: str, name: str, cfg: Config, restrict=False):
        # Unique
        self.skel_id: str = skel_id #kwargs.get(kwargs['skel_id'])
        self.name: str = name
        self.cfg: Config = cfg
        # Categorical
        self.cartridge: str = name[2:4]  # which ommatidia this is from, from neuron_name, not 'ommatidia_XY' annotation
        self.subtype: str = self.which_subtype()  # check cfg to decide if categories are excl
        # nodes and connectors to be included or excluded from the analysis
        if restrict:
            self.node_list, self.r_node_list = nodes_between_tags(skel_id, cfg, invert=True, both=True)
            self.out_cx, self.r_cx = cx_in_skel(skel_id, cfg, r_nodes=self.r_node_list)


    def which_subtype(self, allow_nulltype: bool=False) -> str:

        excl_categories = set(self.cfg.subtypes)
        annotations_found = set(annot_in_skel(self.skel_id, self.cfg))

        intersect = list(excl_categories & annotations_found)
        if len(intersect) > 1:
            raise Exception(f"Skeleton {self.skel_id} subtype ambiguous: {intersect}")
        elif len(intersect) == 0:
            if allow_nulltype:
                return ''
            else:
                raise Exception(f"Skeleton {self.skel_id} does not belong to any category in cat_list")
        else:
            return intersect[0]