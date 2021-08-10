#!/usr/bin/env python

from cx_analysis.catmaid_queries import *
from cx_analysis.config import Config
from cx_analysis.node_ops import nodes_betwixt


class Skeleton:

    def __init__(self, skel_id: str, name: str, group: str, cfg: Config):

        self.cfg: Config = cfg
        self.skel_id: str = skel_id
        self.name: str = name
        self.group: str = group
        self.cartridge: str = name[2:4]  # TODO: not used?
        self.cell_type: str = self.__assign_cell_type()  # check cfg to decide if categories are excl
        self.skel_nodes: List = skel_compact_detail(skel_id, cfg)

        if self.cell_type in cfg.restrict['restrict_for']:  # TODO: restrict should be logged
            # print(f"{self.name}: RESTRICT")
            self.r_nodes: List = nodes_betwixt(skel_id, cfg, cfg.restrict['restrict_tags'], invert=True)
            # print(f"{len(self.r_nodes)} nodes are in the restricted region")
        else:
            self.r_nodes: List = []
        self.out_cx, self.out_links, self.r_cx = cx_in_skel(skel_id, cfg, r_nodes=self.r_nodes)

    def __assign_cell_type(self, allow_nulltype: bool = False) -> str:
        """
        Assign each skeleton with one of the cell_types listed in the config file
        These are exclusive categories, i.e. a skeleton cannot have more than one celltype
        :param allow_nulltype: Allow skeletons to have no cell type, not currently used
        :return cell_type: a string from the list of cell types
        """
        excl_categories = set(self.cfg.cell_types)
        annotations_found = set(annot_in_skel(self.skel_id, self.cfg))

        intersect = list(excl_categories & annotations_found)
        if len(intersect) > 1:
            raise Exception(f"Skeleton {self.skel_id} cell_type ambiguous: {intersect}")
        elif len(intersect) == 0:
            if allow_nulltype:
                return ''
            else:
                raise Exception(f"Skeleton {self.name} does not belong to any category in cat_list")
        else:
            return str(intersect[0])
