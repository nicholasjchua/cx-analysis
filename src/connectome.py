from tqdm import tqdm
from src.catmaid_queries import *
from src.utils import *
from src.skeleton import Skeleton
from src.dataframe_tools import assemble_linkdf, assemble_cxdf
from pandas import to_pickle
"""
connectome.py
Connectome is a class representing a collection of neurons and the synaptic connections between them.
A Connectome is initialized with a config file that details how the neurons should be categorized, the 
number of neurons associated with each subtype or group, along with other options related to filtering
or representing this data. Connectome will initialize a Skeleton object for each neuron, which perform
the API requests to Catmaid. The Connectome object can be saved as a pkl file. This can also call 
methods in dataframe_tools.py to assemble/save various summaries of connectivity data.   
"""
class Connectome:

    def __init__(self, cfg):

        self.cfg = cfg
        self.skel_data, \
        self.ids_to_names, \
        self.grouping = self.__fetch_skeletons()
        self.adj_mat = self.assemble_adj_mat()

        self.linkdf, self.cxdf, \
        self.inter, self.unknowns = self.assemble_dataframes()

    def print_adj_mat(self):
        # TODO get the stuff that formats and prints adjacency matrices from 'connectivity_analysis'
        A = np.ones((self.adj_mat.shape[0], len(self.cfg.subtypes), len(self.cfg.subtypes)), dtype=int)

    def save_connectome(self, path: str = "", overwrite: bool=False) -> None:
        """
        save Connectome instance as .pickle file
        """
        if path == "":
            path = self.cfg.out_dir
        pack_pickle(self, path, "preprocessed")

    def assemble_dataframes(self, save=True) -> Tuple:

        link_df = assemble_linkdf(self)
        cx_df, inter, unknowns = assemble_cxdf(self, link_df)


        return link_df, cx_df, inter, unknowns

    def save_linkdf(self, path: str = ""):
        if path == "":
            path = self.cfg.out_dir

        pack_pickle(self.linkdf, path, "linkdf")

    def save_cxdf(self, path: str=""):
        if path == "":
            path = self.cfg.out_dir
        pack_pickle(self.cxdf, path, "cxdf")



    def query_ids_by(self, by: str, key: str):
        """
        'by' can be 'group' or 'subtype'
        """
        if by == 'group' or by.lower() == 'g':
            return [skel_id for skel_id, data in self.skel_data.items() if data.group == key]

        elif by == 'subtype' or by.lower() == 's':
            return [skel_id for skel_id, data in self.skel_data.items() if data.subtype == key]
        else:
            raise Exception("Argument for 'by' needs to be either 'group' or 'subtype'")

    def assemble_adj_mat(self):
        id_mat = self.__get_id_mat()
        groups = sorted(self.grouping.keys())
        subtypes = sorted(self.cfg.subtypes)

        adj_mat = np.zeros((len(groups), id_mat.shape[1], id_mat.shape[1]), dtype=int)

        for i, g in enumerate(groups):
            for j, pre_skel in enumerate(id_mat[i]):
                if pre_skel == '-1':  # cartridges with missing neurons coded with -1 (only allowed for L4)
                    #print(f'PRESKEL is -1')
                    adj_mat[i, j, :] = -1
                    continue

                for k, post_skel in enumerate(id_mat[i]):
                    if post_skel == '-1':
                        adj_mat[i, j, k] = -1
                    else:
                        adj_mat[i, j, k] = self.__count_connections(pre_skel, post_skel)
        print(adj_mat)
        return adj_mat

    # Private Methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __fetch_skeletons(self) -> Tuple:
        """
        Parse skeletons associated with annotation defined in config file
        :returns skel_data: {id: skeleton_data}
        :returns neurons_ids: [(neuron_name, skel_id)]
        """
        # Catmaid Access

        skel_ids, neuron_names = skels_in_annot(self.cfg.annot, self.cfg)
        ids_to_names = {s: n for s, n in zip(skel_ids, neuron_names)}
        grouping, ids_to_groups = self.__group_neurons(ids_to_names)
        print(f"Found {len(skel_ids)} skeletons annotated with {self.cfg.annot}")

        skel_data = dict.fromkeys(skel_ids)

        for id, n in tqdm(ids_to_names.items()):
            g = ids_to_groups[id]
            ### Skeleton object instantiated here ###
            skel_data[id] = Skeleton(id, n, g, self.cfg)
        return skel_data, ids_to_names, grouping

    def __group_neurons(self, ids_to_names: Dict):
        """
        Use each neuron's names to form groups (in this case by ommatidia)
        :param ids_to_names:
        :return groups: Dict, group: [list of skel_ids]
        :return ids_to_groups: Dict, skel_id: group
        """

        groups = dict()  # {grouping: [skel_ids]}
        ids_to_groups = dict.fromkeys(list(ids_to_names.keys()))

        for id, n in ids_to_names.items():
            # neurons should be named omC2_[annotator initials]_[subtype]
            if self.cfg.groupby is 'annotator':
                g_flag = n.split('_')[-1]
                if g_flag not in self.cfg.annotator_initials:
                    raise Exception(f"Could not find annotator initials in the neuron's name: {n}")
            # neurons should be named 'om[two char id]_[subtype]'
            elif self.cfg.groupby is 'om':
                g_flag = n[2:4]
                if n[0:2] != 'om':
                    raise Exception(f"Could not assign {n} to an ommatidia based on its neuron_name")
            else:
                raise Exception("Invalid 'groupby' argument. Needs to be 'annotator' or 'om'")

            groups.setdefault(g_flag, [id]).append(id)
            ids_to_groups[id] = g_flag

        return groups, ids_to_groups

    def __get_id_mat(self) -> np.array:

        group_list = sorted(self.grouping.keys())
        subtypes = sorted(self.cfg.subtypes)
        ids = []
        for i, g in enumerate(group_list):
            skels_in_g = self.query_ids_by('group', g)
            tmp = []  # the row for each group
            for ii, s in enumerate(subtypes):
                skels_in_s_and_g = [skel for skel in skels_in_g if skel in self.query_ids_by('subtype', s)]
                if len(skels_in_s_and_g) == abs(self.cfg.expected_n[ii]):
                    tmp = [*tmp, *skels_in_s_and_g]
                elif len(skels_in_s_and_g) == 0 and self.cfg.expected_n[ii] == -1:
                    tmp.append('-1')
                    print(f'Warning: No neuron of type {s} found in {g}')
                else:
                    raise Exception(f"Unexpected number of neurons for group: {g} subtype: {s}."
                                    f"Got the following ids: \n{skels_in_s_and_g}")

            ids.append(tmp)
        ids = np.array(ids, dtype=str)
        return ids



    def __count_connections(self, pre_id: str, post_id: str) -> int:

        count = 0
        for l in self.skel_data.get(pre_id).out_links:
            if l['post_skel'] == post_id:
                count += 1
        return count












