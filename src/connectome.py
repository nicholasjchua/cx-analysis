from tqdm import tqdm
from src.catmaid_queries import *
from src.utils import *
from src.skeleton import Skeleton
from pandas import to_pickle

class Connectome:

    def __init__(self, cfg):

        self.cfg = cfg
        self.skel_data, \
            self.ids_to_names, \
            self.grouping \
            = self.__fetch_skeletons()
        self.adj_mat = self.assemble_adj_mat()
        self.linkdf = self.assemble_linkdf()
        self.cxdf, self.inter, self.unknowns = self.assemble_cxdf()

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

    def save_linkdf(self, path: str = ""):
        if path == "":
            path = self.cfg.out_dir
        pack_pickle(self.linkdf, path, "linkdf")

    def save_cxdf(self, path: str=""):
        if path == "":
            path = self.cfg.out_dir
        pack_pickle(self.cxdf, path, "cxdf")

    def assemble_linkdf(self) -> pd.DataFrame:
        df_rows = []
        skel_data = self.skel_data

        for pre_id, pre_sk in skel_data.items():
            assert (type(pre_sk) is Skeleton)
            out_links = pre_sk.out_links  # list containing a Dict for each synaptic link
            for l in out_links:
                post_id = l.get('post_skel')
                post_sk = self.skel_data.get(post_id)

                if post_sk is None:  # unidentified neurites (aka fragments)
                    post_name = ''
                    post_type = 'UNKNOWN'
                    post_om = 'UNKNOWN'
                else:
                    post_name = post_sk.name
                    post_type = post_sk.subtype
                    post_om = post_sk.group
                # TODO pd.Category this?
                df_rows.append({'pre_neuron': pre_sk.name,
                                'pre_type': pre_sk.subtype,
                                'pre_om': pre_sk.group,
                                'pre_skel': pre_id,
                                'post_neuron': post_name,
                                'post_type': post_type,
                                'post_om': post_om,
                                'post_skel': post_id,
                                'link_id': l.get('link_id'),
                                'cx_id': l.get('cx_id')})

        df = pd.DataFrame(data=df_rows, columns=['link_id', 'cx_id',
                                                 'pre_neuron', 'pre_om', 'pre_type', 'pre_skel',
                                                 'post_neuron', 'post_om', 'post_type', 'post_skel'])
        return df

    def assemble_cxdf(self):
        cx_types = [f"{pre}->{post}"
                    for pre, post in itertools.product(self.cfg.subtypes, self.cfg.subtypes)]

        om_list = sorted([str(k) for k in self.grouping.keys()])

        counts = np.zeros((len(om_list), len(cx_types)), dtype=int)
        inter = []
        unknowns = []

        for ind, row in self.linkdf.iterrows():
            this_pre, this_post = (row['pre_type'], row['post_type'])
            if this_pre.upper() == 'UNKNOWN' or this_post.upper() == 'UNKNOWN':
                unknowns.append(row)
            elif row['pre_om'] != row['post_om']:
                inter.append(row)
            else:
                j = cx_types.index(f"{this_pre}->{this_post}")
                i = om_list.index(row['pre_om'])
                counts[i, j] += 1

        om_order = np.array([[om] * len(cx_types) for om in om_list]).reshape(-1)
        cx_order = np.tile(cx_types, len(om_list))
        print(f"om_order: {om_order.shape}, cx_order: {cx_order.shape}")
        pre_order, post_order = np.array([cx.split("->") for cx in cx_order]).T
        print(f"pre_order: {pre_order.shape}, post_order: {post_order.shape}")

        df = pd.DataFrame({'om': pd.Categorical(om_order),
                           'cx_type': pd.Categorical(cx_order),
                           'pre_type': pd.Categorical(pre_order),
                           'post_type': pd.Categorical(post_order),
                           'n_connect': np.ravel(counts)})
        df.loc[df['n_connect'] < 0, 'n_connect'] = np.nan

        return df, inter, unknowns

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
            skel_data[id] = Skeleton(id, n, g, self.cfg)
            #print(skel_data[id].name)
            #print(skel_data[id].group)
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

    def __count_connections(self, pre_id: str, post_id: str) -> int:

        count = 0
        for l in self.skel_data.get(pre_id).out_links:
            if l['post_skel'] == post_id:
                count += 1
        return count












