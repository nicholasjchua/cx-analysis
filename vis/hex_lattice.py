from typing import Dict, Tuple, Union, List
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import networkx as nx
from pprint import pprint

from vis.fig_tools import scale_distance_between

def hexplot(node_data: Dict, edge_data: Dict=None, ax: plt.Axes=None,
            scale_factor: float=0.5, n_rgba: Tuple=(0.8, 0.8, 0.8, 0.5),
            e_colour='r'):
    """
    Plot a map of the hexagonal lattice of the megaphragma compound eye. Each ommatidium will be labelled and coloured
    according to the strings and (r, g, b, a) values passed with node_data.
    :param node_data: Dict, {om: {'label': str,
                                 {'outline': matplotlib line spec, e.g. '-',
                                 {'colour':  (rgba)}}
    :param edge_data:
    :param ax: plt.Axes, if None, will plot to current Axis
    :param scale_factor: float, controls the spacing between nodes
    :param n_rgba: Default node colour (if node_data: colours is None)
    :param e_colour: Default edge colour (if edge_data is used)
    :return:
    """
    G, pos = generate_lattice()
    nx.set_node_attributes(G, pos, name='pos')
    # Handle labels/data
    node_colours = []
    node_outline = []
    node_labels = {}
    name_to_ind = {}
    for nx_ind, data in G.nodes(data=True):
        this_om = get_ret_coords(data['pos'])
        name_to_ind.update({this_om: tuple(nx_ind)})

        nd = node_data.get(this_om, {})
        node_colours.append(nd.get('colour', n_rgba))
        node_outline.append(nd.get('outline', '-'))
        node_labels.update({nx_ind: nd.get('label', this_om)})

    if ax is None:
        ax = plt.gca()

    pos = scale_distance_between(pos, scale_factor)

    nx.draw(G, pos, alpha=1.0, edge_list=[], node_color=node_colours, node_size=1400*4,
            node_shape='H', linewidth=5.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, edge_list=[], font_size=14, ax=ax)



def generate_lattice() -> Tuple:
    """
    hex_lattice
    :param spacing: float, factor that scales the distance between points in pos
    :return: networkx Graph, with nodes representing each ommatidia in *Megaphragma*
    :return: pos, Dict of 2D coordinates for plotting (if scaling != 1.0, these will be different from the coordinates
             in G)
    """
    G = nx.generators.triangular_lattice_graph(4, 10)
    G.remove_node((0, 4))  # delete the most ventro-medial
    pos = nx.get_node_attributes(G, name='pos')  # a dict of node_ref: fig_coordinates

    # Rotate 270 so that vertical axis is anatomical ventral -> dorsal, horizontal is medial -> lateral
    for om in pos.keys():
        tmp = pos[om]
        pos[om] = ((tmp[1] * -1), tmp[0])
        pos[om] = (pos[om][0] + float(4.0 * (np.sqrt(3.0) / 2.0)), pos[om][1] + 0.5)
        if om in [(0, 2), (0, 3), (1, 4)]:  # some nodes need to be repositioned further
            pos[om] = (pos[om][0] + float(np.sqrt(3) / 2), pos[om][1] - 0.5)
    edges = list(G.edges())
    for e in edges:
        G.remove_edge(e[0], e[1])
    #G = nx.set_node_attributes(G, pos, name='pos')

    return G, pos


def get_ret_coords(position: Tuple):
    """
    Convert figure coordinates to a letter-digit string corresponding to the eye coordinates of an ommatidia
    :param position: 2-tuple of floats indicating the ommatidia's figure coordinates
    :return col_row: str, e.g. 'B2'
    """
    col_num = np.rint(4 - position[0]/(np.sqrt(3.0)/2.0))  # A is most right, so this evaluates to 0
    row_num = int(np.floor(position[1]) + col_num/2.0)
    col_letter = chr(int(col_num) + 65)
    return str(col_letter) + str(row_num)