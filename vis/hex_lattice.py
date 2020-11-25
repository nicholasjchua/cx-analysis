from typing import Dict, Tuple, Union, List, Iterable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import networkx as nx
from pprint import pprint

def hexplot(node_data: Union[Dict, pd.DataFrame], decimals: int=0, 
            node_lim: Iterable=None, c: Iterable=None, lc: str='k',
            edge_data: Dict=None, edge_c='r',  # EDGE DATA NOT IMPLEMENTED YET
            ax: plt.Axes=None, scale_factor: float=0.015):
    """
    Plot a map of the hexagonal lattice of the megaphragma compound eye. Each ommatidium will be labelled and coloured
    according to the strings and (r, g, b, a) values passed with node_data. This will also take a 1-col dataframe indexed by om
    TODO: fix dataframe input options, type of cmap as an arg, use plt.scatter instead of all the networkx functions? 
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
    if c is None: # default color
        c = (0.2, 0.2, 0.2, 1)
        
    if isinstance(node_data, pd.DataFrame):
        node_data = __from_pandas(node_data, c=c, node_lim=node_lim)
    
    G, pos = generate_lattice()
    nx.set_node_attributes(G, pos, name='pos')
    # Handle labels/data
    node_colours = []
    node_outline = []
    node_labels = {}
    name_to_ind = {}
    for nx_ind, data in G.nodes(data=True):
        this_om = hex_to_om(data['pos'])
        name_to_ind.update({this_om: tuple(nx_ind)})
        nd = node_data.get(this_om, {})
        label = nd.get('label', this_om)
        
        node_colours.append(nd.get('colour', c))
        node_outline.append(nd.get('outline', '-'))
        
        node_labels.update({nx_ind: nd.get('label', this_om)})

    if ax is None:
        ax = plt.gca()

    pos = scale_distance_between(pos, scale_factor)

    nx.draw(G, pos, alpha=1.0, edge_list=[], node_color=node_colours, node_size=scale_factor * 18000,
            node_shape='H', linewidth=1.0, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, edge_list=[], font_size=5, font_color=lc, ax=ax)
#     ax.set_xmargin(0)
#     ax.set_ymargin(0)
    ax.set_aspect('equal')
    
    return ax


def hexplot_TEST(node_data, decimals: int=None, 
            node_lim: Iterable=None, c: object=None, lc: str='k',
            edge_data: Dict=None, edge_c='r',  # EDGE DATA NOT IMPLEMENTED YET
            ax: plt.Axes=None, scale_factor: float=0.015):
    """
    Plot a map of the hexagonal lattice of the megaphragma compound eye. Each ommatidium will be labelled and coloured
    according to the strings and (r, g, b, a) values passed with node_data. This will also take a 1-col dataframe indexed by om
    TODO: fix dataframe input options, type of cmap as an arg, use plt.scatter instead of all the networkx functions? 
    :param node_data: Dict, {om: {'label': str,
                                 {'outline': matplotlib line spec, e.g. '-',
                                 {'colour':  (rgba)}}
    :param edge_data:
    :param ax: plt.Axes, if None, will plot to current Axis
    :param scale_factor: float, controls the spacing between nodes
    :c: Default node colour (if node_data: colours is None)
    :param e_colour: Default edge colour (if edge_data is used)
    :return:
    """
    if c == None: # default color
        default_c = (0.2, 0.2, 0.2, 1)
    else:
        default_c = c
        
    if isinstance(node_data, pd.DataFrame):
        node_data = __from_pandas(node_data, c=c, node_lim=node_lim)
        
    if ax == None:
        ax = plt.gca()
        
    om_list = sorted([str(om) for om in node_data.keys()])
    pos = [om_to_hex(o) for o in om_list] # 2D figure coords of each om
    node_colours = []#dict.fromkeys(om_list)
    node_outline = []#dict.fromkeys(om_list)
    node_labels = []#dict.fromkeys(om_list)
        
    #name_to_ind = 
    for om, xy in zip(om_list, pos):
        
        if node_data[om].get('label') == None:
            label = om
        elif type(node_data.get('label')) == float and decimal != None:
            label = str(round(label, decimals))
        else:
            label = str(label)
                        
        if (node_data[om].get('colour') == None):
            fill_c = default_c
        else:
            fill_c = node_data[om].get('colour')
            
        x = xy[0] * 0.01
        y = xy[1] * 0.01
        
        ax.scatter(xy[0], xy[1], marker='H', c=fill_c, s=100)
        ax.annotate(label, xy, fontsize=8, color='w', ha='center', va='center')
        
        
    #ax.set_xlim((-30, 4))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #plt.axis('off')
    ax.set_aspect('equal')
    
    return ax




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


def __from_pandas(X: object, c: Iterable, node_lim: Iterable=None, cmap_center: str=None) -> Dict:
    """
    __from_pandas
    This function is called when hexplot() receives a pandas DataFrame instead of 
    a dict for node_data. Right now, the df has to be indexed by 'om'. 
    The different options for the color transfer functions need to be tested
    """
    
    from matplotlib.colors import LinearSegmentedColormap
    
    if len(X.columns) == 1:
        var_name = X.columns[0]
    else:
        raise Warning('DataFrame has too many variable columns, idk which to plot')
    
    if cmap_center is None and X.min().values >= 0:
        cm = LinearSegmentedColormap.from_list(name='mycm', colors=[(1.0, 1.0, 1.0), c])
        if node_lim is None:
            max_val = X.max()
        else:
            max_val = node_lim[1]
        trans_vals = (X - X.min()) / (max_val - X.min())  # 0 - 1
    ### NOT TESTED ###
    elif cmap_center is None:
        cm = LinearSegmentedColormap.from_list(name='mycm', colors=[c[0], (1.0, 1.0, 1.0), c[1]])
        abs_max = X.abs().max()
        trans_vals = ((X / abs_max) + 1)/2
    elif (cmap_center in ['mean', 'average']):
        cm = LinearSegmentedColormap.from_list(name='mycm', colors=[c[0], (1.0, 1.0, 1.0), c[1]])
        abs_max = X.abs().max()
        trans_vals = (X/X.mean()) - 0.5
    ### NOT TESTED ###
    else:
        raise Warning('Something went wrong')
        
    node_data = dict()
    for om, val in X.iterrows():
        this_node = {'colour': cm(trans_vals.loc[om, var_name]),
                     'label': f"{val[var_name]: .2f}"}
        node_data[om] = this_node
            
    return node_data


def hex_to_om(xy: Iterable) -> str:
    """
    Convert a set of figure coordinates to a letter-digit ommatidia ID
    :param position: 2-tuple of floats indicating the ommatidia's figure coordinates
    :return col_row: str, e.g. 'B2'
    """
    assert(len(xy) == 2)
    col_num = np.rint(4 - (2.0/np.sqrt(3.0) * xy[0]))  # A is most anterior, evaluates to 0
    row_num = int(np.floor(xy[1]) + col_num/2.0)
    col_letter = chr(int(col_num) + 65)
    return str(col_letter) + str(row_num)


def om_to_hex(om: str) -> Tuple:
    """
    Convert letter-digit ommatidia ID (e.g. 'A4') to figure coordinates
    :param position: 2-tuple of floats indicating the ommatidia's figure coordinates
    :return col_row: str, e.g. 'B2'
    """
    assert(len(om) == 2)
    x = np.sqrt(3.0)/2.0 * (ord(om[0]) - 65.0) + 4.0
    y = int(om[1]) - (0.5 * ord(om[0]))
    return x, y


def scale_distance_between(positions, scalar):
    """
    Takes a dict of figure coordinates and scales the distance between them
    :param positions: dict(node_ref: (x, y))
    :param scalar: distances between nodes is multiplied by this number
    :return: positions: the results of the scaling operation
    """
    for this_node, this_pos in positions.items():
        positions[this_node] = (this_pos[0] * scalar, this_pos[1] * scalar)
    return positions

