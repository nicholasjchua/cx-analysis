from typing import Union, Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def hex_to_rgb(hex: str) -> Tuple:
    h = hex.lstrip('#')
    assert(len(h) == 6)

    tup = tuple(int(h[i:i+2], 16) / 256.0 for i in (0, 2, 4))
    return tup


def linear_cmap(n_vals: int, max_colour: Union[Tuple, str], min_colour: Union[Tuple, str] = (1.0, 1.0, 1.0),
                mp_colour: Union[Tuple, str]=None) -> LinearSegmentedColormap:
    if mp_colour is None:
        clist = np.array([min_colour, max_colour])
    else:
        clist = np.array([min_colour, mp_colour, max_colour])

    cm = LinearSegmentedColormap.from_list('mycm', clist, N=n_vals)
    return cm


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

def anatomical_axes(x: float, y: float, arrow_len: float, x_text: str="anterior", y_text: str="dorsal") -> None:
    """
    TODO: Use ax.annotate instead so that a predefined Axes object can be passed into the func
    :param x:
    :param y:
    :param arrow_len:
    :param x_text:
    :param y_text:
    :return None:
    """
    plt.arrow(x, y, arrow_len, 0.0, fc='k', ec='k', head_width=0.03, head_length=0.05)
    plt.text(x + arrow_len + 0.1, y, x_text, fontsize=12)
    plt.arrow(x, y, 0.0, arrow_len, fc='k', ec='k', head_width=0.03, head_length=0.05)
    plt.text(x, y + arrow_len + 0.1, y_text, fontsize=12)

def midpoint_2d(point1: Union[List, Tuple], point2: Union[List, Tuple]) -> Tuple:
    """
    Get the midpoint of two points in 2D space
    :param point1:
    :param point2:
    :return mp: tuple of 2D coordinates
    """
    return (point1[0] + point2[0])/2.0, (point1[1] + point2[1])/2.0