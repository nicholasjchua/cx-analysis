import json
from typing import Dict, Tuple
from vis.fig_tools import hex_to_rgb


def subtype_cm(fp: str="vis/lamina_palette.json", rgb: bool=False, include_spr=True) -> Dict:
    """
    Get map of 'subtype' to 'hex colour' from json file
    :param fp: str, File containing colormap
    :param rgb: bool, Returns a Tuple of (R,G,B) instead of hexadecimal
    :param include_spr: Include individual short photoreceptors as keys, e.g. 'R1' in addition to 'R1R4'
    :return palette: Dict, with either hex or tuples as values
    """
    with open(fp) as f:
        p = json.load(f)
    
    if include_spr:
        p.update({'R1': p['R1R4'], 'R4': p['R1R4'],
                        'R2': p['R2R5'], 'R5': p['R2R5'],
                        'R3': p['R3R6'], 'R6': p['R3R6']})
    if rgb:
        return {k: hex_to_rgb(v) for k, v in p.items()}
    else:
        return p


