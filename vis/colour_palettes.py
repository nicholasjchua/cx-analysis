import json
from typing import Dict, Tuple

def hex_to_rgb(hex: str) -> Tuple:
    h = hex.lstrip('#')
    assert(len(h) == 6)
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def subtype_cm(fp: str="vis/lamina_subtype_cm.json", rgb: bool=False) -> Dict:
    """
    Get map of 'subtype' to 'hex colour' from json file
    :param fp: str, File containing colormap
    :param rgb: bool, Returns a Tuple of (R,G,B) instead of hexadecimal
    :return palette: Dict, with either hex or tuples as values
    """
    with open(fp) as f:
        palette = json.load(f)
        if rgb:
            return {k: hex_to_rgb(v) for k, v in palette.items()}
        else:
            return palette

