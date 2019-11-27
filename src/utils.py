import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Union

# Attribute assignment ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def assign_exclusive_subtype(skel_id: str, excl_cats: List, cfg: Dict, ignore_classless: bool=False) -> str:
    """
    Searches through the annotations of a skeleton to find a match with one of the st in excl_list
    Raises an exception if multiple matches are found (i.e. neuron can only belong to one of the categories)
    Can ignore instances where none of the categories are found
    :param skel_id: str, Skeleton ID
    :param excl_cats: List, Contains str of annotations you want to use as categories
    :param cfg: Dict, analysis options
    :param ignore_classless: bool, if False (default), will raise an Exception if a neuron doesn't have any annotions
    in cat_list. If True, returns None instead
    :return: str, the skeleton's category, None if no category was found
    """
    categories = set(cfg['subtypes'])
    annotations = set(annot_in_skel(skel_id, cfg))

    intersect = list(categories & annotations)
    if len(intersect) > 1:
        raise Exception(f"Skeleton {skel_id} can be assigned to more than one category: {intersect}")
    elif len(intersect) == 0:
        if ignore_classless:
            return ''
        else:
            raise Exception(f"Skeleton {skel_id} does not belong to any category in cat_list")
    else:
        return intersect[0]

def div0(a: np.array, b: np.array, decimals: int=None) -> np.array:
    """
    Divide numpy arrays a and b, ignoring the /0 RuntimeWarning
    :param a: Numerator numpy array
    :param b: Denominator numpy array
    :param decimals: round elements in result
    :return: result: a numpy array
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide(a, b)
        result[~np.isfinite(result)] = 0
        if decimals is not None:
            result = np.around(result, decimals=decimals)
    return result

def datestamp() -> str:

    return datetime.today().strftime("%y%m%d")