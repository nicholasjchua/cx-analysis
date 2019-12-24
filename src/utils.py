import numpy as np
import os.path
import pickle
from datetime import datetime
import re
from src.connectome import Connectome
from typing import Dict, List, Tuple, Union


# Save data and results ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




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

def yymmdd_today() -> str:

    return datetime.today().strftime("%y%m%d")

def handle_dupe_filenames(fn: str):
    """
    adds 'dash-[int]' before the last underscore of a filename that exists already, increasing till it
    gets a name that doesn't exist
    """
    if os.path.isfile(fn):
        assert('_' in str(fn))
        head = fn.split('_')[-2]
        tail = fn.split('_')[-1]
        if re.search("-[0-9]+$", head) is not None:
            n = int(head.split('-')[1]) + 1
            head = head.split('-')[0]  # get rid of -n part
        else:
            n = 0
        new_fp = head + f"-{n}" + tail
        fn = handle_dupe_filenames(new_fp)
    else:
        return fn
    return fn




