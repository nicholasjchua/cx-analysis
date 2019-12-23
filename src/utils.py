import numpy as np
import os.path
import pickle
from datetime import datetime
from src.connectome import Connectome
from typing import Dict, List, Tuple, Union

# Attribute assignment ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_preprocessed_connectome(c: Connectome) -> None:

    if os.path.exists(c.cfg.out_dir):
        file_path = os.path.join(c.cfg.out_dir, f"{YYMMDDnow()}_preprocessed.pickle")
        with open(file_path, 'wb') as f:
            pickle.dump(c, f)
        print(f"Preprocessed connectome saved at: {file_path}")
    else:
        raise Exception(f"Directory: {c.cfg.out_dir} not found")

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

def YYMMDDnow() -> str:

    return datetime.today().strftime("%y%m%d")