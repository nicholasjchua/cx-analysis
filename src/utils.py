import numpy as np
import os.path
import pickle
from glob import glob
from datetime import datetime
import re
from typing import Dict, List
import pandas as pd

#from src.connectome import Connectome
from src.skeleton import Skeleton

# Save / Load ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def unpack_pickle(data_dir: str, f_regex: str):

    fp = glob(os.path.join(os.path.expanduser(data_dir), f_regex))
    if len(fp) > 1:
        raise Exception(f"Found multiple files in {data_dir} matching {f_regex}:\n{fp}")
    elif len(fp) == 0:
        raise Exception(f"Unable to find file in {data_dir} matching {f_regex}")
    else:
        path = fp[0]
        with open(path, 'rb') as fh:
            print(f"Pickle loaded from: {path}")
            data = pickle.load(fh)
        return data


def pack_pickle(data, data_dir: str, f_label: str, overwrite: bool=False):
    fn = f"{yymmdd_today()}_{f_label}.pickle"
    file_path = os.path.join(os.path.expanduser(data_dir), fn)
    if os.path.isfile(file_path):
        raise Exception(f"{file_path} already exists")

    with open(file_path, 'wb') as f:
        print(f"Preprocessed data written to:\n {file_path}")
        if type(data) is pd.DataFrame:
            data.to_pickle(f, compression=None)
        else:
            pickle.dump(data, f)



def load_preprocessed_connectome(data_dir: str):

    # TODO change to pkl
    C = unpack_pickle(data_dir, '*_preprocessed.pickle')
    return C


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






