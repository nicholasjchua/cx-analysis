import numpy as np
from datetime import datetime

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