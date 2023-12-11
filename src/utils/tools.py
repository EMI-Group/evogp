import numpy as np
from .enum import FUNCS, FUNCS_NAMES


def dict2cdf(prob_dict):
    assert len(prob_dict) > 0, "Empty probability dictionary"

    cdf = np.zeros(len(FUNCS))

    for key, val in prob_dict.items():
        assert key in FUNCS_NAMES, f"Unknown function name: {key}, total functions are {FUNCS_NAMES}"
        idx = FUNCS_NAMES.index(key)
        cdf[idx] = val

    # normalize
    cdf = cdf / cdf.sum()

    return np.cumsum(cdf)
