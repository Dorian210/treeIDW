import numpy as np
import numpy.typing as npt
import numba as nb

@nb.njit(cache=True)
def compute_weight(dist_squared: float) -> float:
    """
    Compute the IDW weight. Can be changed but is usually `1/dist_squared`.

    Parameters
    ----------
    dist_squared : float
        Squared distance between the interior node and the boundary node.

    Returns
    -------
    weight : float
        IDW weight.
    """
    weight = 1/dist_squared
    return weight