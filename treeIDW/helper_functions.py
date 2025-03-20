import numpy as np
import numpy.typing as npt
import numba as nb

from treeIDW.weight_function import compute_weight

@nb.njit(cache=True)
def inv_dist_weight(boundary_nodes: npt.NDArray[np.float64], 
                    boundary_field: npt.NDArray[np.float64], 
                    internal_nodes: npt.NDArray[np.float64], 
                    relevant_nodes_inds_flat: npt.NDArray[np.int32], 
                    relevant_nodes_inds_sizes: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    """
    Performs the Inverse Distance Weighting interpolation method using only provided boundary nodes
    indices in the weighted sum. This function is designed to be used after selecting the relevant nodes 
    through a KD-tree method.

    Parameters
    ----------
    boundary_nodes : npt.NDArray[np.float64]
        Boundary nodes where the data is known.
        Must be of shape (`n_interp`, `space_dim`).
    boundary_field : npt.NDArray[np.float64]
        Known data.
        Must be of shape (`n_interp`, `field_dim`).
    internal_nodes : npt.NDArray[np.float64]
        Internal nodes where the interpolator is evaluated.
        Must be of shape (`n_eval`, `space_dim`).
    relevant_nodes_inds_flat : npt.NDArray[np.int32]
        Boundary nodes indices that weight in the IDW interpolator for each internal node.
        Must contain `n_eval` concatenated lists of indices between 0 and `n_interp` excluded.
    relevant_nodes_inds_sizes : npt.NDArray[np.int32]
        Sizes of each of the `n_eval` lists concatenated in `relevant_nodes_inds_flat`.

    Returns
    -------
    internal_field : npt.NDArray[np.float64]
        Interpolated data.
        Should be of shape (`n_eval`, `field_dim`).
    """
    field_dim = boundary_field.shape[1]
    n_eval = internal_nodes.shape[0]
    internal_field = np.zeros((n_eval, field_dim), dtype='float')
    offsets_right = np.cumsum(relevant_nodes_inds_sizes)
    offset_left = 0
    for i in range(n_eval):
        offset_right = offsets_right[i]
        inds = relevant_nodes_inds_flat[offset_left:offset_right]
        internal_node = internal_nodes[i]
        weights = []
        for ind in inds:
            boundary_node = boundary_nodes[ind]
            vector = internal_node - boundary_node
            dist_squared = (vector*vector).sum()
            weight = compute_weight(dist_squared)
            weights.append(weight)
        total_weight = sum(weights)
        for weight, ind in zip(weights, inds):
            boundary_field_value = boundary_field[ind]
            internal_field[i] += (weight/total_weight)*boundary_field_value
        offset_left = offset_right
    return internal_field

@nb.njit(parallel=True, cache=True)
def inv_dist_weight_parallel(boundary_nodes: npt.NDArray[np.float64], 
                    boundary_field: npt.NDArray[np.float64], 
                    internal_nodes: npt.NDArray[np.float64], 
                    relevant_nodes_inds_flat: npt.NDArray[np.int32], 
                    relevant_nodes_inds_sizes: npt.NDArray[np.int32]) -> npt.NDArray[np.float64]:
    """
    Performs the Inverse Distance Weighting interpolation method using only provided boundary nodes
    indices in the weighted sum. This function is designed to be used after selecting the relevant nodes 
    through a KD-tree method.

    Parameters
    ----------
    boundary_nodes : npt.NDArray[np.float64]
        Boundary nodes where the data is known.
        Must be of shape (`n_interp`, `space_dim`).
    boundary_field : npt.NDArray[np.float64]
        Known data.
        Must be of shape (`n_interp`, `field_dim`).
    internal_nodes : npt.NDArray[np.float64]
        Internal nodes where the interpolator is evaluated.
        Must be of shape (`n_eval`, `space_dim`).
    relevant_nodes_inds_flat : npt.NDArray[np.int32]
        Boundary nodes indices that weight in the IDW interpolator for each internal node.
        Must contain `n_eval` concatenated lists of indices between 0 and `n_interp` excluded.
    relevant_nodes_inds_sizes : npt.NDArray[np.int32]
        Sizes of each of the `n_eval` lists concatenated in `relevant_nodes_inds_flat`.

    Returns
    -------
    internal_field : npt.NDArray[np.float64]
        Interpolated data.
        Should be of shape (`n_eval`, `field_dim`).
    """
    field_dim = boundary_field.shape[1]
    n_eval = internal_nodes.shape[0]
    internal_field = np.zeros((n_eval, field_dim), dtype='float')
    offsets_right = np.cumsum(relevant_nodes_inds_sizes)
    offset_left = 0
    for i in nb.prange(n_eval):
        offset_right = offsets_right[i]
        inds = relevant_nodes_inds_flat[offset_left:offset_right]
        internal_node = internal_nodes[i]
        weights = []
        for ind in inds:
            boundary_node = boundary_nodes[ind]
            vector = internal_node - boundary_node
            dist_squared = (vector*vector).sum()
            weight = compute_weight(dist_squared)
            weights.append(weight)
        total_weight = sum(weights)
        for weight, ind in zip(weights, inds):
            boundary_field_value = boundary_field[ind]
            internal_field[i] += (weight/total_weight)*boundary_field_value
        offset_left = offset_right
    return internal_field

@nb.njit(cache=True)
def bisect_weight_elem(dist_squared_a: float, dist_squared_b: float, weight_treshold: float, 
                  rtol: float) -> float:
    """
    Compute the squared distance that correspond to a specific weight threshold using a bisection approach.

    Parameters
    ----------
    dist_squared_a : float
        Left boundary of the interval containning the looked for weight treshold preimage.
    dist_squared_b : float
        Right boundary of the interval containning the looked for weight treshold preimage.
    weight_treshold : float
        Image of the looked for squared distance by the weight function of the IDW algorithm.
    rtol : float, optional
        The relative tolerance is used to compute the necessary number of bisection iterations, by default 1e-3

    Returns
    -------
    dist_squared_c : float
        Weight treshold preimage found through bisection.
    """
    weight = lambda dist_squared: compute_weight(dist_squared) - weight_treshold
    weight_a, weight_b = weight(dist_squared_a), weight(dist_squared_b)
    if (weight_a*weight_b)>=0:
        if weight_a<=0:
            return dist_squared_a*(1 + 1e-5)
        elif weight_b>=0:
            return dist_squared_b*(1 + 1e-5)
    nb_iter = int(np.log2((dist_squared_b - dist_squared_a)/(rtol*dist_squared_a)) + 1)
    if nb_iter<=0:
        return 0.5*(dist_squared_a + dist_squared_b)
    for _ in range(nb_iter):
        dist_squared_c = 0.5*(dist_squared_a + dist_squared_b)
        weight_c = weight(dist_squared_c)
        if (weight_a*weight_c)<0:
            dist_squared_b, weight_b = dist_squared_c, weight_c
        else:
            dist_squared_a, weight_a = dist_squared_c, weight_c
    return dist_squared_c

@nb.guvectorize(['void(float64, float64, float64, float64, float64[:])'], '(),(),(),()->()', nopython=True, cache=True)
def bisect_weight(dist_squared_a, dist_squared_b, weight_treshold, rtol, out):
    """
    Vectorized version of bisect_weight_elem that finds squared distances corresponding to weight thresholds.

    Parameters
    ----------
    dist_squared_a : float
        Left boundary of interval containing weight threshold preimage
    dist_squared_b : float
        Right boundary of interval containing weight threshold preimage  
    weight_treshold : float
        Target weight threshold value
    rtol : float
        Relative tolerance for bisection convergence
    out : ndarray
        Output array for storing computed squared distance
    """
    out[:] = bisect_weight_elem(dist_squared_a, dist_squared_b, weight_treshold, rtol)

@nb.guvectorize(['void(float64, float64, float64, float64, float64[:])'], '(),(),(),()->()', target='parallel', nopython=True, cache=True)
def bisect_weight_parallel(dist_squared_a, dist_squared_b, weight_treshold, rtol, out):
    """
    Vectorized version of bisect_weight_elem that finds squared distances corresponding to weight thresholds.
    Parallelized version of the function.

    Parameters
    ----------
    dist_squared_a : float
        Left boundary of interval containing weight threshold preimage
    dist_squared_b : float
        Right boundary of interval containing weight threshold preimage  
    weight_treshold : float
        Target weight threshold value
    rtol : float
        Relative tolerance for bisection convergence
    out : ndarray
        Output array for storing computed squared distance
    """
    out[:] = bisect_weight_elem(dist_squared_a, dist_squared_b, weight_treshold, rtol)

@nb.guvectorize(['void(float64, float64[:])'], '()->()', nopython=True, cache=True)
def compute_weight_vectorized(dist_squared, out):
    """
    Vectorized version of compute_weight function for IDW calculations.

    Parameters
    ----------
    dist_squared : float
        Squared distance between interior and boundary nodes.
    out : ndarray
        Output array containing the computed IDW weights.
    """
    out[:] = compute_weight(dist_squared)

@nb.guvectorize(['void(float64, float64[:])'], '()->()', target='parallel', nopython=True, cache=True)
def compute_weight_vectorized_parallel(dist_squared, out):
    """
    Vectorized version of compute_weight function for IDW calculations.
    Parallelized version of the function.

    Parameters
    ----------
    dist_squared : float
        Squared distance between interior and boundary nodes.
    out : ndarray
        Output array containing the computed IDW weights.
    """
    out[:] = compute_weight(dist_squared)