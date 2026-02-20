import numpy as np
from scipy.spatial import KDTree

from treeIDW.helper_functions import (
    compute_weight_vectorized,
    bisect_weight,
    inv_dist_weight,
    compute_weight_vectorized_parallel,
    bisect_weight_parallel,
    inv_dist_weight_parallel,
)


def treeIDW(
    boundary_nodes: np.ndarray[np.floating],
    boundary_field: np.ndarray[np.floating],
    internal_nodes: np.ndarray[np.floating],
    neglectible_treshold: float = 0.2,
    bisect_rtol: float = 1e-3,
    parallel: bool = False,
) -> np.ndarray[np.floating]:
    """
    Performs IDW interpolation using a KD-tree to select relevant boundary nodes for each internal node.
    Only boundary nodes with significant weights are included in the interpolation.

    Parameters
    ----------
    boundary_nodes : np.ndarray[np.floating]
        Boundary nodes where the data is known.
        Must be of shape (`n_interp`, `space_dim`).
    boundary_field : np.ndarray[np.floating]
        Known data.
        Must be of shape (`n_interp`, `field_dim`).
    internal_nodes : np.ndarray[np.floating]
        Internal nodes where the interpolator is evaluated.
        Must be of shape (`n_eval`, `space_dim`).
    neglectible_treshold : float, optional
        Relative weight threshold below which boundary nodes are ignored, by default 0.2
    bisect_rtol : float, optional
        Relative tolerance for bisection convergence, by default 1e-3
    parallel : bool, optional
        Whether to use parallel implementation, by default False

    Returns
    -------
    internal_field : np.ndarray[np.floating]
        Interpolated data.
        Should be of shape (`n_eval`, `field_dim`).
    """
    if parallel:
        compute_weight_vectorized_ = compute_weight_vectorized_parallel
        bisect_weight_ = bisect_weight_parallel
        tree_workers = -1
        inv_dist_weight_ = inv_dist_weight_parallel
    else:
        compute_weight_vectorized_ = compute_weight_vectorized
        bisect_weight_ = bisect_weight
        tree_workers = 1
        inv_dist_weight_ = inv_dist_weight

    tree = KDTree(boundary_nodes)
    d_min_vals, closest_nodes_inds = tree.query(
        internal_nodes, workers=(-1 if parallel else 1)
    )
    d_max = np.linalg.norm(
        np.minimum(boundary_nodes.min(axis=0), internal_nodes.min(axis=0))
        - np.maximum(boundary_nodes.max(axis=0), internal_nodes.max(axis=0))
    )

    vector_min = internal_nodes - boundary_nodes[closest_nodes_inds]
    del closest_nodes_inds

    dist_squared_min = (vector_min * vector_min).sum(axis=-1)
    del vector_min

    highest_weight = compute_weight_vectorized_(dist_squared_min)
    del dist_squared_min

    lowest_relevant_weights = neglectible_treshold * highest_weight
    del highest_weight

    search_radii_squared = bisect_weight_(
        d_min_vals * d_min_vals, d_max * d_max, lowest_relevant_weights, bisect_rtol
    )
    del d_min_vals, d_max, lowest_relevant_weights

    search_radii = np.sqrt(search_radii_squared)
    del search_radii_squared

    relevant_nodes_inds = tree.query_ball_point(
        internal_nodes, search_radii, workers=tree_workers
    )
    relevant_nodes_inds_sizes = np.array(list(map(len, relevant_nodes_inds)))
    relevant_nodes_inds_flat = np.hstack(relevant_nodes_inds)
    del relevant_nodes_inds

    internal_field = inv_dist_weight_(
        boundary_nodes,
        boundary_field,
        internal_nodes,
        relevant_nodes_inds_flat,
        relevant_nodes_inds_sizes,
    )

    return internal_field
