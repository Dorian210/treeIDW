# %%
import numpy as np
from treeIDW import treeIDW

np.random.seed(0)
N_boundary = 1_000
N_internal = 1_000_000
boundary_nodes = np.random.rand(N_boundary, 3) # shape (N_boundary, 3) 3 for the 3D space
boundary_field = np.random.rand(N_boundary, 1) # shape (N_boundary, 1) 1 for a scalar field
internal_nodes = np.random.rand(N_internal, 3) # shape (N_internal, 3) 3 for the 3D space

internal_field = treeIDW(boundary_nodes, boundary_field, internal_nodes, 
                         parallel=True) # shape (N_internal, 1) 1 beacause the field is a scalar field
# %%
