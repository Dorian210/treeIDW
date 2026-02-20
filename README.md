# treeIDW

<p align="center">
  <img src="https://raw.githubusercontent.com/Dorian210/treeIDW/main/docs/logo.png" width="500" />
</p>

**treeIDW** is a Python library for performing **Inverse Distance Weighting (IDW)** interpolation using an efficient **KD-tree-based selection strategy**.  
It is designed to be easy to use for newcomers while offering fine-grained control and performance-oriented options for advanced users in numerical methods and spatial data analysis.

---

## Key Features

- Efficient IDW interpolation using KD-tree nearest-neighbor selection
- Automatic exclusion of boundary nodes with negligible contribution
- Optimized numerical kernels powered by `numba`
- Scalable to large datasets (millions of interpolation points)
- Simple API with expert-level tunable parameters

---

## Installation

**treeIDW is available on PyPI.**

```bash
pip install treeIDW
```

### Development installation (from source)

```bash
git clone https://github.com/Dorian210/treeIDW
cd treeIDW
pip install -e .
```

---

## Dependencies

The core dependencies are:

- `numpy`
- `scipy`
- `numba`

These are automatically installed when using `pip`.

---

## Package Structure

- **treeIDW.treeIDW**  
  Core IDW interpolation engine.  
  Uses a KD-tree to select only boundary nodes with significant influence, improving both accuracy and performance.

- **treeIDW.helper_functions**  
  Low-level, performance-critical routines for IDW weight computation.  
  Implemented with `numba`, including vectorized and parallelized variants.

- **treeIDW.weight_function**  
  Definition of the IDW weight function.  
  The default implementation uses inverse squared distance, but custom weight laws can be implemented if needed.

---

## Examples

Example scripts are provided in the `examples/` directory:

- **Graphical demonstration**  
  Interpolation of a rotating vector field inside a square domain.

- **Large-scale computation**  
  Propagation of a scalar field from 1,000 boundary nodes to 1,000,000 internal points, highlighting scalability.

- **Logo generation**  
  The generation process of the *treeIDW* logo itself, where the letters “IDW” are encoded as a vector field and interpolated on a 2D grid.

---

## Documentation

- Online documentation: https://dorian210.github.io/treeIDW/
- API reference is also available in the `docs/` directory of the repository.

---

## Contributions

This project is currently not open to active development contributions.  
However, bug reports and suggestions are welcome via the issue tracker.

---

## License

This project is distributed under the **CeCILL License**.  
See [LICENSE.txt](LICENSE.txt) for details.
