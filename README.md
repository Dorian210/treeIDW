# treeIDW

<p align="center">
  <img src=docs/logo.png width="500" />
</p>

**treeIDW** is a Python library designed for performing Inverse Distance Weighting (IDW) interpolation with an efficient KD-tree approach. It offers a user-friendly interface for newcomers while providing advanced features and optimizations that will appeal to experts in numerical methods and spatial data analysis.

## Introduction

Imagine you have data known at certain points (boundary nodes) and you need to estimate values at other points (internal nodes). **treeIDW** simplifies this task by automatically selecting the most relevant boundary nodes using a KD-tree, ensuring that only nodes with significant contributions are considered. This not only makes the interpolation more accurate but also speeds up the computation considerably. Whether you're a researcher, engineer, or someone exploring spatial data, **treeIDW** offers a robust solution that adapts to your level of expertise.

For non-experts, the library hides complex mathematical operations behind a simple interface, while experts will appreciate the fine-tuned control over parameters such as weight thresholds, bisection tolerances, and parallel processing options.

## Installation

Since **treeIDW** is not yet available on PyPI, you can install it locally as follows:

```bash
git clone https://github.com/Dorian210/treeIDW
cd treeIDW
pip install -e .
```

### Dependencies

Ensure that you have the following dependencies installed:

- `numpy`
- `numba`
- `scipy`

## Main Modules

- **treeIDW.treeIDW**  
  Implements the core IDW interpolation method using a KD-tree to select the most relevant boundary nodes. This module efficiently ignores nodes with negligible impact, improving both accuracy and performance.

- **treeIDW.helper_functions**  
  Contains optimized functions (leveraging `numba`) for computing IDW weights. These include both vectorized and parallelized implementations, which are ideal for handling large datasets.

- **treeIDW.weight_function**  
  Defines the specific IDW weight calculation function. By default, this function computes the weight as the inverse of the squared distance, but it can be adapted to suit specific needs.

## Examples

Several example scripts demonstrating the usage of **treeIDW** are available in the `examples/` directory. These include:

- **Graphical Demonstration:** A visualization of a square domain with a rotating vector field. The field is propagated to multiple internal points and plotted for intuitive understanding.
- **Large-Scale Computation:** A more computationally intensive example where a scalar field is propagated from 1,000 boundary nodes to 1,000,000 internal nodes, showcasing the efficiency of the KD-tree selection.
- **Logo Generation:** A unique example illustrating the process of creating the treeIDW logo. The logo consists of the letters "IDW" represented as a vector field, which is then propagated onto a 2D meshgrid to generate the final design.

## Documentation

Complete API documentation is available in the `docs/` directory of the project or through the [online documentation portal](https://dorian210.github.io/treeIDW/).

## Contributions

Currently, I am not actively reviewing contributions. However, if you encounter any issues or have suggestions, please feel free to open an issue on the repository.

## License

This project is licensed under the [CeCILL License](LICENSE.txt).

