# NumZig

NumZig is a numerical computing library for the Zig programming language, inspired by NumPy. It provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

## Features

- **N-Dimensional Arrays**: Efficient, contiguous memory arrays with broadcasting capabilities.
- **Mathematical Operations**: Elementwise addition, subtraction, multiplication, division, trigonometry, logical, and bitwise operations.
- **Linear Algebra**: Matrix multiplication, solving linear systems, determinants, inverses, norms, decompositions.
- **Statistics**: Mean, variance, standard deviation, histograms, percentiles.
- **Random Number Generation**: Uniform, normal, integer distributions, permutations, sampling.
- **Manipulation**: Reshaping, stacking, tiling, repeating, transposing.
- **Indexing**: Boolean masking, conditional selection (where), non-zero search.
- **Polynomials**: Evaluation, arithmetic, roots, derivatives, integrals.
- **Signal Processing**: Convolution, correlation.
- **Set Operations**: Unique, intersection, union, difference, XOR.
- **Sorting**: Sort, argsort, partition, argpartition.
- **Interpolation**: 1D linear interpolation.
- **Sparse Matrices**: CSR format support.
- **Autograd**: Automatic differentiation (Tensor).
- **DataFrame**: Tabular data structures (DataFrame, Series).
- **Optimization**: Gradient Descent optimizer.
- **Machine Learning Primitives**: Layers (Dense, Dropout), activations (ReLU, Sigmoid, etc.), loss functions (MSE, CrossEntropy), and optimizers (SGD, Adam).

- **FFT**: Fast Fourier Transform implementations.
- **Complex Numbers**: Real, imag, conj, abs.
- **File I/O**: Save and load arrays to disk (Binary, CSV).

## Installation

Add `num.zig` to your `build.zig.zon` or clone the repository and add it as a module in your `build.zig`.

## Documentation

- [Quick Start](guide/quick-start.md)
- [NDArray Guide](guide/ndarray.md)
- [Operations Guide](guide/operations.md)
- [Machine Learning Guide](guide/ml.md)
- [API Reference](api/overview.md)
