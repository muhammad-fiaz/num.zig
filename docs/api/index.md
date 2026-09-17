# API Overview

Welcome to the `num.zig` API reference. `num.zig` provides a comprehensive, type-safe API for numerical arrays and scientific computing in Zig 0.16.0.

---

## Subsystem Navigation

### Core
- [**Core Array & Memory**](/api/core): Array struct, Shape, Strides, Buffer allocation, SBO.
- [**DType & Type Promotion**](/api/dtype): DType enum, type inspection, promotion rules.

### Math & Operations
- [**Elementwise Math**](/api/elementwise): Arithmetic, trigonometry, logarithms, exponentials, and special functions.
- [**Comparisons & Logic**](/api/compare): Elementwise comparisons, floating-point predicates (`isnan`, `isinf`), and masking.
- [**Reductions & Accumulations**](/api/reduce): `sum`, `prod`, `mean`, `min`, `max`, `argmin`, `argmax`, `cumsum`, `diff`.
- [**Shape Manipulation**](/api/manip): Reshaping, transpose, axis swapping, concatenation, stacking, padding, rolling.

### Scientific Computing
- [**Linear Algebra**](/api/linalg): Matrix multiplications, inversions, solvers, eigenvalues, SVD, decompositions.
- [**Fast Fourier Transform**](/api/fft): 1D and multi-dimensional FFT, real-valued FFT, frequency utilities.
- [**Sparse Matrices**](/api/sparse): CSR/CSC formats, sparse matrix-vector products, CG and GMRES iterative solvers.
- [**CPU Parallel Execution**](/api/parallel): Multi-threaded `run` work scheduler.
- [**Random Number Generation**](/api/random): Seeded PRNG, uniform, normal, discrete integer, and shuffle generators.
- [**Statistics**](/api/stats): Variances, standard deviations, quantiles, covariance, correlation, histograms.
- [**Sorting & Sets**](/api/sort): In-place sort, argsort, binary search, unique sets, intersections.
- [**Polynomials**](/api/poly): Polynomial evaluation, derivatives, anti-derivatives, curve fitting.
- [**Serialization & I/O**](/api/io): NZIG v1.0 binary files and CSV/delimited text import/export.
