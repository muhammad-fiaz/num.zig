# Runnable Examples

The `num.zig` repository includes complete, runnable examples demonstrating every aspect of the library. Execute any example with `zig build run-<example-name>`.

---

## Example Catalog

| Example Name | Source File | Description | Command |
| :--- | :--- | :--- | :--- |
| `array_creation` | `examples/array_creation.zig` | Zeros, ones, full, arange, linspace, eye | `zig build run-array_creation` |
| `elementwise` | `examples/elementwise.zig` | Arithmetic, trig, exponentials, masking | `zig build run-elementwise` |
| `broadcasting` | `examples/broadcasting.zig` | Multidimensional broadcasting rules | `zig build run-broadcasting` |
| `bitwise_ops` | `examples/bitwise_ops.zig` | Integer bitwise logic, shifts, bit counts | `zig build run-bitwise_ops` |
| `complex_ops` | `examples/complex_ops.zig` | Complex conj, real/imag, magnitude, phase | `zig build run-complex_ops` |
| `slicing` | `examples/slicing.zig` | Strided sub-array slices and views | `zig build run-slicing` |
| `reshape` | `examples/reshape.zig` | Reshape, ravel, flatten, squeeze, expandDims | `zig build run-reshape` |
| `transpose` | `examples/transpose.zig` | Transpose, swapAxes, moveAxis | `zig build run-transpose` |
| `concatenation` | `examples/concatenation.zig` | Concat, stack, split, tile, repeat | `zig build run-concatenation` |
| `reductions` | `examples/reductions.zig` | Global and axis-wise reductions | `zig build run-reductions` |
| `sorting` | `examples/sorting.zig` | In-place sort, sorted, argsort | `zig build run-sorting` |
| `searching` | `examples/searching.zig` | searchSorted, unique, nonzero, set ops | `zig build run-searching` |
| `random_generation` | `examples/random_generation.zig` | Uniform, normal, integers, shuffle | `zig build run-random_generation` |
| `statistics` | `examples/statistics.zig` | Mean, median, variance, percentiles | `zig build run-statistics` |
| `correlation` | `examples/correlation.zig` | Covariance, correlation, histogram | `zig build run-correlation` |
| `matmul` | `examples/matmul.zig` | Matrix multiplication, dot, inner, outer | `zig build run-matmul` |
| `norms` | `examples/norms.zig` | Vector and matrix norms | `zig build run-norms` |
| `solve` | `examples/solve.zig` | Solvers, inverse, determinant | `zig build run-solve` |
| `decompositions` | `examples/decompositions.zig` | LU, QR, Cholesky | `zig build run-decompositions` |
| `eigenvalues` | `examples/eigenvalues.zig` | Eigenvalues and eigenvectors | `zig build run-eigenvalues` |
| `svd` | `examples/svd.zig` | Singular Value Decomposition | `zig build run-svd` |
| `fft` | `examples/fft.zig` | FFT, IFFT, spectra | `zig build run-fft` |
| `sparse_matrix` | `examples/sparse_matrix.zig` | CSR matrices and CG solver | `zig build run-sparse_matrix` |
| `polynomials` | `examples/polynomials.zig` | Evaluation, derivatives, fit, roots | `zig build run-polynomials` |
| `serialization` | `examples/serialization.zig` | NZIG binary serialization | `zig build run-serialization` |
| `text_io` | `examples/text_io.zig` | CSV text save/load | `zig build run-text_io` |
| `parallel_basic` | `examples/parallel_basic.zig` | Automatic parallel execution | `zig build run-parallel_basic` |
| `parallel_config` | `examples/parallel_config.zig` | Explicit workers and chunk size | `zig build run-parallel_config` |
| `parallel_large_array` | `examples/parallel_large_array.zig` | Large array parallel processing | `zig build run-parallel_large_array` |
| `parallel_f32` | `examples/parallel_f32.zig` | Single-precision parallel arrays | `zig build run-parallel_f32` |
| `parallel_f64` | `examples/parallel_f64.zig` | Double-precision parallel decay | `zig build run-parallel_f64` |
| `parallel_non_contiguous` | `examples/parallel_non_contiguous.zig` | Parallel on strided views | `zig build run-parallel_non_contiguous` |
| `parallel_reduction` | `examples/parallel_reduction.zig` | Multithreaded tree reduction | `zig build run-parallel_reduction` |

---

## Running All Examples

Run all examples sequentially to verify functionality:

```bash
zig build run-all-examples
```
