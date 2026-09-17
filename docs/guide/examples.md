# Runnable Examples

The `num.zig` repository includes 25 complete, runnable examples demonstrating every aspect of the library. You can execute any example using `zig build run-<example-name>`.

---

## Example Catalog

| Example Name | Source File | Description | Command |
| :--- | :--- | :--- | :--- |
| `basic` | `examples/basic.zig` | Array creation, basic indexing, arithmetic | `zig build run-basic` |
| `matmul` | `examples/matmul.zig` | Matrix multiplication and timing | `zig build run-matmul` |
| `broadcasting` | `examples/broadcasting.zig` | Multi-dimensional broadcasting demos | `zig build run-broadcasting` |
| `reductions` | `examples/reductions.zig` | Global and axis-wise reductions | `zig build run-reductions` |
| `slicing` | `examples/slicing.zig` | Strided sub-array slices and views | `zig build run-slicing` |
| `manipulation` | `examples/manipulation.zig` | Reshaping, transpose, concat, stack | `zig build run-manipulation` |
| `math_ops` | `examples/math_ops.zig` | Trigonometry, logarithms, exponentials | `zig build run-math_ops` |
| `linear_algebra` | `examples/linear_algebra.zig` | Solvers, decompositions (LU, QR, Cholesky) | `zig build run-linear_algebra` |
| `fft` | `examples/fft.zig` | 1D & 2D FFT spectral transforms | `zig build run-fft` |
| `statistics` | `examples/statistics.zig` | Variance, quantiles, correlation | `zig build run-statistics` |
| `random` | `examples/random.zig` | Normal, uniform, shuffling, sampling | `zig build run-random` |
| `sorting` | `examples/sorting.zig` | In-place sort, argsort, unique sets | `zig build run-sorting` |
| `polynomials` | `examples/polynomials.zig` | Polyval, derivatives, integration, curve fit | `zig build run-polynomials` |
| `sparse` | `examples/sparse.zig` | CSR matrices and Conjugate Gradient | `zig build run-sparse` |
| `parallel_f32` | `examples/parallel_f32.zig` | Multi-threaded vector calculations | `zig build run-parallel_f32` |
| `io_nzig` | `examples/io_nzig.zig` | Binary NZIG file serialization and reload | `zig build run-io_nzig` |
| `io_csv` | `examples/io_csv.zig` | Text CSV export and import parsing | `zig build run-io_csv` |

---

## Running All Examples

Run all examples sequentially to verify functionality:

```bash
zig build run-all-examples
```
