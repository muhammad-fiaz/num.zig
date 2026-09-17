# Performance Benchmarks

`num.zig` is continuously profiled and benchmarked across numerical workloads to ensure minimal overhead, optimal cache efficiency, and SIMD throughput.

---

## 1. Running Benchmarks

Compile and run the built-in benchmark suite in release mode:

```bash
zig build bench -Doptimize=ReleaseFast
```

---

## 2. Key Benchmarked Operations

The suite measures throughput (GFLOPS and GB/s) across representative workloads:

### Matrix Multiplication (`matmul`)
- **Tiled Cache Exploitation**: 64x64 cache blocking achieves near-peak CPU FLOP limits.
- **Sizes**: Evaluated on $64 \times 64$, $256 \times 256$, and $1024 \times 1024$ matrices.

### Elementwise Vector Arithmetic (`add`, `mul`)
- Evaluates SIMD auto-vectorization performance on contiguous $10^6$ element vectors.
- Memory throughput approaches physical DRAM bandwidth.

### Parallel Reduction (`sum`)
- Measures single-threaded SIMD vs. multi-threaded thread pool scaling across multiple CPU cores.

### In-Place Sorting (`sort`)
- Evaluates IntroSort performance on random, already-sorted, and reverse-sorted distributions.

### Binary Serialization (`NZIG v1.0`)
- Validates read/write throughput approaching SSD read/write speeds through zero-copy buffer dumps.
