# Architecture & Small-Buffer Optimization (SBO)

`num.zig` is engineered from the ground up for high-performance numerical computing in Zig 0.16.0. It prioritizes memory efficiency, cache locality, explicit allocation semantics, and zero-overhead abstractions.

---

## 1. Small-Buffer Optimization (SBO)

In high-performance numerical workloads, allocating small arrays (such as 3D coordinate vectors, small transformation matrices, or kernel masks) on the heap introduces significant allocator overhead and cache pollution.

`num.zig` implements **inline Small-Buffer Optimization (SBO)** for array metadata:
shape dimensions and strides for up to `MAX_RANK = 8` dimensions live directly
inside every `Array` (`shape_dims: [8]usize`, `stride_vals: [8]isize`), so
shape/stride inspection never touches the heap. Element storage itself uses
64-byte aligned heap buffers (`Buffer`, `SIMD_ALIGNMENT = 64`) for vector
throughput.

### How It Works
1. **Zero Allocations for Metadata**: Rank, shape, and strides up to 8 dimensions are stored inline in the `Array` struct.
2. **Explicit Allocator Retained**: Every array keeps its `Allocator` so owned storage frees correctly; views borrow storage and `deinit()` is a safe no-op for them.
3. **Ranks Above the Limit Rejected**: Shapes longer than `MAX_RANK` return `error.RankExceeded`.

---

## 2. Dynamic vs. Fixed Dimension Representation

To balance arbitrary N-dimensional flexibility with minimal struct footprint, dimensions and strides are managed using a bounded inline vector of ranks:

- Maximum statically reserved rank: `MAX_RANK = 8`
- Dynamically bounded rank representation:
  ```zig
  pub const Shape = struct {
      dims: [8]usize,
      ndim: u8,

      pub fn init(slice: []const usize) Shape { ... }
      pub fn elementCount(self: Shape) usize { ... }
  };
  ```

Because dimensions and strides fit in cache lines alongside the buffer tag and pointer, operations avoid dereferencing separate heap metadata for shape inspections.

---

## 3. Strided Memory Layout

`num.zig` natively supports both row-major (C-contiguous) and arbitrary strided views:

- **Contiguous Array**:
  ```text
  Element offset = sum(index[i] * stride[i])
  Stride[rank - 1] = 1
  Stride[i] = Stride[i + 1] * Shape[i + 1]
  ```
- **Zero-Copy Views**:
  - Slicing (`slice`) produces a sub-view sharing storage with adjusted strides and base offset.
  - Transposition (`transpose`) and axis swapping (`swapAxes`) simply permute the shape and stride arrays without moving data elements in memory when viewed.
  - Broadcasting (`broadcastTo`) creates virtual dimensions with `stride = 0`, enabling scalar and vector expansion with zero additional memory.

---

## 4. SIMD and Vectorization Pipeline

Where vector registers are available, elementwise routines inspect the contiguous layout flag:
1. **Contiguous Vectorized Path**: Checks if strides are C-contiguous. If so, SIMD loops (`@Vector(N, T)`) operate directly across flat pointer slices with alignment awareness.
2. **Strided Fallback Path**: If non-contiguous or transposed, coordinates are walked using multidimensional strided index iterators with loop unrolling.

---

## 5. Threading & Parallel Subsystem

For large operations that exceed CPU cache thresholds, `num.zig` provides a built-in multi-threaded pool:
- **Work Stealing / Chunking**: Iteration spaces are partitioned into optimal block sizes.
- **Recursive Splitting Guard**: Nested parallel calls automatically collapse into sequential loops to prevent worker thread saturation or thread starvation.
- **Sequential Fallback**: Any operation executed with single-element workloads or on single-core systems incurs zero threading overhead.
