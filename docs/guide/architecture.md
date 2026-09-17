# Architecture & Small-Buffer Optimization (SBO)

`num.zig` is engineered from the ground up for high-performance numerical computing in Zig 0.16.0. It prioritizes memory efficiency, cache locality, explicit allocation semantics, and zero-overhead abstractions.

---

## 1. Small-Buffer Optimization (SBO)

In high-performance numerical workloads, allocating small arrays (such as 3D coordinate vectors, small transformation matrices, or kernel masks) on the heap introduces significant allocator overhead and cache pollution.

`num.zig` implements **Inline Small-Buffer Optimization (SBO)** directly inside every `Array`:

```zig
pub const SBO_CAPACITY = 64; // up to 64 bytes stored inline without heap allocation

pub const Buffer = union(enum) {
    inline_buf: [SBO_CAPACITY]u8,
    heap: []u8,
};
```

### How SBO Works
1. **Zero Allocations for Small Arrays**: If `byte_size <= SBO_CAPACITY`, data is stored entirely inside the `Array` struct.
2. **Explicit Allocator Retained**: Even if an array is currently stored inline, the `Allocator` reference is preserved so dynamic resizing, growing, or operations requiring heap allocation seamlessly work.
3. **No-op Deinitialization**: When deinitializing an array whose buffer is inline, `deinit()` performs a zero-cost no-op rather than invoking the heap allocator.

---

## 2. Dynamic vs. Fixed Dimension Representation

To balance arbitrary N-dimensional flexibility with minimal struct footprint, dimensions and strides are managed using a bounded inline vector of ranks:

- Maximum statically reserved rank: `MAX_RANK = 8`
- Dynamically bounded rank representation:
  ```zig
  pub const Shape = struct {
      dims: [8]usize,
      rank: usize,

      pub fn init(slice: []const usize) Shape { ... }
      pub fn totalElements(self: Shape) usize { ... }
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
  - Slicing (`slice`) produces a sub-view sharing storage or copying depending on mutability requirements.
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
