# Sparse Matrices API Reference

The `sparse` module provides sparse matrix data structures.

## CSRMatrix

Compressed Sparse Row matrix.

```zig
pub fn CSRMatrix(comptime T: type) type
```

### init

Initialize a sparse matrix.

```zig
pub fn init(allocator: Allocator, rows: usize, cols: usize) !Self
```

### deinit

Deinitialize the matrix.

```zig
pub fn deinit(self: *Self, allocator: Allocator) void
```

### fromDense

Convert a dense array to a CSR matrix.

```zig
pub fn fromDense(allocator: Allocator, arr: NDArray(T)) !Self
```

### toDense

Convert the CSR matrix to a dense array.

```zig
pub fn toDense(self: Self, allocator: Allocator) !NDArray(T)
```