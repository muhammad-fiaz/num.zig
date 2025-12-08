# Set Operations API Reference

The `setops` module provides set operations for 1D arrays.

## Functions

### unique

Find the unique elements of an array.

```zig
pub fn unique(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T)
```

### in1d

Test whether each element of a 1-D array is also present in a second array.

```zig
pub fn in1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(bool)
```

### intersect1d

Find the intersection of two arrays.

```zig
pub fn intersect1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T)
```

### union1d

Find the union of two arrays.

```zig
pub fn union1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T)
```

### setdiff1d

Find the set difference of two arrays.

```zig
pub fn setdiff1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T)
```

### setxor1d

Find the set exclusive-or of two arrays.

```zig
pub fn setxor1d(allocator: Allocator, comptime T: type, ar1: NDArray(T), ar2: NDArray(T)) !NDArray(T)
```