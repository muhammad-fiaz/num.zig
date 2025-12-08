# Sorting API Reference

The `algo.sort` module provides sorting algorithms and related functions.

## Sorting

### sort

Return a sorted copy of an array.

```zig
pub fn sort(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(T)
```

### sort1D

Sort a 1D array in-place (or return copy? check impl).

```zig
pub fn sort1D(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(T)
```

### argsort

Returns the indices that would sort an array.

```zig
pub fn argsort(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(usize)
```

### argsort1D

Returns the indices that would sort a 1D array.

```zig
pub fn argsort1D(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(usize)
```

### partition

Return a partitioned copy of an array.

```zig
pub fn partition(allocator: Allocator, comptime T: type, arr: NDArray(T), kth: usize, axis: usize) !NDArray(T)
```

### argpartition

Return the indices that would partition an array.

```zig
pub fn argpartition(allocator: Allocator, comptime T: type, arr: NDArray(T), kth: usize, axis: usize) !NDArray(usize)
```

## Searching / Counting

### nonzero

Return the indices of the elements that are non-zero.

```zig
pub fn nonzero(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(usize)
```

### flatnonzero

Return indices that are non-zero in the flattened version of a.

```zig
pub fn flatnonzero(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(usize)
```