# Indexing API Reference

The `indexing` module provides functions for advanced indexing and slicing.

## Functions

### slice

Extract a slice from an array.

```zig
pub fn slice(allocator: Allocator, comptime T: type, arr: NDArray(T), slices: []const Slice) !NDArray(T)
```

### where

Return elements chosen from x or y depending on condition.

```zig
pub fn where(allocator: Allocator, comptime T: type, condition: NDArray(bool), x: NDArray(T), y: NDArray(T)) !NDArray(T)
```

### booleanMask

Return elements of an array where a mask is true.

```zig
pub fn booleanMask(allocator: Allocator, comptime T: type, arr: NDArray(T), mask: NDArray(bool)) !NDArray(T)
```

### take

Take elements from an array along an axis.

```zig
pub fn take(allocator: Allocator, comptime T: type, arr: NDArray(T), indices: NDArray(usize), axis: usize) !NDArray(T)
```