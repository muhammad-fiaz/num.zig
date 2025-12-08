# Manipulation API Reference

The `manipulation` module provides functions for changing array shapes and content.

## Shape Manipulation

### ravel

Return a contiguous flattened array.

```zig
pub fn ravel(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(T)
```

### moveaxis

Move axes of an array to new positions.

```zig
pub fn moveaxis(allocator: Allocator, comptime T: type, arr: NDArray(T), source: []const usize, destination: []const usize) !NDArray(T)
```

### swapaxes

Interchange two axes of an array.

```zig
pub fn swapaxes(allocator: Allocator, comptime T: type, arr: NDArray(T), axis1: usize, axis2: usize) !NDArray(T)
```

### flip

Reverse the order of elements in an array along the given axis.

```zig
pub fn flip(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(T)
```

### roll

Roll array elements along a given axis.

```zig
pub fn roll(allocator: Allocator, comptime T: type, arr: NDArray(T), shift: isize, axis: usize) !NDArray(T)
```

## Joining Arrays

### vstack

Stack arrays in sequence vertically (row wise).

```zig
pub fn vstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T)
```

### hstack

Stack arrays in sequence horizontally (column wise).

```zig
pub fn hstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T)
```

### dstack

Stack arrays in sequence depth wise (along third axis).

```zig
pub fn dstack(allocator: Allocator, comptime T: type, arrays: []const NDArray(T)) !NDArray(T)
```

## Tiling and Repeating

### tile

Construct an array by repeating A the number of times given by reps.

```zig
pub fn tile(allocator: Allocator, comptime T: type, arr: NDArray(T), reps: []const usize) !NDArray(T)
```

### repeat

Repeat elements of an array.

```zig
pub fn repeat(allocator: Allocator, comptime T: type, arr: NDArray(T), repeats: usize, axis: ?usize) !NDArray(T)
```

## Search and Replace

### find

Find indices where a predicate is true.

```zig
pub fn find(allocator: Allocator, comptime T: type, a: *const NDArray(T), predicate: anytype) !NDArray(usize)
```

### replace

Replace all occurrences of a value.

```zig
pub fn replace(allocator: Allocator, comptime T: type, a: *const NDArray(T), old_val: T, new_val: T) !NDArray(T)
```

### replaceWhere

Replace values where a predicate is true.

```zig
pub fn replaceWhere(allocator: Allocator, comptime T: type, a: *const NDArray(T), predicate: anytype, value: T) !NDArray(T)
```

### replaceFirst

Replace the first occurrence of a value.

```zig
pub fn replaceFirst(allocator: Allocator, comptime T: type, a: *const NDArray(T), old_val: T, new_val: T) !NDArray(T)
```

### replaceLast

Replace the last occurrence of a value.

```zig
pub fn replaceLast(allocator: Allocator, comptime T: type, a: *const NDArray(T), old_val: T, new_val: T) !NDArray(T)
```

### delete

Return a new array with sub-arrays along an axis deleted.

```zig
pub fn delete(allocator: Allocator, comptime T: type, a: *const NDArray(T), indices: []const usize) !NDArray(T)
```