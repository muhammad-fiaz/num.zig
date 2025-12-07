# Sorting

The `sort` module provides sorting algorithms for arrays.

## Functions

### `sortByAlgo`

Sorts an array along a specified axis using a chosen algorithm.

```zig
pub fn sortByAlgo(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize, algo: SortAlgo) !NDArray(T)
```

**Parameters:**
- `axis`: The axis along which to sort.
- `algo`: The sorting algorithm (`.QuickSort`, `.MergeSort`, `.HeapSort`, `.InsertionSort`, `.BubbleSort`, `.TimSort`).

### `argsort`

Returns the indices that would sort an array.

```zig
pub fn argsort(allocator: Allocator, comptime T: type, arr: NDArray(T), axis: usize) !NDArray(usize)
```

### `nonzero`

Return the indices of the elements that are non-zero.

```zig
pub fn nonzero(allocator: Allocator, comptime T: type, arr: NDArray(T)) !NDArray(usize)
```

## Example

```zig
const std = @import("std");
const num = @import("num");
const sort = num.algo.sort;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = try num.NDArray(f32).init(allocator, &.{3}, &.{3.0, 1.0, 2.0});
    
    var sorted = try sort.sortByAlgo(allocator, f32, a, 0, .QuickSort);
    defer sorted.deinit();
    // sorted is {1.0, 2.0, 3.0}
}
```
