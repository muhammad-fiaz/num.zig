# Statistics API Reference

The `stats` module provides functions for statistical analysis and data reduction.

## Reductions

### sum

Calculates the sum of all elements in the array.

```zig
pub fn sum(allocator: Allocator, comptime T: type, a: NDArray(T)) !T
```

### prod

Calculates the product of all elements in the array.

```zig
pub fn prod(allocator: Allocator, comptime T: type, a: NDArray(T)) !T
```

### min

Finds the minimum value in the array.

```zig
pub fn min(allocator: Allocator, comptime T: type, a: NDArray(T)) !T
```

### max

Finds the maximum value in the array.

```zig
pub fn max(allocator: Allocator, comptime T: type, a: NDArray(T)) !T
```

### mean

Calculates the arithmetic mean of the array elements.

```zig
pub fn mean(allocator: Allocator, comptime T: type, a: NDArray(T)) !f64
```

### median

Calculates the median of the array elements.

```zig
pub fn median(allocator: Allocator, comptime T: type, a: NDArray(T)) !f64
```

### variance

Calculates the variance of the array elements.

```zig
pub fn variance(allocator: Allocator, comptime T: type, a: NDArray(T)) !f64
```

### stdDev

Calculates the standard deviation of the array elements.

```zig
pub fn stdDev(allocator: Allocator, comptime T: type, a: NDArray(T)) !f64
```

## Axis Operations

### sumAxis

Calculates the sum of array elements along a specified axis.

```zig
pub fn sumAxis(allocator: Allocator, comptime T: type, a: NDArray(T), axis: usize) !NDArray(T)
```

### meanAxis

Calculates the mean of array elements along a specified axis.

```zig
pub fn meanAxis(allocator: Allocator, comptime T: type, a: NDArray(T), axis: usize) !NDArray(f64)
```

## Index Operations

### argmin

Returns the index of the minimum value in the flattened array.

```zig
pub fn argmin(allocator: Allocator, comptime T: type, a: NDArray(T)) !usize
```

### argmax

Returns the index of the maximum value in the flattened array.

```zig
pub fn argmax(allocator: Allocator, comptime T: type, a: NDArray(T)) !usize
```

## Other

### bincount

Count number of occurrences of each value in array of non-negative ints.

```zig
pub fn bincount(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(usize)
```

### unique

Find the unique elements of an array.

```zig
pub fn unique(allocator: Allocator, comptime T: type, a: NDArray(T)) !NDArray(T)
```