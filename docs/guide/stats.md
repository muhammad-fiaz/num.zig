# Statistics

The `num.stats` module provides statistical functions and reductions.

## Reductions

These operations reduce the dimensions of an array.

- `sum`: Sum of all elements.
- `mean`: Arithmetic mean of all elements.
- `min`: Minimum value.
- `max`: Maximum value.
- `prod`: Product of all elements.

```zig
const mean_val = num.stats.mean(f32, &arr);
```

## Axis Operations

Perform reductions along a specific axis.

```zig
// Sum along rows (axis 0)
var row_sums = try num.stats.sumAxis(f32, allocator, &arr, 0);
```

## Distributions

- `stdDev`: Standard Deviation.
- `variance`: Variance.
- `median`: Median value.
- `percentile`: q-th percentile.
- `histogram`: Compute histogram of data.

```zig
var std = try num.stats.stdDev(f32, allocator, &arr, 0);
```
