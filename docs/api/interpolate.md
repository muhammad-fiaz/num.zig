# Interpolation

The `interpolate` module provides functions for data interpolation.

## Functions

### `interp1d`

Performs 1D linear interpolation.

```zig
pub fn interp1d(allocator: Allocator, comptime T: type, x: NDArray(T), y: NDArray(T), xi: NDArray(T)) !NDArray(T)
```

**Parameters:**
- `allocator`: Memory allocator.
- `T`: Element type (must be float).
- `x`: X coordinates of data points (must be sorted).
- `y`: Y coordinates of data points.
- `xi`: X coordinates to evaluate the interpolated values at.

**Returns:**
- A new `NDArray(T)` containing the interpolated values.

**Description:**
Uses linear interpolation to find values at `xi`. Points outside the range of `x` are clamped to the nearest edge value (constant extrapolation).

## Example

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Data points
    var x = try num.NDArray(f64).init(allocator, &.{3}); // [0, 1, 2]
    var y = try num.NDArray(f64).init(allocator, &.{3}); // [0, 10, 20]
    
    // Query points
    var xi = try num.NDArray(f64).init(allocator, &.{2}); // [0.5, 1.5]

    // Interpolate
    var yi = try num.interpolate.interp1d(allocator, f64, x, y, xi);
    defer yi.deinit();
    
    // yi should be [5.0, 15.0]
}
```
