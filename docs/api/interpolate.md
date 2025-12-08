# Interpolation API Reference

The `interpolate` module provides interpolation functions.

## Functions

### interp1d

Interpolate a 1-D function.

```zig
pub fn interp1d(allocator: Allocator, comptime T: type, x: NDArray(T), y: NDArray(T), xi: NDArray(T), options: InterpOptions(T)) !NDArray(T)
```

## Types

### InterpOptions

Options for interpolation.

```zig
pub fn InterpOptions(comptime T: type) type
```