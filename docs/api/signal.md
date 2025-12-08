# Signal Processing API Reference

The `signal` module provides signal processing functions.

## Convolution

### convolve

Convolve two N-dimensional arrays.

```zig
pub fn convolve(allocator: Allocator, comptime T: type, a: NDArray(T), v: NDArray(T), mode: ConvolveMode) !NDArray(T)
```

### correlate

Cross-correlate two N-dimensional arrays.

```zig
pub fn correlate(allocator: Allocator, comptime T: type, a: NDArray(T), v: NDArray(T), mode: ConvolveMode) !NDArray(T)
```

## Types

### ConvolveMode

Enum for convolution mode.

```zig
pub const ConvolveMode = enum {
    full,
    valid,
    same,
};
```