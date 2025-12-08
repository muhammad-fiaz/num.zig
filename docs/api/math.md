# Math API Reference

The `math` module provides mathematical functions and SIMD operations.

## SIMD Operations

### add

Perform SIMD addition.

```zig
pub fn add(comptime T: type, dest: []T, a: []const T, b: []const T) void
```

### mul

Perform SIMD multiplication.

```zig
pub fn mul(comptime T: type, dest: []T, a: []const T, b: []const T) void
```