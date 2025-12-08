# Complex Numbers API Reference

The `complex` module provides functions for working with complex numbers.

## Functions

### real

Return the real part of the complex argument.

```zig
pub fn real(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```

### imag

Return the imaginary part of the complex argument.

```zig
pub fn imag(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```

### conj

Return the complex conjugate, element-wise.

```zig
pub fn conj(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(Complex(T))
```

### angle

Return the angle of the complex argument.

```zig
pub fn angle(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```

### abs

Calculate the absolute value (magnitude) element-wise.

```zig
pub fn abs(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```