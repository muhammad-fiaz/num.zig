# Complex Numbers

The `complex` module provides functions for working with complex number arrays.

## Functions

### `real`

Extracts the real part of a complex array.

```zig
pub fn real(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```

**Parameters:**
- `allocator`: Memory allocator.
- `T`: The underlying float type (e.g., `f64`).
- `a`: Input complex array.

**Returns:**
- `NDArray(T)` containing the real components.

### `imag`

Extracts the imaginary part of a complex array.

```zig
pub fn imag(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```

**Parameters:**
- `allocator`: Memory allocator.
- `T`: The underlying float type.
- `a`: Input complex array.

**Returns:**
- `NDArray(T)` containing the imaginary components.

### `conj`

Computes the complex conjugate.

```zig
pub fn conj(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(Complex(T))
```

**Returns:**
- A new complex array with conjugated elements.

### `angle`

Computes the phase angle (argument) of complex elements.

```zig
pub fn angle(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```

**Returns:**
- Array of angles in radians.

### `abs`

Computes the magnitude (absolute value) of complex elements.

```zig
pub fn abs(allocator: Allocator, comptime T: type, a: NDArray(Complex(T))) !NDArray(T)
```

**Returns:**
- Array of magnitudes.

## Example

```zig
const std = @import("std");
const num = @import("num");
const Complex = std.math.Complex;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Create complex array
    var z = try num.NDArray(Complex(f64)).init(allocator, &.{2});
    // ... fill z ...

    // Get real part
    var r = try num.complex.real(allocator, f64, z);
    defer r.deinit();
    
    // Get magnitude
    var m = try num.complex.abs(allocator, f64, z);
    defer m.deinit();
}
```
