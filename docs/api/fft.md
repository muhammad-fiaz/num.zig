# FFT

The `fft` module provides Fast Fourier Transform operations.

## Functions

### `fft`

Computes the 1D Fast Fourier Transform.

```zig
pub fn fft(allocator: std.mem.Allocator, input: *const NDArray(f32)) !NDArray(Complex(f32))
```

**Parameters:**
- `allocator`: Memory allocator.
- `input`: Input 1D array of real numbers (length must be power of 2).

**Returns:**
- `NDArray(Complex(f32))` containing FFT coefficients.

### `ifft`

Computes the 1D Inverse Fast Fourier Transform.

```zig
pub fn ifft(allocator: std.mem.Allocator, input: *const NDArray(Complex(f32))) !NDArray(Complex(f32))
```

**Parameters:**
- `allocator`: Memory allocator.
- `input`: Input 1D array of complex numbers.

**Returns:**
- `NDArray(Complex(f32))` containing IFFT result.

## Example

```zig
const std = @import("std");
const num = @import("num");
const FFT = num.fft.FFT;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    var a = try num.NDArray(f32).init(allocator, &.{4});
    a.set(&.{0}, 1.0);
    a.set(&.{1}, 1.0);
    a.set(&.{2}, 1.0);
    a.set(&.{3}, 1.0);

    var result = try FFT.fft(allocator, &a);
    defer result.deinit();
    
    // result is approx {4+0i, 0+0i, 0+0i, 0+0i}
}
```
