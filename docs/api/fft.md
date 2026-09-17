# Fast Fourier Transform API

Module: `@import("num").fft`

---

## 1D Fourier Transforms

Both `fft` and `ifft` operate along a specified axis and support complex and real inputs.
Real inputs are automatically promoted to the appropriate complex dtype (`c64` for `f32`, `c128` for `f64`).

```zig
pub fn fft(a: Array, options: struct { axis: isize = -1, norm: enum { backward, ortho, forward } = .backward }) !Array;
pub fn ifft(a: Array, options: struct { axis: isize = -1, norm: enum { backward, ortho, forward } = .backward }) !Array;
```

---

## Frequency Helpers

```zig
/// DFT sample frequencies for a transform of length `n`.
pub fn fftfreq(
    allocator: std.mem.Allocator,
    n: usize,
    options: struct { d: f64 = 1.0, dtype: DType = .f64 },
) !Array;

/// DFT sample frequencies for a real-input transform of length `n` (length = n/2 + 1).
pub fn rfftfreq(
    allocator: std.mem.Allocator,
    n: usize,
    options: struct { d: f64 = 1.0, dtype: DType = .f64 },
) !Array;
```

---

## Shift Helpers

```zig
/// Shift zero-frequency component to center of spectrum along specified axis.
pub fn fftshift(a: Array, options: struct { axis: ?isize = null }) !Array;

/// Inverse of fftshift.
pub fn ifftshift(a: Array, options: struct { axis: ?isize = null }) !Array;
```

---

## Complex Types

```zig
pub const Complex64 = std.math.Complex(f32);   // c64 element type
pub const Complex128 = std.math.Complex(f64);  // c128 element type
```
