# Complex Helpers API

Module: `@import("num").ops`

Complex dtypes (`c64`, `c128`) only; other dtypes return `DTypeError.UnsupportedDType`. Comparisons and ordering on complex arrays are unsupported. All operations return newly owned arrays.

---

## Conjugation

```zig
pub fn conj(a: Array) !Array; // alias: conjugate; negates the imaginary part, preserves dtype
pub fn conjTranspose(a: Array) !Array; // 2D Hermitian transpose via transpose + conj
```

---

## Components & Polar Form

```zig
pub fn real(a: Array) !Array;      // c64 -> f32, c128 -> f64
pub fn imag(a: Array) !Array;      // c64 -> f32, c128 -> f64
pub fn magnitude(a: Array) !Array; // |z| = hypot(re, im)
pub fn phase(a: Array) !Array;     // arg(z) = atan2(im, re) in radians
```

Complex element types reuse Zig 0.16.0 `std.math.Complex(f32)` (`c64`) and `std.math.Complex(f64)` (`c128`).

---

## Example

```zig
var c = try num.ops.conj(z);
defer c.deinit();

var mag = try num.ops.magnitude(z);
defer mag.deinit();

var h = try num.ops.conjTranspose(m);
defer h.deinit();
```
