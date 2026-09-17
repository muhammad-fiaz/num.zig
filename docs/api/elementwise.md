# Elementwise Math API

Module: `@import("num").ops`

---

## Binary Operations

All binary operations support automatic multi-dimensional broadcasting:

```zig
pub fn add(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn subtract(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn multiply(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn divide(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn pow(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn remainder(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn maximum(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn minimum(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn hypot(a: Array, b: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn atan2(y: Array, x: Array, options: struct { dtype: ?DType = null }) !Array;
```

---

## Unary Operations

```zig
pub fn negate(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn positive(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn abs(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn sign(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn sqrt(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn cbrt(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn square(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn reciprocal(a: Array, options: struct { dtype: ?DType = null }) !Array;
```

---

## Exponential & Logarithmic

```zig
pub fn exp(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn exp2(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn expm1(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn log(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn log2(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn log10(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn log1p(a: Array, options: struct { dtype: ?DType = null }) !Array;
```

---

## Trigonometric & Hyperbolic

```zig
pub fn sin(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn cos(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn tan(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn asin(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn acos(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn atan(a: Array, options: struct { dtype: ?DType = null }) !Array;

pub fn sinh(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn cosh(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn tanh(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn asinh(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn acosh(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn atanh(a: Array, options: struct { dtype: ?DType = null }) !Array;
```

---

## Rounding & Angles

```zig
pub fn floor(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn ceil(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn round(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn trunc(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn degreesToRadians(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn radiansToDegrees(a: Array, options: struct { dtype: ?DType = null }) !Array;
```

---

## Special Functions & Selection

```zig
pub fn gamma(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn lgamma(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn erf(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn erfc(a: Array, options: struct { dtype: ?DType = null }) !Array;
pub fn clip(a: Array, options: struct { min: ?anytype = null, max: ?anytype = null }) !Array;
pub fn where(condition: Array, x: Array, y: Array, options: struct { dtype: ?DType = null }) !Array;
```

---

## Deliberate Aliases

To support common mathematical shorthand without sacrificing readability, `num.zig` provides direct aliases mapping to canonical elementwise operations:

| Canonical Operation | Deliberate Alias | Purpose |
|:---|:---|:---|
| `subtract` | `sub` | Direct alias for subtraction |
| `multiply` | `mul` | Direct alias for multiplication |
| `divide` | `div` | Direct alias for division |
| `remainder` | `rem`, `mod` | Direct aliases for remainder |
| `pow` | `power` | Direct alias for power |
| `abs` | `absolute` | Direct alias for absolute value |
| `negate` | `negative` | Direct alias for negation |

Aliases are implemented as thin compile-time re-exports (`pub const sub = subtract;`) with identical behavior, signature, and performance.

---

## Integer Bitwise Operations

Integer dtypes only with broadcasting. Floating-point, boolean, and complex inputs return `DTypeError.UnsupportedDType`.

```zig
pub fn bitwiseAnd(a: Array, b: Array) !Array;
pub fn bitwiseOr(a: Array, b: Array) !Array;
pub fn bitwiseXor(a: Array, b: Array) !Array;
pub fn bitwiseNot(a: Array) !Array;
pub fn leftShift(a: Array, b: Array) !Array;
pub fn rightShift(a: Array, b: Array) !Array;
pub fn bitCount(a: Array) !Array;
pub fn clz(a: Array) !Array;       // leading-zero count; alias: leadingZeros
pub fn ctz(a: Array) !Array;       // trailing-zero count; alias: trailingZeros
pub const popcount = bitCount;
```

---

## Complex Helpers

Complex dtypes (`c64`, `c128`) only. Comparisons and ordering on complex arrays are unsupported.

```zig
pub fn conj(a: Array) !Array;        // alias: conjugate; preserves dtype
pub fn real(a: Array) !Array;        // c64 -> f32, c128 -> f64
pub fn imag(a: Array) !Array;        // c64 -> f32, c128 -> f64
pub fn magnitude(a: Array) !Array;   // |z|, c64 -> f32, c128 -> f64
pub fn phase(a: Array) !Array;       // arg(z) in radians
```


