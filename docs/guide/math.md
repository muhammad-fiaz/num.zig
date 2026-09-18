# Vectorized Math & Special Functions

`num.zig` provides vectorized implementations for arithmetic, exponential, logarithmic, trigonometric, hyperbolic, and special numerical functions.

---

## 1. Vectorized Arithmetic

All elementary arithmetic operations support SIMD optimization and multi-dimensional broadcasting:

```zig
var a = try num.full(allocator, .{ .shape = &.{ 1000 }, .value = @as(f64, 2.0) });
defer a.deinit();
var b = try num.full(allocator, .{ .shape = &.{ 1000 }, .value = @as(f64, 3.0) });
defer b.deinit();

var c_add = try num.ops.add(a, b, .{}); defer c_add.deinit();
var c_sub = try num.ops.subtract(a, b, .{}); defer c_sub.deinit();
var c_mul = try num.ops.multiply(a, b, .{}); defer c_mul.deinit();
var c_div = try num.ops.divide(a, b, .{}); defer c_div.deinit();
var c_pow = try num.ops.pow(a, b, .{}); defer c_pow.deinit();
var c_rem = try num.ops.remainder(a, b, .{}); defer c_rem.deinit();
var c_min = try num.ops.minimum(a, b, .{}); defer c_min.deinit();
var c_max = try num.ops.maximum(a, b, .{}); defer c_max.deinit();

// Unary: negate/positive preserve dtype; deliberate aliases absolute/negative/power/mod.
var neg = try num.ops.negate(a, .{}); defer neg.deinit();
var pos = try num.ops.positive(a, .{}); defer pos.deinit();
var ab = try num.ops.absolute(a, .{}); defer ab.deinit();
```

---

## 2. Exponential & Logarithmic Functions

- `num.ops.exp(arr, .{})`: Elementwise $e^x$
- `num.ops.exp2(arr, .{})`: Elementwise $2^x$
- `num.ops.expm1(arr, .{})`: Elementwise $e^x - 1$
- `num.ops.log(arr, .{})`: Natural logarithm $\ln(x)$
- `num.ops.log2(arr, .{})`: Base-2 logarithm $\log_2(x)$
- `num.ops.log10(arr, .{})`: Base-10 logarithm $\log_{10}(x)$
- `num.ops.log1p(arr, .{})`: $\ln(1 + x)$ for small $x$

---

## 3. Trigonometric & Hyperbolic Functions

- **Trigonometric**: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- **Hyperbolic**: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`
- **Conversion**: `degreesToRadians`, `radiansToDegrees`

---

## 4. Special & Utility Functions

- **Error functions**: `erf`, `erfc`
- **Gamma / Factorials**: `gamma`, `lgamma`
- **Clipping**: `clip(arr, .{ .min = 0.0, .max = 1.0 })`
- **Conditional Selection**:
  ```zig
  // where(condition, x, y) selects elements from x where cond is true, else y
  var out = try num.ops.where(cond_arr, x_arr, y_arr);
  defer out.deinit();
  ```

---

## 5. Integer Bitwise Operations

Integer dtypes only with broadcasting. Floating-point inputs are rejected:

```zig
var band = try num.ops.bitwiseAnd(a_int, b_int);
defer band.deinit();
var bor = try num.ops.bitwiseOr(a_int, b_int);
defer bor.deinit();
var bxor = try num.ops.bitwiseXor(a_int, b_int);
defer bxor.deinit();
var bnot = try num.ops.bitwiseNot(a_int);
defer bnot.deinit();
var shl = try num.ops.leftShift(a_int, b_int);
defer shl.deinit();
var shr = try num.ops.rightShift(a_int, b_int);
defer shr.deinit();
var pc = try num.ops.bitCount(a_int);
defer pc.deinit();
```

---

## 6. Complex Helpers

Complex dtypes (`c64`, `c128`); comparisons on complex arrays are unsupported:

```zig
var c = try num.ops.conj(z);
defer c.deinit();
var re = try num.ops.real(z);
defer re.deinit();
var im = try num.ops.imag(z);
defer im.deinit();
var mag = try num.ops.magnitude(z);
defer mag.deinit();
var ph = try num.ops.phase(z);
defer ph.deinit();
```

---

## 7. Comparisons & Logical Operations

Broadcasting comparisons return boolean arrays; logical operators combine them.
See the [Comparisons & Logic API](/api/compare) for full signatures:

```zig
var eq = try num.ops.equal(a, b);
defer eq.deinit();
var lt = try num.ops.less(a, b);
defer lt.deinit();
var both = try num.ops.logicalAnd(eq, lt);
defer both.deinit();
var either = try num.ops.logicalOr(eq, lt);
defer either.deinit();
var negated = try num.ops.logicalNot(eq);
defer negated.deinit();

// Floating-point predicates and tolerance checks
var nan_mask = try num.ops.isNaN(a);
defer nan_mask.deinit();
var finite_mask = try num.ops.isFinite(a);
defer finite_mask.deinit();
var close = try num.ops.isClose(a, b, .{ .rtol = 1e-5, .atol = 1e-8 });
defer close.deinit();
const all_close = try num.ops.allClose(a, b, .{});
```
