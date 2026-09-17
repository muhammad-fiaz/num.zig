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
  var out = try num.ops.where(cond_arr, x_arr, y_arr, .{});
  defer out.deinit();
  ```
