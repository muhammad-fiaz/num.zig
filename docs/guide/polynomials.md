# Polynomial Calculus (`num.poly`)

`num.zig` represents polynomials as 1D coefficient arrays ordered highest-degree first:
`[c_n, ..., c_0]` for `P(x) = c_n x^n + ... + c_0`.

---

## 1. Polynomial Evaluation (`val`)

Evaluates a polynomial at array points using Horner's method. Output matches `x` shape:

```zig
// Represents P(x) = 2x^2 - 3x + 5.
var coeffs = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 2.0, -3.0, 5.0 },
    .shape = &.{3},
});
defer coeffs.deinit();

var x = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{2.0},
    .shape = &. {},
});
defer x.deinit();

// p(2.0) = 7.0
var y = try num.poly.val(coeffs, x);
defer y.deinit();
```

---

## 2. Differentiation & Integration

### `der` (Derivative)

```zig
// p(x) = 3x^2 + 4x + 5 -> p'(x) = 6x + 4
var d1 = try num.poly.der(coeffs, 1);
defer d1.deinit();
```

### `integ` (Indefinite Integral)

```zig
var integral = try num.poly.integ(coeffs, 0.0);
defer integral.deinit();
```

---

## 3. Polynomial Arithmetic

```zig
var s = try num.poly.add(pa, pb);
defer s.deinit();
var d = try num.poly.sub(pa, pb);
defer d.deinit();
var m = try num.poly.mul(pa, pb);
defer m.deinit();
```

---

## 4. Root Finding (`roots`)

Returns complex roots (`c128`). Linear/quadratic cases are analytic; higher
degrees use Durand-Kerner iteration:

```zig
// x^2 - 5x + 6 = (x-2)(x-3)
var rts = try num.poly.roots(coeffs);
defer rts.deinit();
```

---

## 5. Least-Squares Curve Fitting (`fit`)

Fits a polynomial of degree `deg` to 1D `(x, y)` points via QR least-squares:

```zig
var p_fit = try num.poly.fit(x_points, y_vals, 1);
defer p_fit.deinit();
```
