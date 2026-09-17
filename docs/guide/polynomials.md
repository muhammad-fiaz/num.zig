# Polynomial Calculus (`num.poly`)

`num.zig` provides a polynomial evaluation and calculus module representing polynomials as 1D coefficient arrays ordered by ascending powers:
$$P(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_n x^n$$

---

## 1. Polynomial Evaluation (`polyval`)

Evaluates a polynomial at one or more scalar/array points using Horner's method:

```zig
// Represents P(x) = 1 + 2x + 3x^2
var coeffs = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 1.0, 2.0, 3.0 },
    .shape = &.{3},
});
defer coeffs.deinit();

var x_points = try num.fromSlice(allocator, f64, .{
    .data = &[_]f64{ 0.0, 1.0, 2.0 },
    .shape = &.{3},
});
defer x_points.deinit();

// At x=0: 1, at x=1: 6, at x=2: 17
var y_vals = try num.poly.val(allocator, f64, &coeffs, &x_points);
defer y_vals.deinit();
```

---

## 2. Differentiation & Integration

### `der` (Derivative)
Computes the derivative of polynomial coefficients:
$$P'(x) = c_1 + 2 c_2 x + \dots + n c_n x^{n-1}$$

```zig
var d1 = try num.poly.der(allocator, f64, &coeffs, 1);
defer d1.deinit(); // [2.0, 6.0] -> 2 + 6x
```

### `integ` (Indefinite Integral)
Computes the anti-derivative with a specified integration constant $k$:
```zig
var integral = try num.poly.integ(allocator, f64, &coeffs, 1, 0.0);
defer integral.deinit(); // [0.0, 1.0, 1.0, 1.0] -> x + x^2 + x^3
```

---

## 3. Polynomial Arithmetic

- **`add`**: Sums two polynomials.
- **`sub`**: Subtracts two polynomials.
- **`mul`**: Polynomial multiplication (discrete convolution).
- **`div`**: Polynomial long division returning quotient and remainder.

---

## 4. Least-Squares Curve Fitting (`polyfit`)

Fits a polynomial of specified degree $m$ to $(x, y)$ coordinates using Vandermonde matrix linear regression:

```zig
var p_fit = try num.poly.fit(allocator, f64, &x_points, &y_vals, 2);
defer p_fit.deinit();
```
