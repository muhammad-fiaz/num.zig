# Polynomial API

Module: `@import("num").poly`

Coefficients are stored with highest degree first: `[c_deg, c_{deg-1}, ..., c_0]`.

---

## Evaluation

```zig
/// Evaluate polynomial `coeffs` at points `x` using Horner's method.
/// `coeffs` must be 1D. `x` can have any shape; output matches `x` shape.
pub fn val(coeffs: Array, x: Array) !Array;
```

---

## Calculus

```zig
/// Compute the m-th derivative of polynomial `coeffs`.
pub fn der(coeffs: Array, m: usize) !Array;

/// Compute the indefinite integral of polynomial `coeffs`, with integration constant `k`.
pub fn integ(coeffs: Array, k: f64) !Array;
```

---

## Regression / Curve Fitting

```zig
/// Fit a polynomial of degree `deg` to data points `(x, y)` via QR least-squares.
/// Returns coefficients highest degree first: [c_deg, ..., c_0].
pub fn fit(x: Array, y: Array, deg: usize) !Array;
```

---

## Roots

```zig
/// Find polynomial roots. Returns a 1D c128 array of length deg.
/// Linear/quadratic cases are analytic; higher degrees use Durand-Kerner iteration.
pub fn roots(coeffs: Array) !Array;
```

---

## Coefficient Arithmetic

```zig
/// Add, subtract, or multiply polynomials (highest-degree-first, trimmed).
pub fn add(a: Array, b: Array) !Array;
pub fn sub(a: Array, b: Array) !Array;
pub fn mul(a: Array, b: Array) !Array;
```

