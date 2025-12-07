# Polynomials

`num.zig` provides a `num.poly` module for working with polynomials.

## Basics

Polynomials are represented by their coefficients. For example, $3x^2 + 2x + 1$ is represented as `[3, 2, 1]`.

```zig
const num = @import("num");

// Define p(x) = 3x^2 + 2x + 1
var p = try num.NDArray(f32).init(allocator, &.{3});
try p.set(&.{0}, 3);
try p.set(&.{1}, 2);
try p.set(&.{2}, 1);
```

## Evaluation

You can evaluate a polynomial at specific points using `num.poly.polyval`.

```zig
// Evaluate at x = 2
var x = try num.NDArray(f32).init(allocator, &.{1});
try x.set(&.{0}, 2);

var y = try num.poly.polyval(allocator, f32, p, x);
// y is [17] (3*4 + 2*2 + 1 = 17)
```

## Arithmetic

You can add, subtract, and multiply polynomials.

```zig
var p1 = ...; // x + 1
var p2 = ...; // x - 1

var sum = try num.poly.polyadd(allocator, f32, p1, p2); // 2x
var prod = try num.poly.polymul(allocator, f32, p1, p2); // x^2 - 1
```

## Roots

Find the roots of a polynomial using `num.poly.roots`.

```zig
// Roots of x^2 - 1 are 1 and -1
var roots = try num.poly.roots(allocator, f32, prod);
```
