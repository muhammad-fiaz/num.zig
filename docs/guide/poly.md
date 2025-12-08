# Polynomials

The `poly` module provides comprehensive polynomial arithmetic operations including evaluation, addition, multiplication, differentiation, integration, and root finding. Essential for numerical analysis and curve fitting.

## Polynomial Evaluation

Evaluate a polynomial at given points using Horner's method:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Polynomial: p(x) = 2x^2 + 3x + 1
    // Coefficients in descending order: [2, 3, 1]
    const p_shape = [_]usize{3};
    const p_data = [_]f32{ 2.0, 3.0, 1.0 };
    var p = num.core.NDArray(f32).init(&p_shape, @constCast(&p_data));
    
    // Evaluate at x = [0, 1, 2, 3]
    const x_shape = [_]usize{4};
    const x_data = [_]f32{ 0.0, 1.0, 2.0, 3.0 };
    var x = num.core.NDArray(f32).init(&x_shape, @constCast(&x_data));
    
    var result = try num.poly.polyval(allocator, f32, p, x);
    defer result.deinit(allocator);
    
    std.debug.print("p(x) values: {any}\n", .{result.data});
}
// Output:
// p(x) values: [1.0, 6.0, 15.0, 28.0]
// p(0) = 1, p(1) = 6, p(2) = 15, p(3) = 28
```

## Polynomial Addition

Add two polynomials:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // p1(x) = 3x^2 + 2x + 1
    const p1_shape = [_]usize{3};
    const p1_data = [_]f32{ 3.0, 2.0, 1.0 };
    var p1 = num.core.NDArray(f32).init(&p1_shape, @constCast(&p1_data));
    
    // p2(x) = x^2 + 4x + 5
    const p2_shape = [_]usize{3};
    const p2_data = [_]f32{ 1.0, 4.0, 5.0 };
    var p2 = num.core.NDArray(f32).init(&p2_shape, @constCast(&p2_data));
    
    // p1 + p2 = 4x^2 + 6x + 6
    var sum = try num.poly.polyadd(allocator, f32, p1, p2);
    defer sum.deinit(allocator);
    
    std.debug.print("p1 + p2 coefficients: {any}\n", .{sum.data});
}
// Output:
// p1 + p2 coefficients: [4.0, 6.0, 6.0]
// Result: 4x^2 + 6x + 6
```

## Polynomial Subtraction

Subtract one polynomial from another:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // p1(x) = 5x^2 + 3x + 2
    const p1_shape = [_]usize{3};
    const p1_data = [_]f32{ 5.0, 3.0, 2.0 };
    var p1 = num.core.NDArray(f32).init(&p1_shape, @constCast(&p1_data));
    
    // p2(x) = 2x^2 + x + 1
    const p2_shape = [_]usize{3};
    const p2_data = [_]f32{ 2.0, 1.0, 1.0 };
    var p2 = num.core.NDArray(f32).init(&p2_shape, @constCast(&p2_data));
    
    // p1 - p2 = 3x^2 + 2x + 1
    var diff = try num.poly.polysub(allocator, f32, p1, p2);
    defer diff.deinit(allocator);
    
    std.debug.print("p1 - p2 coefficients: {any}\n", .{diff.data});
}
// Output:
// p1 - p2 coefficients: [3.0, 2.0, 1.0]
// Result: 3x^2 + 2x + 1
```

## Polynomial Multiplication

Multiply two polynomials:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // p1(x) = x + 1
    const p1_shape = [_]usize{2};
    const p1_data = [_]f32{ 1.0, 1.0 };
    var p1 = num.core.NDArray(f32).init(&p1_shape, @constCast(&p1_data));
    
    // p2(x) = x + 2
    const p2_shape = [_]usize{2};
    const p2_data = [_]f32{ 1.0, 2.0 };
    var p2 = num.core.NDArray(f32).init(&p2_shape, @constCast(&p2_data));
    
    // (x + 1)(x + 2) = x^2 + 3x + 2
    var product = try num.poly.polymul(allocator, f32, p1, p2);
    defer product.deinit(allocator);
    
    std.debug.print("p1 * p2 coefficients: {any}\n", .{product.data});
}
// Output:
// p1 * p2 coefficients: [1.0, 3.0, 2.0]
// Result: x^2 + 3x + 2
```

## Polynomial Derivative

Compute the derivative of a polynomial:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // p(x) = 3x^3 + 2x^2 + 5x + 1
    const p_shape = [_]usize{4};
    const p_data = [_]f32{ 3.0, 2.0, 5.0, 1.0 };
    var p = num.core.NDArray(f32).init(&p_shape, @constCast(&p_data));
    
    // First derivative: p'(x) = 9x^2 + 4x + 5
    var deriv = try num.poly.polyder(allocator, f32, p, 1);
    defer deriv.deinit(allocator);
    
    std.debug.print("p'(x) coefficients: {any}\n", .{deriv.data});
}
// Output:
// p'(x) coefficients: [9.0, 4.0, 5.0]
// Result: 9x^2 + 4x + 5
```

## Higher Order Derivatives

Compute second or higher derivatives:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // p(x) = x^4 + 2x^3 + 3x^2 + 4x + 5
    const p_shape = [_]usize{5};
    const p_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var p = num.core.NDArray(f32).init(&p_shape, @constCast(&p_data));
    
    // Second derivative: p''(x) = 12x^2 + 12x + 6
    var deriv2 = try num.poly.polyder(allocator, f32, p, 2);
    defer deriv2.deinit(allocator);
    
    std.debug.print("p''(x) coefficients: {any}\n", .{deriv2.data});
}
// Output:
// p''(x) coefficients: [12.0, 12.0, 6.0]
// Result: 12x^2 + 12x + 6
```

## Polynomial Integration

Compute the indefinite integral of a polynomial:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // p(x) = 3x^2 + 2x + 1
    const p_shape = [_]usize{3};
    const p_data = [_]f32{ 3.0, 2.0, 1.0 };
    var p = num.core.NDArray(f32).init(&p_shape, @constCast(&p_data));
    
    // Integral with constant k=0: ∫p(x)dx = x^3 + x^2 + x
    var integral = try num.poly.polyint(allocator, f32, p, 1, 0.0);
    defer integral.deinit(allocator);
    
    std.debug.print("∫p(x)dx coefficients: {any}\n", .{integral.data});
}
// Output:
// ∫p(x)dx coefficients: [1.0, 1.0, 1.0, 0.0]
// Result: x^3 + x^2 + x + 0
```

## Polynomial Roots

Find the roots of a polynomial:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // p(x) = x^2 - 5x + 6 = (x-2)(x-3)
    // Roots should be x = 2 and x = 3
    const p_shape = [_]usize{3};
    const p_data = [_]f32{ 1.0, -5.0, 6.0 };
    var p = num.core.NDArray(f32).init(&p_shape, @constCast(&p_data));
    
    var roots_arr = try num.poly.roots(allocator, f32, p);
    defer roots_arr.deinit(allocator);
    
    std.debug.print("Roots: {any}\n", .{roots_arr.data});
}
// Output:
// Roots: [3.0, 2.0]
// (Order may vary)
```

## Practical Example: Curve Fitting

Fit a polynomial to data points and evaluate:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Fitted polynomial (from curve fitting): p(x) = 0.5x^2 + 1.5x + 2
    const p_shape = [_]usize{3};
    const p_data = [_]f32{ 0.5, 1.5, 2.0 };
    var p = num.core.NDArray(f32).init(&p_shape, @constCast(&p_data));
    
    // Evaluate at new points
    const x_shape = [_]usize{6};
    const x_data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0, 3.0 };
    var x = num.core.NDArray(f32).init(&x_shape, @constCast(&x_data));
    
    var predictions = try num.poly.polyval(allocator, f32, p, x);
    defer predictions.deinit(allocator);
    
    std.debug.print("Curve predictions:\n", .{});
    for (0..6) |i| {
        std.debug.print("  x={d:.1}, y={d:.2}\n", .{ x_data[i], predictions.data[i] });
    }
}
// Output:
// Curve predictions:
//   x=-2.0, y=0.00
//   x=-1.0, y=1.00
//   x=0.0, y=2.00
//   x=1.0, y=4.00
//   x=2.0, y=7.00
//   x=3.0, y=11.00
```

## Practical Example: Taylor Series

Construct and evaluate a Taylor series approximation:

```zig
const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Taylor series for e^x around x=0: 1 + x + x^2/2! + x^3/3! + ...
    // Approximation: 1 + x + 0.5x^2 + 0.1667x^3
    const p_shape = [_]usize{4};
    const p_data = [_]f32{ 0.1667, 0.5, 1.0, 1.0 };
    var taylor_exp = num.core.NDArray(f32).init(&p_shape, @constCast(&p_data));
    
    // Evaluate at x = [0, 0.5, 1.0]
    const x_shape = [_]usize{3};
    const x_data = [_]f32{ 0.0, 0.5, 1.0 };
    var x = num.core.NDArray(f32).init(&x_shape, @constCast(&x_data));
    
    var approx = try num.poly.polyval(allocator, f32, taylor_exp, x);
    defer approx.deinit(allocator);
    
    std.debug.print("Taylor approximation vs actual e^x:\n", .{});
    for (0..3) |i| {
        const actual = @exp(x_data[i]);
        std.debug.print("  x={d:.1}: approx={d:.4}, actual={d:.4}\n", .{
            x_data[i],
            approx.data[i],
            actual,
        });
    }
}
// Output:
// Taylor approximation vs actual e^x:
//   x=0.0: approx=1.0000, actual=1.0000
//   x=0.5: approx=1.6458, actual=1.6487
//   x=1.0: approx=2.6667, actual=2.7183
```