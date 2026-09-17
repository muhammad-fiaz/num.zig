//! Demonstrates polynomial evaluation, differentiation, integration, and curve fitting.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Polynomial evaluation using Horner's method
    // p(x) = 2x^2 - 3x + 5: coeffs = [2, -3, 5]
    const c_data = [_]f64{ 2.0, -3.0, 5.0 };
    var coeffs = try num.fromSlice(allocator, f64, .{ .data = &c_data, .shape = &.{3} });
    defer coeffs.deinit();

    // Evaluate at x = 2.0 -> 2(4) - 3(2) + 5 = 7.0
    const x_val = [_]f64{2.0};
    var x = try num.fromSlice(allocator, f64, .{ .data = &x_val, .shape = &.{} });
    defer x.deinit();

    var eval_res = try num.poly.val(coeffs, x);
    defer eval_res.deinit();
    std.debug.print("Poly val p(2.0) for 2x^2 - 3x + 5: {d:.1} (expected 7.0)\n", .{try eval_res.get(f64, &.{})});

    // 2. Polynomial derivative: p'(x) = 4x - 3
    var deriv = try num.poly.der(coeffs, 1);
    defer deriv.deinit();
    std.debug.print("Poly der p'(x): [{d:.1}, {d:.1}]\n", .{
        try deriv.get(f64, &.{0}),
        try deriv.get(f64, &.{1}),
    });

    // 3. Polynomial curve fitting (linear regression on y = 2x + 1)
    const pts_x = [_]f64{ 0.0, 1.0, 2.0, 3.0 };
    const pts_y = [_]f64{ 1.0, 3.0, 5.0, 7.0 };
    var x_arr = try num.fromSlice(allocator, f64, .{ .data = &pts_x, .shape = &.{4} });
    defer x_arr.deinit();
    var y_arr = try num.fromSlice(allocator, f64, .{ .data = &pts_y, .shape = &.{4} });
    defer y_arr.deinit();

    var fit = try num.poly.fit(x_arr, y_arr, 1);
    defer fit.deinit();
    std.debug.print("Poly fit line: slope = {d:.4}, intercept = {d:.4}\n", .{
        try fit.get(f64, &.{0}),
        try fit.get(f64, &.{1}),
    });
}
