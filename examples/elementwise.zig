const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const d1 = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const d2 = [_]f64{ 10.0, 20.0, 30.0, 40.0 };

    var a = try num.fromSlice(allocator, f64, .{ .data = &d1, .shape = &.{4} });
    defer a.deinit();
    var b = try num.fromSlice(allocator, f64, .{ .data = &d2, .shape = &.{4} });
    defer b.deinit();

    // 1. Arithmetic: add, subtract, multiply, divide
    var sum = try num.ops.add(a, b, .{});
    defer sum.deinit();
    var prod = try num.ops.multiply(a, b, .{});
    defer prod.deinit();

    std.debug.print("Elementwise Arithmetic:\n", .{});
    std.debug.print("  a + b: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{
        try sum.get(f64, &.{0}),
        try sum.get(f64, &.{1}),
        try sum.get(f64, &.{2}),
        try sum.get(f64, &.{3}),
    });
    std.debug.print("  a * b: [{d:.1}, {d:.1}, {d:.1}, {d:.1}]\n", .{
        try prod.get(f64, &.{0}),
        try prod.get(f64, &.{1}),
        try prod.get(f64, &.{2}),
        try prod.get(f64, &.{3}),
    });

    // 2. Math functions: sqrt, exp, log
    var sqrt_b = try num.ops.sqrt(b, .{});
    defer sqrt_b.deinit();
    std.debug.print("  sqrt(b)[3] (sqrt(40.0)): {d:.4}\n", .{try sqrt_b.get(f64, &.{3})});

    // 3. Comparisons: greater, equal
    var is_gt = try num.ops.greater(b, a);
    defer is_gt.deinit();
    std.debug.print("  b > a [0]: {}\n", .{try is_gt.get(bool, &.{0})});

    // 4. Conditional where selection
    var selected = try num.ops.where(is_gt, b, a, .{});
    defer selected.deinit();
    std.debug.print("  where(b > a, b, a)[0]: {d:.1}\n", .{try selected.get(f64, &.{0})});

    // 5. Reciprocal and base-2 exponential
    var recip = try num.ops.reciprocal(b, .{});
    defer recip.deinit();
    std.debug.print("  reciprocal(b)[0]: {d:.2}\n", .{try recip.get(f64, &.{0})});
}
