const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Math Example ===\n\n", .{});

    var a = try num.NDArray(f32).ones(allocator, &.{ 2, 2 });
    defer a.deinit();
    var b = try num.NDArray(f32).full(allocator, &.{ 2, 2 }, 2.0);
    defer b.deinit();

    // Elementwise Add
    var c = try num.ops.add(f32, allocator, &a, &b);
    defer c.deinit();
    std.debug.print("Ones + Twos:\n", .{});
    for (0..c.shape[0]) |i| {
        for (0..c.shape[1]) |j| {
            std.debug.print("{d:.1} ", .{try c.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Stats
    const mean_val = try num.stats.mean(f32, &c);
    std.debug.print("\nMean: {d:.2}\n", .{mean_val});

    const sum_val = try num.stats.sum(f32, &c);
    std.debug.print("Sum: {d:.2}\n", .{sum_val});

    // Broadcasting
    var d = try num.NDArray(f32).init(allocator, &.{ 1, 2 });
    defer d.deinit();
    try d.set(&.{ 0, 0 }, 10);
    try d.set(&.{ 0, 1 }, 20);

    var e = try num.ops.add(f32, allocator, &c, &d);
    defer e.deinit();
    std.debug.print("\nBroadcasting Add (3.0 + [10, 20]):\n", .{});
    for (0..e.shape[0]) |i| {
        for (0..e.shape[1]) |j| {
            std.debug.print("{d:.1} ", .{try e.get(&.{ i, j })});
        }
        std.debug.print("\n", .{});
    }

    // Trigonometry
    std.debug.print("\n--- Trigonometry ---\n", .{});
    var angles = try num.NDArray(f32).linspace(allocator, 0, std.math.pi, 3);
    defer angles.deinit();

    var sines = try num.elementwise.sin(allocator, f32, angles);
    defer sines.deinit();
    std.debug.print("Sin(0, pi/2, pi): ", .{});
    for (0..sines.size()) |i| {
        std.debug.print("{d:.2} ", .{try sines.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Logical
    std.debug.print("\n--- Logical ---\n", .{});
    var bool_a = try num.NDArray(bool).init(allocator, &.{2});
    defer bool_a.deinit();
    try bool_a.set(&.{0}, true);
    try bool_a.set(&.{1}, false);

    var bool_b = try num.NDArray(bool).init(allocator, &.{2});
    defer bool_b.deinit();
    try bool_b.set(&.{0}, true);
    try bool_b.set(&.{1}, true);

    var bool_and = try num.elementwise.logical_and(allocator, bool_a, bool_b);
    defer bool_and.deinit();
    std.debug.print("T/F AND T/T: {any} {any}\n", .{ try bool_and.get(&.{0}), try bool_and.get(&.{1}) });

    // Power, Exp, Log, Sqrt
    std.debug.print("\n--- Power, Exp, Log, Sqrt ---\n", .{});
    var base = try num.NDArray(f32).full(allocator, &.{3}, 2.0);
    defer base.deinit();
    var exponent = try num.NDArray(f32).linspace(allocator, 1, 3, 3); // 1, 2, 3
    defer exponent.deinit();

    var power = try num.elementwise.pow(allocator, f32, base, exponent);
    defer power.deinit();
    std.debug.print("2^[1, 2, 3]: ", .{});
    for (0..power.size()) |i| {
        std.debug.print("{d:.1} ", .{try power.get(&.{i})});
    }
    std.debug.print("\n", .{});

    var sqrt_val = try num.elementwise.sqrt(allocator, f32, power);
    defer sqrt_val.deinit();
    std.debug.print("Sqrt: ", .{});
    for (0..sqrt_val.size()) |i| {
        std.debug.print("{d:.2} ", .{try sqrt_val.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Rounding & Clipping
    std.debug.print("\n--- Rounding & Clipping ---\n", .{});
    var vals = try num.NDArray(f32).init(allocator, &.{5});
    defer vals.deinit();
    // -1.5, -0.5, 0.5, 1.5, 2.5
    try vals.set(&.{0}, -1.5);
    try vals.set(&.{1}, -0.5);
    try vals.set(&.{2}, 0.5);
    try vals.set(&.{3}, 1.5);
    try vals.set(&.{4}, 2.5);

    var clipped = try num.elementwise.clip(allocator, f32, vals, -1.0, 2.0);
    defer clipped.deinit();
    std.debug.print("Clip [-1.5, ..., 2.5] to [-1, 2]: ", .{});
    for (0..clipped.size()) |i| {
        std.debug.print("{d:.1} ", .{try clipped.get(&.{i})});
    }
    std.debug.print("\n", .{});

    var rounded = try num.elementwise.round(allocator, f32, vals);
    defer rounded.deinit();
    std.debug.print("Round: ", .{});
    for (0..rounded.size()) |i| {
        std.debug.print("{d:.1} ", .{try rounded.get(&.{i})});
    }
    std.debug.print("\n", .{});

    var abs_vals = try num.elementwise.abs(allocator, f32, vals);
    defer abs_vals.deinit();
    std.debug.print("Abs: ", .{});
    for (0..abs_vals.size()) |i| {
        std.debug.print("{d:.1} ", .{try abs_vals.get(&.{i})});
    }
    std.debug.print("\n", .{});

    // Min/Max/Argmin/Argmax
    std.debug.print("\n--- Min/Max/Argmin/Argmax ---\n", .{});
    var m = try num.NDArray(f32).init(allocator, &.{5});
    defer m.deinit();
    try m.set(&.{0}, 10);
    try m.set(&.{1}, 2);
    try m.set(&.{2}, 5);
    try m.set(&.{3}, 20);
    try m.set(&.{4}, 1);

    std.debug.print("Array: 10 2 5 20 1\n", .{});
    std.debug.print("Min: {d}\n", .{try num.stats.min(f32, &m)});
    std.debug.print("Max: {d}\n", .{try num.stats.max(f32, &m)});
    std.debug.print("Argmin: {d}\n", .{try num.stats.argmin(f32, &m)});
    std.debug.print("Argmax: {d}\n", .{try num.stats.argmax(f32, &m)});

    // Clip (if available, otherwise skip or implement manually to show how)
    // Assuming clip might not be in elementwise yet, I'll skip it or check later.
    // But I can show abs/neg

    std.debug.print("\n--- Negate ---\n", .{});
    var neg = try num.elementwise.neg(allocator, f32, m);
    defer neg.deinit();
    std.debug.print("Negated: ", .{});
    for (0..neg.size()) |i| {
        std.debug.print("{d} ", .{try neg.get(&.{i})});
    }
    std.debug.print("\n", .{});
}
