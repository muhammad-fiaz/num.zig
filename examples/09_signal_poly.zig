const std = @import("std");
const num = @import("num");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Signal Processing & Polynomials Example ===\n\n", .{});

    // --- Signal Processing ---
    std.debug.print("--- Convolution ---\n", .{});
    var sig = try num.NDArray(f32).init(allocator, &.{5});
    defer sig.deinit(allocator);
    // Signal: [1, 2, 3, 4, 5]
    for (0..5) |i| try sig.set(&.{i}, @floatFromInt(i + 1));

    var kernel = try num.NDArray(f32).init(allocator, &.{3});
    defer kernel.deinit(allocator);
    // Kernel: [0.5, 1.0, 0.5]
    try kernel.set(&.{0}, 0.5);
    try kernel.set(&.{1}, 1.0);
    try kernel.set(&.{2}, 0.5);

    var conv_full = try num.signal.convolve(allocator, f32, sig, kernel, .full);
    defer conv_full.deinit(allocator);

    std.debug.print("Signal: [1, 2, 3, 4, 5]\n", .{});
    std.debug.print("Kernel: [0.5, 1.0, 0.5]\n", .{});
    std.debug.print("Full Convolution: ", .{});
    for (0..conv_full.size()) |i| {
        std.debug.print("{d:.2} ", .{try conv_full.get(&.{i})});
    }
    std.debug.print("\n\n", .{});

    // --- Polynomials ---
    std.debug.print("--- Polynomials ---\n", .{});
    // p(x) = x^2 + 2x + 3 -> [1, 2, 3]
    var p = try num.NDArray(f32).init(allocator, &.{3});
    defer p.deinit(allocator);
    try p.set(&.{0}, 1);
    try p.set(&.{1}, 2);
    try p.set(&.{2}, 3);

    // Evaluate at x = 2
    // 2^2 + 2*2 + 3 = 4 + 4 + 3 = 11
    var x = try num.NDArray(f32).init(allocator, &.{1});
    defer x.deinit(allocator);
    try x.set(&.{0}, 2);

    var y = try num.poly.polyval(allocator, f32, p, x);
    defer y.deinit(allocator);
    std.debug.print("polyval([1, 2, 3], 2) = {d:.2}\n", .{try y.get(&.{0})});

    // Poly Add
    // q(x) = 4x + 5 -> [4, 5]
    var q = try num.NDArray(f32).init(allocator, &.{2});
    defer q.deinit(allocator);
    try q.set(&.{0}, 4);
    try q.set(&.{1}, 5);

    var pq = try num.poly.polyadd(allocator, f32, p, q);
    defer pq.deinit(allocator);
    // Result: x^2 + 6x + 8 -> [1, 6, 8]
    std.debug.print("polyadd: ", .{});
    for (0..pq.size()) |i| {
        std.debug.print("{d:.1} ", .{try pq.get(&.{i})});
    }
    std.debug.print("\n\n", .{});

    // --- Calculus ---
    std.debug.print("--- Calculus ---\n", .{});
    // diff([1, 2, 4, 7, 0]) -> [1, 2, 3, -7]
    var d = try num.NDArray(f32).init(allocator, &.{5});
    defer d.deinit(allocator);
    try d.set(&.{0}, 1);
    try d.set(&.{1}, 2);
    try d.set(&.{2}, 4);
    try d.set(&.{3}, 7);
    try d.set(&.{4}, 0);

    var diff1 = try num.diff.diff(allocator, f32, d, 1, 0);
    defer diff1.deinit(allocator);

    std.debug.print("diff([1, 2, 4, 7, 0]): ", .{});
    for (0..diff1.size()) |i| {
        std.debug.print("{d:.1} ", .{try diff1.get(&.{i})});
    }
    std.debug.print("\n", .{});
}
