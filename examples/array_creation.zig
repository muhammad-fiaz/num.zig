//! Demonstrates N-dimensional array creation in num.zig.
//!
//! Covers zeros, ones, full, arange, linspace, eye, and fromSlice.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Create array filled with zeros
    var z = try num.zeros(allocator, .{ .shape = &.{ 2, 3 }, .dtype = .f64 });
    defer z.deinit();
    std.debug.print("Zeros (2x3):\n  ndim: {d}, elements: {d}\n", .{ z.ndim, z.elementCount() });

    // 2. Create array filled with ones
    var o = try num.ones(allocator, .{ .shape = &.{ 3, 3 }, .dtype = .f32 });
    defer o.deinit();
    std.debug.print("Ones (3x3):\n  ndim: {d}, val[0,0]: {d:.1}\n", .{ o.ndim, try o.get(f32, &.{ 0, 0 }) });

    // 3. Create array filled with a constant value
    var f = try num.full(allocator, .{ .shape = &.{ 2, 2 }, .value = @as(i32, 42) });
    defer f.deinit();
    std.debug.print("Full (2x2 with 42):\n  val[1,1]: {d}\n", .{try f.get(i32, &.{ 1, 1 })});

    // 4. Create arithmetic progression with arange
    var r = try num.arange(allocator, .{ .start = 0, .stop = 10, .step = 2, .dtype = .i64 });
    defer r.deinit();
    std.debug.print("Arange [0..10 step 2]:\n  length: {d}, first: {d}, last: {d}\n", .{
        r.elementCount(),
        try r.get(i64, &.{0}),
        try r.get(i64, &.{r.elementCount() - 1}),
    });

    // 5. Create linearly spaced intervals with linspace
    var ls = try num.linspace(allocator, .{ .start = 0.0, .stop = 1.0, .num = 5, .dtype = .f64 });
    defer ls.deinit();
    std.debug.print("Linspace [0.0..1.0 (5 steps)]:\n", .{});
    for (0..ls.elementCount()) |i| {
        std.debug.print("  [{d}] = {d:.2}\n", .{ i, try ls.get(f64, &.{i}) });
    }

    // 6. Identity matrix
    var eye_mat = try num.eye(allocator, .{ .n = 3, .dtype = .f64 });
    defer eye_mat.deinit();
    std.debug.print("Identity (3x3):\n  diag[1,1]: {d:.1}, off-diag[0,1]: {d:.1}\n", .{
        try eye_mat.get(f64, &.{ 1, 1 }),
        try eye_mat.get(f64, &.{ 0, 1 }),
    });

    // 7. From native slice
    const raw = [_]f64{ 1.1, 2.2, 3.3, 4.4 };
    var from_raw = try num.fromSlice(allocator, f64, .{ .data = &raw, .shape = &.{ 2, 2 } });
    defer from_raw.deinit();
    std.debug.print("FromSlice (2x2):\n  val[1,0]: {d:.1}\n", .{try from_raw.get(f64, &.{ 1, 0 })});
}
