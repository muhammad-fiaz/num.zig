//! Demonstrates typed scalar access, assignment, views, and index selection.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Typed scalar access across dtypes and ranks.
    const f32_data = [_]f32{ 1.5, 2.5, 3.5 };
    var v32 = try num.fromSlice(allocator, f32, .{ .data = &f32_data, .shape = &.{3} });
    defer v32.deinit();
    std.debug.print("get(f32, [0]): {d:.1}\n", .{try v32.get(f32, &.{0})});

    const f64_data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var m64 = try num.fromSlice(allocator, f64, .{ .data = &f64_data, .shape = &.{ 2, 3 } });
    defer m64.deinit();
    std.debug.print("get(f64, [1, 2]): {d:.1}\n", .{try m64.get(f64, &.{ 1, 2 })});

    const i32_data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var t32 = try num.fromSlice(allocator, i32, .{ .data = &i32_data, .shape = &.{ 2, 2, 2 } });
    defer t32.deinit();
    std.debug.print("get(i32, [1, 0, 1]): {d}\n", .{try t32.get(i32, &.{ 1, 0, 1 })});

    // Scalar (0D) array.
    var s = try num.scalar(allocator, .{ .value = @as(f64, 42.0) });
    defer s.deinit();
    std.debug.print("scalar: {d:.1}\n", .{try s.get(f64, &.{})});

    // Assignment through a transposed view affects the underlying storage.
    try m64.set(f64, &.{ 0, 1 }, 99.0);
    var tv = try num.manip.transpose(m64, .{});
    defer tv.deinit();
    std.debug.print("view[1, 0] after set: {d:.1}\n", .{try tv.get(f64, &.{ 1, 0 })});

    // Index-based selection with take, and scalar broadcast via full.
    const idx_data = [_]i64{ 2, 0 };
    var idx = try num.fromSlice(allocator, i64, .{ .data = &idx_data, .shape = &.{2} });
    defer idx.deinit();
    var picked = try v32.take(idx, .{});
    defer picked.deinit();
    std.debug.print("take(v32, [2, 0]): [{d:.1}, {d:.1}]\n", .{
        try picked.get(f32, &.{0}), try picked.get(f32, &.{1}),
    });
}
