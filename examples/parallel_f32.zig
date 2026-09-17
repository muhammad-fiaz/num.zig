//! Demonstrates multi-threaded CPU parallel execution with f32 precision and typed scalar access.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const N: usize = 50_000;
    var a = try num.empty(allocator, .{ .shape = &.{N}, .dtype = .f32 });
    defer a.deinit();

    const Context = struct {
        arr: num.Array,
    };
    const ctx = Context{ .arr = a };

    num.parallel.run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| {
                const x: f32 = @floatFromInt(i);
                c.arr.set(f32, &.{i}, x * 0.1) catch unreachable;
            }
        }
    }.kernel, .{ .workers = 4, .threshold = 1000 });

    std.debug.print("Parallel f32 array processing:\n", .{});
    const val0 = try a.get(f32, &.{0});
    const val10 = try a.get(f32, &.{10});
    const val500 = try a.get(f32, &.{500});

    std.debug.print("  a[0]   (f32) = {d:.2}\n", .{val0});
    std.debug.print("  a[10]  (f32) = {d:.2}\n", .{val10});
    std.debug.print("  a[500] (f32) = {d:.2}\n", .{val500});
}
