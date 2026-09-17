//! Demonstrates multi-threaded CPU parallel execution with f64 precision and typed scalar access.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const N: usize = 50_000;
    var a = try num.empty(allocator, .{ .shape = &.{N}, .dtype = .f64 });
    defer a.deinit();

    const Context = struct {
        arr: num.Array,
    };
    const ctx = Context{ .arr = a };

    num.parallel.run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| {
                const x: f64 = @floatFromInt(i);
                c.arr.set(f64, &.{i}, std.math.exp(-0.001 * x)) catch unreachable;
            }
        }
    }.kernel, .{ .workers = 4, .threshold = 1000 });

    std.debug.print("Parallel f64 exponential decay processing:\n", .{});
    const val0 = try a.get(f64, &.{0});
    const val1000 = try a.get(f64, &.{1000});
    const val5000 = try a.get(f64, &.{5000});

    std.debug.print("  exp(-0.001 * 0)    (f64) = {d:.6}\n", .{val0});
    std.debug.print("  exp(-0.001 * 1000) (f64) = {d:.6}\n", .{val1000});
    std.debug.print("  exp(-0.001 * 5000) (f64) = {d:.6}\n", .{val5000});
}
