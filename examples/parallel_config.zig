//! Demonstrates explicit worker and chunk configuration in num.zig parallel execution.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const N: usize = 64_000;
    var a = try num.empty(allocator, .{ .shape = &.{N}, .dtype = .f64 });
    defer a.deinit();

    const Context = struct {
        arr: num.Array,
    };
    const ctx = Context{ .arr = a };

    // Explicitly configure 4 workers with 4096 elements per chunk
    num.parallel.run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| {
                const x = @as(f64, @floatFromInt(i));
                c.arr.set(f64, &.{i}, std.math.sqrt(x)) catch unreachable;
            }
        }
    }.kernel, .{
        .workers = 4,
        .chunkSize = 4096,
        .threshold = 1000,
    });

    std.debug.print("Parallel configured execution completed (4 workers, 4096 chunk size):\n", .{});
    std.debug.print("  sqrt(0) = {d:.2}\n", .{try a.get(f64, &.{0})});
    std.debug.print("  sqrt(16) = {d:.2}\n", .{try a.get(f64, &.{16})});
    std.debug.print("  sqrt(10000) = {d:.2}\n", .{try a.get(f64, &.{10000})});
}
