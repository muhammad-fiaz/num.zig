//! Demonstrates basic multi-threaded CPU parallel execution in num.zig.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const N: usize = 100_000;
    var a = try num.empty(allocator, .{ .shape = &.{N}, .dtype = .f64 });
    defer a.deinit();

    const Context = struct {
        arr: num.Array,
    };
    const ctx = Context{ .arr = a };

    // Automatic worker scheduling: uses CPU core count automatically
    num.parallel.run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| {
                const x = @as(f64, @floatFromInt(i));
                c.arr.set(f64, &.{i}, x * x) catch unreachable;
            }
        }
    }.kernel, .{});

    std.debug.print("Parallel basic execution completed on {d} elements.\n", .{N});
    std.debug.print("  a[0] = {d:.1}\n", .{try a.get(f64, &.{0})});
    std.debug.print("  a[10] = {d:.1}\n", .{try a.get(f64, &.{10})});
    std.debug.print("  a[999] = {d:.1}\n", .{try a.get(f64, &.{999})});
}
