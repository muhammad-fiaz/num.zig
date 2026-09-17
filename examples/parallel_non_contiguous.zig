//! Demonstrates multi-threaded parallel execution across non-contiguous strided array views.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create 1000x20 matrix
    var a = try num.empty(allocator, .{ .shape = &.{ 1000, 20 }, .dtype = .f64 });
    defer a.deinit();

    for (0..1000) |r| {
        for (0..20) |c| {
            try a.set(f64, &.{ r, c }, @as(f64, @floatFromInt(r * 20 + c)));
        }
    }

    // Transpose produces a non-contiguous strided view of shape (20, 1000)
    var view = try num.manip.transpose(a, .{});
    defer view.deinit();

    std.debug.print("View is contiguous: {}\n", .{view.isContiguous()});

    const total = view.elementCount();
    const Context = struct {
        arr: num.Array,
    };
    const ctx = Context{ .arr = view };

    num.parallel.run(total, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |idx| {
                const r = idx / 1000;
                const col = idx % 1000;
                const v = c.arr.get(f64, &.{ r, col }) catch unreachable;
                c.arr.set(f64, &.{ r, col }, v * 2.0) catch unreachable;
            }
        }
    }.kernel, .{ .workers = 4, .threshold = 1000 });

    std.debug.print("Parallel operation on non-contiguous view completed successfully.\n", .{});
    std.debug.print("  view[5, 10] (orig 10, 5 = 205 * 2) = {d:.1}\n", .{try view.get(f64, &.{ 5, 10 })});
}
