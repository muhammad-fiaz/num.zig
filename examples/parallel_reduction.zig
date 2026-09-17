//! Demonstrates parallel tree reduction combining per-thread partial accumulations.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const N: usize = 200_000;
    var a = try num.ones(allocator, .{ .shape = &.{N}, .dtype = .f64 });
    defer a.deinit();

    const num_workers: usize = 4;
    var partial_sums = [_]f64{0.0} ** num_workers;
    const chunk_size = (N + num_workers - 1) / num_workers;

    const Context = struct {
        arr: num.Array,
        partials: []f64,
        chunk_size: usize,
    };
    const ctx = Context{
        .arr = a,
        .partials = &partial_sums,
        .chunk_size = chunk_size,
    };

    num.parallel.run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            const worker_id = start / c.chunk_size;
            var acc: f64 = 0.0;
            for (start..end) |i| {
                acc += c.arr.get(f64, &.{i}) catch 0.0;
            }
            c.partials[worker_id] = acc;
        }
    }.kernel, .{
        .workers = num_workers,
        .chunkSize = chunk_size,
        .threshold = 1000,
    });

    var total_sum: f64 = 0.0;
    for (partial_sums, 0..) |partial, idx| {
        std.debug.print("  Worker {d} partial sum: {d:.1}\n", .{ idx, partial });
        total_sum += partial;
    }

    std.debug.print("Total parallel reduction sum: {d:.1} (expected {d:.1})\n", .{ total_sum, @as(f64, @floatFromInt(N)) });
}
