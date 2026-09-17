//! Demonstrates high-throughput parallel processing of large arrays in num.zig.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const N: usize = 1_000_000;
    var a = try num.empty(allocator, .{ .shape = &.{N}, .dtype = .f64 });
    defer a.deinit();

    const Context = struct {
        arr: num.Array,
    };
    const ctx = Context{ .arr = a };

    var io_threaded: std.Io.Threaded = .init(allocator, .{});
    defer io_threaded.deinit();
    const io = io_threaded.io();
    const clock: std.Io.Clock = .boot;

    const t0 = clock.now(io).nanoseconds;

    // Scale automatically across all available CPU threads
    num.parallel.run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| {
                const val = @as(f64, @floatFromInt(i % 1000));
                c.arr.set(f64, &.{i}, std.math.sin(val) * std.math.cos(val)) catch unreachable;
            }
        }
    }.kernel, .{});

    const t1 = clock.now(io).nanoseconds;
    const elapsed_ns: u64 = @intCast(@max(1, t1 - t0));
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

    std.debug.print("Processed {d} elements in {d:.2} ms.\n", .{ N, elapsed_ms });
    std.debug.print("  result[100] = {d:.4}\n", .{try a.get(f64, &.{100})});
}
