//! Performance benchmark suite for num.zig.
//!
//! Benchmarks matrix multiplication, elementwise arithmetic, reductions,
//! sorting, and NZIG v1.0 binary serialization throughput.

const std = @import("std");
const num = @import("num");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    var io_threaded: std.Io.Threaded = .init(allocator, .{});
    defer io_threaded.deinit();
    const io = io_threaded.io();
    const clock: std.Io.Clock = .boot;

    std.debug.print("====================================================\n", .{});
    std.debug.print(" num.zig Performance Benchmark Suite (ReleaseFast) \n", .{});
    std.debug.print("====================================================\n\n", .{});

    // 1. Matrix Multiplication Benchmark: 200x200 f64
    {
        const N: usize = 200;
        var a = try num.full(allocator, .{ .shape = &.{ N, N }, .value = @as(f64, 1.01), .dtype = .f64 });
        defer a.deinit();
        var b = try num.full(allocator, .{ .shape = &.{ N, N }, .value = @as(f64, 0.99), .dtype = .f64 });
        defer b.deinit();

        const t0 = clock.now(io).nanoseconds;
        var c = try num.linalg.matmul(a, b, .{});
        defer c.deinit();
        const t1 = clock.now(io).nanoseconds;

        const elapsed_ns: u64 = @intCast(@max(1, t1 - t0));
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        const gflops = (2.0 * @as(f64, N) * @as(f64, N) * @as(f64, N)) / (@as(f64, @floatFromInt(elapsed_ns)));
        std.debug.print("1. Matmul ({d}x{d} f64):\n   Time: {d:.2} ms | {d:.2} GFLOPS\n\n", .{ N, N, elapsed_ms, gflops });
    }

    // 2. Elementwise Addition Benchmark: 1,000,000 f64
    {
        const N: usize = 1_000_000;
        var a = try num.full(allocator, .{ .shape = &.{N}, .value = @as(f64, 2.5), .dtype = .f64 });
        defer a.deinit();
        var b = try num.full(allocator, .{ .shape = &.{N}, .value = @as(f64, 1.5), .dtype = .f64 });
        defer b.deinit();

        const t0 = clock.now(io).nanoseconds;
        var c = try num.ops.add(a, b, .{});
        defer c.deinit();
        const t1 = clock.now(io).nanoseconds;

        const elapsed_ns: u64 = @intCast(@max(1, t1 - t0));
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
        const throughput_mbe = (@as(f64, N) / 1_000_000.0) / (@as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0);

        std.debug.print("2. Elementwise Add (1M f64):\n   Time: {d:.2} ms | {d:.2} M elements/sec\n\n", .{ elapsed_ms, throughput_mbe });
    }

    // 3. Reduction Benchmark: sum over 1,000,000 f64
    {
        const N: usize = 1_000_000;
        var a = try num.full(allocator, .{ .shape = &.{N}, .value = @as(f64, 1.0), .dtype = .f64 });
        defer a.deinit();

        const t0 = clock.now(io).nanoseconds;
        var s = try num.reduce.sum(a, .{});
        defer s.deinit();
        const t1 = clock.now(io).nanoseconds;

        const elapsed_ns: u64 = @intCast(@max(1, t1 - t0));
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("3. Reduction Sum (1M f64):\n   Time: {d:.2} ms | Sum: {d:.0}\n\n", .{ elapsed_ms, try s.get(f64, &.{}) });
    }

    // 4. In-Place Sort: 100,000 f64
    {
        const N: usize = 100_000;
        var rng = num.random.Prng.init(42);
        var arr = try num.random.uniform(allocator, .{ .low = 0.0, .high = 1000.0, .shape = &.{N}, .rng = &rng });
        defer arr.deinit();

        const t0 = clock.now(io).nanoseconds;
        try num.sort.sort(&arr, .{ .order = .asc });
        const t1 = clock.now(io).nanoseconds;

        const elapsed_ns: u64 = @intCast(@max(1, t1 - t0));
        const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;

        std.debug.print("4. QuickSort (100k f64):\n   Time: {d:.2} ms\n\n", .{elapsed_ms});
    }

    // 5. NZIG v1.0 Serialization & Deserialization: 500,000 f64 (4 MB payload)
    {
        const N: usize = 500_000;
        var arr = try num.full(allocator, .{ .shape = &.{N}, .value = @as(f64, 3.14159), .dtype = .f64 });
        defer arr.deinit();

        const payload_bytes = N * @sizeOf(f64);
        const buf = try allocator.alloc(u8, payload_bytes + num.io.nzig.HEADER_SIZE);
        defer allocator.free(buf);

        var ms = num.io.MemoryStream.init(buf);

        const t0_w = clock.now(io).nanoseconds;
        try num.io.writeToStream(arr, &ms);
        const t1_w = clock.now(io).nanoseconds;
        const write_ns: u64 = @intCast(@max(1, t1_w - t0_w));

        var rs = num.io.MemoryStream.init(ms.getWritten());
        rs.written = ms.written;

        const t0_r = clock.now(io).nanoseconds;
        var loaded = try num.io.readFromStream(allocator, &rs);
        defer loaded.deinit();
        const t1_r = clock.now(io).nanoseconds;
        const read_ns: u64 = @intCast(@max(1, t1_r - t0_r));

        const mb = @as(f64, @floatFromInt(payload_bytes)) / (1024.0 * 1024.0);
        const write_mb_s = mb / (@as(f64, @floatFromInt(write_ns)) / 1_000_000_000.0);
        const read_mb_s = mb / (@as(f64, @floatFromInt(read_ns)) / 1_000_000_000.0);

        std.debug.print("5. NZIG v1.0 Serialization ({d:.1} MB payload):\n", .{mb});
        std.debug.print("   Write: {d:.2} GB/s ({d:.2} ms)\n", .{ write_mb_s / 1024.0, @as(f64, @floatFromInt(write_ns)) / 1_000_000.0 });
        std.debug.print("   Read:  {d:.2} GB/s ({d:.2} ms)\n\n", .{ read_mb_s / 1024.0, @as(f64, @floatFromInt(read_ns)) / 1_000_000.0 });
    }

    std.debug.print("====================================================\n", .{});
    std.debug.print(" All benchmarks completed successfully.            \n", .{});
    std.debug.print("====================================================\n", .{});
}
