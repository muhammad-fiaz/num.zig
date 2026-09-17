//! High-performance CPU parallel execution subsystem for num.zig.
//!
//! Provides deterministic data-parallel partitioning using native Zig 0.16.0 `std.Thread`
//! with automatic sequential fallback for small workloads, explicit inline caller control,
//! nested parallelism protection, and safe resource cleanup.

const std = @import("std");
const builtin = @import("builtin");
const DType = @import("../core/dtype.zig").DType;
const Array = @import("../core/array.zig").Array;
const empty = @import("../core/array.zig").empty;

/// Default minimum elements required to trigger multithreaded spawning.
pub const DEFAULT_THRESHOLD: usize = 16384;

/// Maximum number of concurrently spawned worker threads per operation.
pub const MAX_WORKERS: usize = 64;

/// Thread-local guard to prevent nested thread explosion (workers spawning workers).
threadlocal var in_parallel_worker: bool = false;

/// Executes a kernel function across range `[0, count)` partitioned across available CPU workers.
///
/// Supports automatic scheduling:
/// ```zig
/// num.parallel.run(count, context, kernel, .{});
/// ```
///
/// Supports explicit inline configuration:
/// ```zig
/// num.parallel.run(count, context, kernel, .{
///     .workers = 8,
///     .chunkSize = 4096,
///     .threshold = 1000,
/// });
/// ```
pub fn run(
    count: usize,
    context: anytype,
    comptime kernel: fn (@TypeOf(context), usize, usize) void,
    args: anytype,
) void {
    if (count == 0) return;

    // Single-threaded targets or already running inside a parallel worker thread -> sequential fallback
    if (builtin.single_threaded or in_parallel_worker) {
        kernel(context, 0, count);
        return;
    }

    // Inspect optional inline configuration
    const OptsType = @TypeOf(args);
    const has_opts = @typeInfo(OptsType) == .@"struct";

    const explicit_workers: ?usize = if (has_opts and @hasField(OptsType, "workers"))
        args.workers
    else
        null;

    const explicit_chunk_size: ?usize = if (has_opts and @hasField(OptsType, "chunkSize"))
        args.chunkSize
    else
        null;

    const threshold: usize = if (has_opts and @hasField(OptsType, "threshold"))
        args.threshold
    else
        DEFAULT_THRESHOLD;

    // If caller explicitly requested 1 worker or count is below threshold (and workers not explicitly forced > 1), run sequentially
    if (explicit_workers) |w| {
        if (w <= 1) {
            kernel(context, 0, count);
            return;
        }
    } else if (count < threshold) {
        kernel(context, 0, count);
        return;
    }

    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (cpu_count <= 1 and (explicit_workers == null or explicit_workers.? <= 1)) {
        kernel(context, 0, count);
        return;
    }

    // Determine target worker count
    var num_workers: usize = if (explicit_workers) |w|
        @min(w, MAX_WORKERS)
    else
        @min(cpu_count, MAX_WORKERS);

    // Ensure worker count does not exceed element count
    num_workers = @max(1, @min(num_workers, count));

    // Determine chunk size
    const chunk_size: usize = if (explicit_chunk_size) |cs|
        @max(1, cs)
    else
        @max(1, count / num_workers);

    // If chunk size is specified, ensure we don't spawn more workers than needed
    if (explicit_chunk_size != null) {
        const needed_workers = (count + chunk_size - 1) / chunk_size;
        num_workers = @max(1, @min(num_workers, needed_workers));
    }

    if (num_workers <= 1) {
        kernel(context, 0, count);
        return;
    }

    const WorkerContext = struct {
        ctx: @TypeOf(context),
        start: usize,
        end: usize,

        fn execute(self: @This()) void {
            in_parallel_worker = true;
            defer in_parallel_worker = false;
            kernel(self.ctx, self.start, self.end);
        }
    };

    var threads: [MAX_WORKERS]std.Thread = undefined;
    var spawned: usize = 0;
    const to_spawn = num_workers - 1;

    for (0..to_spawn) |w| {
        const start = w * chunk_size;
        const end = @min(start + chunk_size, count);
        if (start >= count) break;

        const wc = WorkerContext{
            .ctx = context,
            .start = start,
            .end = end,
        };

        threads[spawned] = std.Thread.spawn(.{}, WorkerContext.execute, .{wc}) catch {
            // If thread spawning fails, run the remaining range sequentially on the main thread
            break;
        };
        spawned += 1;
    }

    // Main thread computes the remaining workload
    const main_start = spawned * chunk_size;
    if (main_start < count) {
        kernel(context, main_start, count);
    }

    // Safely join all spawned worker threads
    for (0..spawned) |w| {
        threads[w].join();
    }
}

test "parallel.run automatic and sequential fallback" {
    // 0 elements
    const DummyCtx = struct {};
    run(0, DummyCtx{}, struct {
        fn kernel(_: DummyCtx, _: usize, _: usize) void {}
    }.kernel, .{});

    // 1 element
    var one_elem = [_]i32{0};
    const OneCtx = struct { slice: []i32 };
    run(1, OneCtx{ .slice = &one_elem }, struct {
        fn kernel(c: OneCtx, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] = 42;
        }
    }.kernel, .{});
    try std.testing.expectEqual(@as(i32, 42), one_elem[0]);

    // Explicit workers = 1
    var data10 = [_]u32{0} ** 10;
    const Ctx10 = struct { slice: []u32 };
    run(10, Ctx10{ .slice = &data10 }, struct {
        fn kernel(c: Ctx10, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] = @intCast(i * 3);
        }
    }.kernel, .{ .workers = 1 });
    for (0..10) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i * 3)), data10[i]);
    }
}

test "parallel.run worker count scaling and chunk sizes" {
    const allocator = std.testing.allocator;
    const N: usize = 20000;
    const data = try allocator.alloc(f64, N);
    defer allocator.free(data);

    const Context = struct { slice: []f64 };
    const ctx = Context{ .slice = data };

    // Test workers = 2 with small chunk
    run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] = @as(f64, @floatFromInt(i)) * 1.5;
        }
    }.kernel, .{ .workers = 2, .chunkSize = 256, .threshold = 100 });

    for (0..N) |i| {
        try std.testing.expectEqual(@as(f64, @floatFromInt(i)) * 1.5, data[i]);
    }

    // Test workers = 4 with chunk larger than workload
    run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] += 10.0;
        }
    }.kernel, .{ .workers = 4, .chunkSize = 50000, .threshold = 100 });

    for (0..N) |i| {
        try std.testing.expectEqual(@as(f64, @floatFromInt(i)) * 1.5 + 10.0, data[i]);
    }

    // Test workers > CPU count (e.g. 32)
    run(N, ctx, struct {
        fn kernel(c: Context, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] = @as(f64, @floatFromInt(i));
        }
    }.kernel, .{ .workers = 32, .threshold = 100 });

    for (0..N) |i| {
        try std.testing.expectEqual(@as(f64, @floatFromInt(i)), data[i]);
    }
}

test "parallel.run dtypes coverage (f32, i64, bool, Complex)" {
    const allocator = std.testing.allocator;
    const N: usize = 5000;

    // f32
    const f32_buf = try allocator.alloc(f32, N);
    defer allocator.free(f32_buf);
    const F32Ctx = struct { slice: []f32 };
    run(N, F32Ctx{ .slice = f32_buf }, struct {
        fn kernel(c: F32Ctx, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] = @as(f32, @floatFromInt(i)) * 0.5;
        }
    }.kernel, .{ .workers = 4, .threshold = 100 });
    try std.testing.expectEqual(@as(f32, 2.5), f32_buf[5]);

    // i64
    const i64_buf = try allocator.alloc(i64, N);
    defer allocator.free(i64_buf);
    const I64Ctx = struct { slice: []i64 };
    run(N, I64Ctx{ .slice = i64_buf }, struct {
        fn kernel(c: I64Ctx, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] = @as(i64, @intCast(i)) * 100;
        }
    }.kernel, .{ .workers = 4, .threshold = 100 });
    try std.testing.expectEqual(@as(i64, 500), i64_buf[5]);

    // bool
    const bool_buf = try allocator.alloc(bool, N);
    defer allocator.free(bool_buf);
    const BoolCtx = struct { slice: []bool };
    run(N, BoolCtx{ .slice = bool_buf }, struct {
        fn kernel(c: BoolCtx, start: usize, end: usize) void {
            for (start..end) |i| c.slice[i] = (i % 2 == 0);
        }
    }.kernel, .{ .workers = 4, .threshold = 100 });
    try std.testing.expectEqual(true, bool_buf[4]);
    try std.testing.expectEqual(false, bool_buf[5]);

    // Complex(f64)
    const Complex64 = std.math.Complex(f64);
    const c128_buf = try allocator.alloc(Complex64, N);
    defer allocator.free(c128_buf);
    const C128Ctx = struct { slice: []Complex64 };
    run(N, C128Ctx{ .slice = c128_buf }, struct {
        fn kernel(c: C128Ctx, start: usize, end: usize) void {
            for (start..end) |i| {
                c.slice[i] = Complex64.init(@as(f64, @floatFromInt(i)), -@as(f64, @floatFromInt(i)));
            }
        }
    }.kernel, .{ .workers = 4, .threshold = 100 });
    try std.testing.expectEqual(@as(f64, 10.0), c128_buf[10].re);
    try std.testing.expectEqual(@as(f64, -10.0), c128_buf[10].im);
}

test "parallel.run tree reduction pattern" {
    const allocator = std.testing.allocator;
    const N: usize = 100000;
    const data = try allocator.alloc(f64, N);
    defer allocator.free(data);
    for (0..N) |i| data[i] = 1.0;

    const num_workers: usize = 4;
    var partial_sums = [_]f64{0.0} ** num_workers;

    const ReduceCtx = struct {
        input: []const f64,
        partials: []f64,
        chunk_size: usize,
    };

    const chunk_size = (N + num_workers - 1) / num_workers;
    const ctx = ReduceCtx{
        .input = data,
        .partials = &partial_sums,
        .chunk_size = chunk_size,
    };

    run(N, ctx, struct {
        fn kernel(c: ReduceCtx, start: usize, end: usize) void {
            const worker_idx = start / c.chunk_size;
            var acc: f64 = 0.0;
            for (start..end) |i| {
                acc += c.input[i];
            }
            c.partials[worker_idx] = acc;
        }
    }.kernel, .{ .workers = num_workers, .chunkSize = chunk_size, .threshold = 1000 });

    var total_sum: f64 = 0.0;
    for (partial_sums) |s| {
        total_sum += s;
    }
    try std.testing.expectEqual(@as(f64, 100000.0), total_sum);
}

test "parallel.run non-contiguous strided array views" {
    const allocator = std.testing.allocator;

    // Create 100x10 array and transpose to (10, 100) strided view
    var arr = try empty(allocator, .{ .shape = &.{ 100, 10 }, .dtype = .f64 });
    defer arr.deinit();

    for (0..100) |r| {
        for (0..10) |c| {
            try arr.set(f64, &.{ r, c }, @as(f64, @floatFromInt(r * 10 + c)));
        }
    }

    var transposed = try @import("../manip/transpose.zig").transpose(arr, .{});
    defer transposed.deinit();
    try std.testing.expect(!transposed.isContiguous());

    const total = transposed.elementCount();
    const ViewCtx = struct {
        view: Array,
    };

    run(total, ViewCtx{ .view = transposed }, struct {
        fn kernel(c: ViewCtx, start: usize, end: usize) void {
            for (start..end) |idx| {
                const r = idx / 100;
                const col = idx % 100;
                const val = c.view.get(f64, &.{ r, col }) catch unreachable;
                c.view.set(f64, &.{ r, col }, val + 1.0) catch unreachable;
            }
        }
    }.kernel, .{ .workers = 4, .threshold = 100 });

    // Verify row 2, col 5 of transposed (original row 5, col 2 = 52) was incremented to 53
    try std.testing.expectEqual(@as(f64, 53.0), try transposed.get(f64, &.{ 2, 5 }));
}
