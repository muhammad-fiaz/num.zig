const std = @import("std");
const num = @import("num");
const NDArray = num.NDArray;
const Complex = std.math.Complex;

const BenchmarkResult = struct {
    name: []const u8,
    time_ms: f64,
    details: []const u8,
};

const BenchmarkSuite = struct {
    results: std.ArrayListUnmanaged(BenchmarkResult),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuite {
        return .{
            .results = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BenchmarkSuite) void {
        self.results.deinit(self.allocator);
    }

    pub fn add(self: *BenchmarkSuite, name: []const u8, time_ms: f64, details: []const u8) !void {
        try self.results.append(self.allocator, .{
            .name = name,
            .time_ms = time_ms,
            .details = details,
        });
    }

    pub fn printSummary(self: *BenchmarkSuite) void {
        std.debug.print("\n=====================================================================================\n", .{});
        std.debug.print("                             BENCHMARK RESULTS SUMMARY                               \n", .{});
        std.debug.print("=====================================================================================\n", .{});
        std.debug.print("{s:<40} | {s:<15} | {s}\n", .{ "Benchmark", "Time (ms)", "Details" });
        std.debug.print("-----------------------------------------|-----------------|-------------------------\n", .{});

        for (self.results.items) |res| {
            std.debug.print("{s:<40} | {d:>10.4} ms   | {s}\n", .{ res.name, res.time_ms, res.details });
        }
        std.debug.print("=====================================================================================\n", .{});
    }
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var suite = BenchmarkSuite.init(allocator);
    defer suite.deinit();

    std.debug.print("Starting Num.Zig Comprehensive Benchmarks...\n", .{});

    try benchCore(allocator, &suite);
    try benchElementwise(allocator, &suite);
    try benchLinalg(allocator, &suite);
    try benchStats(allocator, &suite);
    try benchML(allocator, &suite);
    try benchFFT(allocator, &suite);

    suite.printSummary();
}

fn benchCore(allocator: std.mem.Allocator, suite: *BenchmarkSuite) !void {
    const size = 1_000_000;
    var timer = try std.time.Timer.start();

    // 1. Allocation & Initialization (zeros)
    timer.reset();
    var a = try NDArray(f32).zeros(allocator, &.{size});
    defer a.deinit();
    const t_zeros = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Core: zeros", t_zeros, "1M elements");

    // 2. ones
    timer.reset();
    var b = try NDArray(f32).ones(allocator, &.{size});
    defer b.deinit();
    const t_ones = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Core: ones", t_ones, "1M elements");

    // 3. full
    timer.reset();
    var c = try NDArray(f32).full(allocator, &.{size}, 3.14);
    defer c.deinit();
    const t_full = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Core: full", t_full, "1M elements");

    // 4. arange
    timer.reset();
    var d = try NDArray(f32).arange(allocator, 0, @as(f32, @floatFromInt(size)), 1);
    defer d.deinit();
    const t_arange = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Core: arange", t_arange, "1M elements");

    // 5. reshape
    timer.reset();
    var reshaped = try d.reshape(&.{ 1000, 1000 });
    defer reshaped.deinit();
    const t_reshape = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Core: reshape", t_reshape, "1M elements (1D -> 2D)");
}

fn benchElementwise(allocator: std.mem.Allocator, suite: *BenchmarkSuite) !void {
    const size = 1_000_000;
    var a = try NDArray(f32).full(allocator, &.{size}, 1.5);
    defer a.deinit();
    var b = try NDArray(f32).full(allocator, &.{size}, 2.5);
    defer b.deinit();

    var timer = try std.time.Timer.start();

    // Add
    timer.reset();
    var c = try num.elementwise.add(allocator, f32, a, b);
    defer c.deinit();
    const t_add = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Elementwise: add", t_add, "1M elements");

    // Mul
    timer.reset();
    var d = try num.elementwise.mul(allocator, f32, a, b);
    defer d.deinit();
    const t_mul = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Elementwise: mul", t_mul, "1M elements");

    // Sub
    timer.reset();
    var e = try num.elementwise.sub(allocator, f32, a, b);
    defer e.deinit();
    const t_sub = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Elementwise: sub", t_sub, "1M elements");

    // Div
    timer.reset();
    var f = try num.elementwise.div(allocator, f32, a, b);
    defer f.deinit();
    const t_div = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Elementwise: div", t_div, "1M elements");
}

fn benchLinalg(allocator: std.mem.Allocator, suite: *BenchmarkSuite) !void {
    // Small/Medium Matrix Multiplication: 256x256
    {
        const n = 256;
        var a = try NDArray(f32).full(allocator, &.{ n, n }, 1.0);
        defer a.deinit();
        var b = try NDArray(f32).full(allocator, &.{ n, n }, 2.0);
        defer b.deinit();

        var timer = try std.time.Timer.start();
        var c = try num.linalg.matmul(f32, allocator, &a, &b);
        defer c.deinit();
        const t_matmul = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
        try suite.add("Linalg: matmul (256x256)", t_matmul, "256x256 f32");
    }

    // Larger Matrix Multiplication: 512x512
    {
        const n = 512;
        var a = try NDArray(f32).full(allocator, &.{ n, n }, 1.0);
        defer a.deinit();
        var b = try NDArray(f32).full(allocator, &.{ n, n }, 2.0);
        defer b.deinit();

        var timer = try std.time.Timer.start();
        var c = try num.linalg.matmul(f32, allocator, &a, &b);
        defer c.deinit();
        const t_matmul = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
        try suite.add("Linalg: matmul (512x512)", t_matmul, "512x512 f32");
    }

    // Dot Product (Large Vector)
    {
        const size = 1_000_000;
        var a = try NDArray(f32).full(allocator, &.{size}, 1.0);
        defer a.deinit();
        var b = try NDArray(f32).full(allocator, &.{size}, 2.0);
        defer b.deinit();

        var timer = try std.time.Timer.start();
        const res = try num.linalg.dot(f32, allocator, &a, &b);
        _ = res;
        const t_dot = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
        try suite.add("Linalg: dot", t_dot, "1M elements");
    }
}

fn benchStats(allocator: std.mem.Allocator, suite: *BenchmarkSuite) !void {
    const size = 1_000_000;
    var a = try NDArray(f32).arange(allocator, 0, @as(f32, @floatFromInt(size)), 1);
    defer a.deinit();

    var timer = try std.time.Timer.start();

    // Sum
    timer.reset();
    _ = try num.stats.sum(f32, &a);
    const t_sum = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Stats: sum", t_sum, "1M elements");

    // Mean
    timer.reset();
    _ = try num.stats.mean(f32, &a);
    const t_mean = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Stats: mean", t_mean, "1M elements");

    // Std Dev
    timer.reset();
    _ = try num.stats.std_val(f32, &a);
    const t_std = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("Stats: std_val", t_std, "1M elements");
}

fn benchML(allocator: std.mem.Allocator, suite: *BenchmarkSuite) !void {
    // Dense Layer Forward Pass simulation: batch=32, input=512 -> output=256
    const batch_size = 32;
    const input_features = 512;
    const output_neurons = 256;

    var input = try NDArray(f32).full(allocator, &.{ batch_size, input_features }, 1.0);
    defer input.deinit();
    var weights = try NDArray(f32).full(allocator, &.{ input_features, output_neurons }, 0.5);
    defer weights.deinit();

    var timer = try std.time.Timer.start();

    // Matmul
    var z = try num.linalg.matmul(f32, allocator, &input, &weights);
    defer z.deinit();

    // Add Bias (using a full bias array and add)
    var bias = try NDArray(f32).full(allocator, &.{output_neurons}, 0.1);
    defer bias.deinit();
    var output = try num.ops.add(f32, allocator, &z, &bias);
    defer output.deinit();

    // ReLU (max with 0)
    var zeros = try NDArray(f32).zeros(allocator, output.shape);
    defer zeros.deinit();
    var activated = try num.elementwise.maximum(allocator, f32, output, zeros);
    defer activated.deinit();

    const t_ml = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("ML: Dense Layer Forward", t_ml, "32x512 -> 256 (Matmul+Add+ReLU)");
}

fn benchFFT(allocator: std.mem.Allocator, suite: *BenchmarkSuite) !void {
    const fft_size = 32768; // Power of 2
    var signal = try NDArray(f32).init(allocator, &.{fft_size});
    defer signal.deinit();
    for (signal.data) |*val| val.* = 1.0;

    var timer = try std.time.Timer.start();
    var fft_res = try num.fft.FFT.fft(allocator, &signal);
    defer fft_res.deinit();
    const fft_time = @as(f64, @floatFromInt(timer.read())) / 1_000_000.0;
    try suite.add("FFT: 1D FFT", fft_time, "32768 points");
}
