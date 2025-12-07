const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Allocator = std.mem.Allocator;

/// Provides random number generation capabilities for NDArrays.
///
/// Wraps Zig's standard library random number generator.
///
/// Example:
/// ```zig
/// var rng = random.Random.init(42);
/// var arr = try rng.uniform(allocator, &.{2, 3});
/// defer arr.deinit();
/// ```
pub const Random = struct {
    prng: std.Random.DefaultPrng,

    /// Initializes a new Random instance with a seed.
    ///
    /// Arguments:
    ///     seed: The seed for the pseudo-random number generator.
    ///
    /// Returns:
    ///     A new Random instance.
    ///
    /// Example:
    /// ```zig
    /// var rng = random.Random.init(12345);
    /// ```
    pub fn init(seed: u64) Random {
        return Random{
            .prng = std.Random.DefaultPrng.init(seed),
        };
    }

    /// Generates an array with values drawn from a uniform distribution [0, 1).
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     allocator: The allocator to use for the result array.
    ///     shape: The shape of the output array.
    ///
    /// Returns:
    ///     A new NDArray(f32) containing the random values.
    ///
    /// Example:
    /// ```zig
    /// var rng = random.Random.init(42);
    /// var arr = try rng.uniform(allocator, &.{2, 2});
    /// defer arr.deinit();
    /// // arr contains random floats between 0.0 and 1.0
    /// ```
    pub fn uniform(self: *Random, allocator: Allocator, shape: []const usize) !NDArray(f32) {
        const arr = try NDArray(f32).init(allocator, shape);
        const rand = self.prng.random();
        for (arr.data) |*val| {
            val.* = rand.float(f32);
        }
        return arr;
    }

    /// Generates an array with values drawn from a normal distribution.
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     allocator: The allocator to use for the result array.
    ///     shape: The shape of the output array.
    ///     mean: The mean of the distribution.
    ///     stddev: The standard deviation of the distribution.
    ///
    /// Returns:
    ///     A new NDArray(f32) containing the random values.
    ///
    /// Example:
    /// ```zig
    /// var rng = random.Random.init(42);
    /// var arr = try rng.normal(allocator, &.{2, 2}, 0.0, 1.0);
    /// defer arr.deinit();
    /// // arr contains random floats from N(0, 1)
    /// ```
    pub fn normal(self: *Random, allocator: Allocator, shape: []const usize, mean: f32, stddev: f32) !NDArray(f32) {
        const arr = try NDArray(f32).init(allocator, shape);
        const rand = self.prng.random();
        for (arr.data) |*val| {
            val.* = rand.floatNorm(f32) * stddev + mean;
        }
        return arr;
    }

    /// Generates an array with values drawn from a uniform integer distribution [low, high).
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     allocator: The allocator to use for the result array.
    ///     shape: The shape of the output array.
    ///     low: The lower bound (inclusive).
    ///     high: The upper bound (exclusive).
    ///
    /// Returns:
    ///     A new NDArray(i32) containing the random integers.
    ///
    /// Example:
    /// ```zig
    /// var rng = random.Random.init(42);
    /// var arr = try rng.randint(allocator, &.{5}, 0, 10);
    /// defer arr.deinit();
    /// // arr contains random integers between 0 and 9
    /// ```
    pub fn randint(self: *Random, allocator: Allocator, shape: []const usize, low: i32, high: i32) !NDArray(i32) {
        const arr = try NDArray(i32).init(allocator, shape);
        const rand = self.prng.random();
        for (arr.data) |*val| {
            val.* = rand.intRangeAtMost(i32, low, high - 1);
        }
        return arr;
    }

    /// Shuffles the array along the first axis in-place.
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     T: The data type of the array elements.
    ///     arr: The NDArray to shuffle.
    ///
    /// Example:
    /// ```zig
    /// var rng = random.Random.init(42);
    /// var arr = try NDArray(f32).init(allocator, &.{3}, &.{1.0, 2.0, 3.0});
    /// defer arr.deinit();
    ///
    /// rng.shuffle(f32, &arr);
    /// // arr is shuffled, e.g., {2.0, 1.0, 3.0}
    /// ```
    pub fn shuffle(self: *Random, comptime T: type, arr: *NDArray(T)) void {
        const n = arr.shape[0];
        const rand = self.prng.random();

        var i: usize = 0;
        while (i < n - 1) : (i += 1) {
            const j = rand.intRangeAtMost(usize, i, n - 1);
            if (i != j) {
                // Swap row i and j
                if (arr.rank() == 1) {
                    const temp = arr.data[i];
                    arr.data[i] = arr.data[j];
                    arr.data[j] = temp;
                } else {
                    // Assuming contiguous row-major for simplicity
                    const row_stride = arr.strides[0];
                    var k: usize = 0;
                    while (k < row_stride) : (k += 1) {
                        const idx_i = i * row_stride + k;
                        const idx_j = j * row_stride + k;

                        const temp = arr.data[idx_i];
                        arr.data[idx_i] = arr.data[idx_j];
                        arr.data[idx_j] = temp;
                    }
                }
            }
        }
    }

    /// Randomly permute a sequence, or return a permuted range.
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     allocator: The allocator to use for the result array.
    ///     n: The size of the range to permute.
    ///
    /// Returns:
    ///     A new NDArray(usize) containing the permuted range [0, n).
    ///
    /// Example:
    /// ```zig
    /// var rng = random.Random.init(42);
    /// var p = try rng.permutation(allocator, 5);
    /// defer p.deinit();
    /// // p is a permutation of 0..4
    /// ```
    pub fn permutation(self: *Random, allocator: Allocator, n: usize) !NDArray(usize) {
        var arr = try NDArray(usize).arange(allocator, 0, n, 1);
        self.shuffle(usize, &arr);
        return arr;
    }

    /// Generates a random sample from a given 1-D array.
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     allocator: The allocator to use for the result array.
    ///     T: The data type of the array elements.
    ///     a: The input 1-D NDArray.
    ///     size: The number of samples to draw.
    ///     replace: Whether the sample is with or without replacement.
    ///
    /// Returns:
    ///     A new NDArray(T) containing the sampled values.
    ///
    /// Example:
    /// ```zig
    /// var rng = random.Random.init(42);
    /// var a = try NDArray(f32).init(allocator, &.{3}, &.{10.0, 20.0, 30.0});
    /// defer a.deinit();
    ///
    /// var sample = try rng.choice(allocator, f32, a, 2, false);
    /// defer sample.deinit();
    /// // sample contains 2 elements from a
    /// ```
    pub fn choice(self: *Random, allocator: Allocator, comptime T: type, a: NDArray(T), size: usize, replace: bool) !NDArray(T) {
        if (a.rank() != 1) return core.Error.RankMismatch;

        const rand = self.prng.random();
        var result = try NDArray(T).init(allocator, &.{size});

        if (replace) {
            for (result.data) |*val| {
                const idx = rand.intRangeAtMost(usize, 0, a.size() - 1);
                val.* = a.data[idx * a.strides[0]];
            }
        } else {
            if (size > a.size()) return core.Error.DimensionMismatch;

            var indices = try allocator.alloc(usize, a.size());
            defer allocator.free(indices);
            for (0..a.size()) |i| indices[i] = i;

            // Partial shuffle (Fisher-Yates)
            var i: usize = 0;
            while (i < size) : (i += 1) {
                const j = rand.intRangeAtMost(usize, i, a.size() - 1);
                const temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;

                result.data[i] = a.data[indices[i] * a.strides[0]];
            }
        }
        return result;
    }

    /// Generates an array with values drawn from an exponential distribution.
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     allocator: The allocator to use for the result array.
    ///     shape: The shape of the output array.
    ///     scale: The scale parameter (1/lambda).
    ///
    /// Returns:
    ///     A new NDArray(f32) containing the random values.
    pub fn exponential(self: *Random, allocator: Allocator, shape: []const usize, scale: f32) !NDArray(f32) {
        const arr = try NDArray(f32).init(allocator, shape);
        const rand = self.prng.random();
        for (arr.data) |*val| {
            // -scale * ln(1 - u)
            // rand.float(f32) returns [0, 1). 1-u is (0, 1].
            const u = rand.float(f32);
            val.* = -scale * std.math.ln(1.0 - u);
        }
        return arr;
    }

    /// Generates an array with values drawn from a Poisson distribution.
    ///
    /// Arguments:
    ///     self: The Random instance.
    ///     allocator: The allocator to use for the result array.
    ///     shape: The shape of the output array.
    ///     lam: The expectation of interval (lambda).
    ///
    /// Returns:
    ///     A new NDArray(usize) containing the random values.
    pub fn poisson(self: *Random, allocator: Allocator, shape: []const usize, lam: f32) !NDArray(usize) {
        const arr = try NDArray(usize).init(allocator, shape);
        const rand = self.prng.random();

        // Knuth's algorithm for small lambda
        // For large lambda, normal approximation or other methods are better.
        // Here we use simple Knuth's algorithm.

        const L = std.math.exp(-lam);

        for (arr.data) |*val| {
            var k: usize = 0;
            var p: f32 = 1.0;
            while (p > L) {
                k += 1;
                p *= rand.float(f32);
            }
            val.* = k - 1;
        }
        return arr;
    }
};

test "random distribution stats" {
    const allocator = std.testing.allocator;
    var rng = Random.init(42);
    var arr = try rng.normal(allocator, &.{1000}, 0.0, 1.0);
    defer arr.deinit();

    // Simple check if values are somewhat distributed
    var sum: f32 = 0;
    for (arr.data) |v| sum += v;
    const mean = sum / 1000.0;
    try std.testing.expect(mean > -0.2 and mean < 0.2);
}
