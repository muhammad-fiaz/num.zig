const std = @import("std");
const core = @import("core.zig");
const NDArray = core.NDArray;
const Complex = std.math.Complex;

/// Provides Fast Fourier Transform (FFT) operations.
///
/// This struct contains static methods for computing FFT and Inverse FFT.
pub const FFT = struct {
    /// Computes the 1D Fast Fourier Transform of the input array.
    ///
    /// The input array must be 1-dimensional and its length must be a power of 2.
    /// The Cooley-Tukey algorithm is used.
    ///
    /// Arguments:
    ///     allocator: The allocator to use for the result array.
    ///     input: The input array of real numbers.
    ///
    /// Returns:
    ///     A new array containing the complex FFT coefficients.
    ///
    /// Example:
    /// ```zig
    /// var a = try NDArray(f32).init(allocator, &.{4}, &.{1.0, 1.0, 1.0, 1.0});
    /// defer a.deinit();
    ///
    /// var result = try FFT.fft(allocator, &a);
    /// defer result.deinit();
    /// // result is approx {4+0i, 0+0i, 0+0i, 0+0i}
    /// ```
    pub fn fft(allocator: std.mem.Allocator, input: *const NDArray(f32)) !NDArray(Complex(f32)) {
        if (input.shape.len != 1) return error.NotImplemented;
        const n = input.shape[0];
        if (!std.math.isPowerOfTwo(n)) return error.InvalidShape;

        var output = try NDArray(Complex(f32)).init(allocator, &.{n});

        // Copy input to output
        for (input.data, 0..) |val, i| {
            output.data[i] = Complex(f32).init(val, 0);
        }

        bit_reverse_permutation(output.data);

        // Butterfly operations
        var len: usize = 2;
        while (len <= n) : (len <<= 1) {
            const angle = -2.0 * std.math.pi / @as(f32, @floatFromInt(len));
            const wlen = Complex(f32).init(std.math.cos(angle), std.math.sin(angle));

            var i: usize = 0;
            while (i < n) : (i += len) {
                var w = Complex(f32).init(1.0, 0.0);
                const half_len = len / 2;
                for (0..half_len) |k| {
                    const u = output.data[i + k];
                    const v = output.data[i + k + half_len].mul(w);

                    output.data[i + k] = u.add(v);
                    output.data[i + k + half_len] = u.sub(v);

                    w = w.mul(wlen);
                }
            }
        }

        return output;
    }

    /// Computes the 1D Inverse Fast Fourier Transform of the input array.
    ///
    /// The input array must be 1-dimensional and its length must be a power of 2.
    ///
    /// Arguments:
    ///     allocator: The allocator to use for the result array.
    ///     input: The input array of complex numbers.
    ///
    /// Returns:
    ///     A new array containing the complex IFFT result.
    ///
    /// Example:
    /// ```zig
    /// var a = try NDArray(Complex(f32)).init(allocator, &.{4});
    /// defer a.deinit();
    /// a.data[0] = Complex(f32).init(4.0, 0.0);
    /// // ... set other elements to 0 ...
    ///
    /// var result = try FFT.ifft(allocator, &a);
    /// defer result.deinit();
    /// // result is approx {1+0i, 1+0i, 1+0i, 1+0i}
    /// ```
    pub fn ifft(allocator: std.mem.Allocator, input: *const NDArray(Complex(f32))) !NDArray(Complex(f32)) {
        if (input.shape.len != 1) return error.NotImplemented;
        const n = input.shape[0];
        if (!std.math.isPowerOfTwo(n)) return error.InvalidShape;

        var output = try input.copy(allocator);

        bit_reverse_permutation(output.data);
        // Butterfly operations
        var len: usize = 2;
        while (len <= n) : (len <<= 1) {
            const angle = 2.0 * std.math.pi / @as(f32, @floatFromInt(len)); // Positive angle for IFFT
            const wlen = Complex(f32).init(std.math.cos(angle), std.math.sin(angle));

            var i: usize = 0;
            while (i < n) : (i += len) {
                var w = Complex(f32).init(1.0, 0.0);
                const half_len = len / 2;
                for (0..half_len) |k| {
                    const u = output.data[i + k];
                    const v = output.data[i + k + half_len].mul(w);

                    output.data[i + k] = u.add(v);
                    output.data[i + k + half_len] = u.sub(v);

                    w = w.mul(wlen);
                }
            }
        }

        // Scale by 1/n
        const scale = 1.0 / @as(f32, @floatFromInt(n));
        for (output.data) |*val| {
            val.re *= scale;
            val.im *= scale;
        }

        return output;
    }

    fn bit_reverse_permutation(data: []Complex(f32)) void {
        const n = data.len;
        var j: usize = 0;
        for (0..n) |i| {
            if (i < j) {
                const temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
            var m = n / 2;
            while (m >= 1 and j >= m) {
                j -= m;
                m /= 2;
            }
            j += m;
        }
    }

    /// Computes the N-dimensional FFT.
    pub fn fftn(allocator: std.mem.Allocator, input: *const NDArray(Complex(f32))) !NDArray(Complex(f32)) {
        var current = try input.copy(allocator);

        for (0..current.rank()) |axis| {
            const next = try fftAxis(allocator, &current, axis, false);
            current.deinit();
            current = next;
        }
        return current;
    }

    /// Computes the N-dimensional Inverse FFT.
    pub fn ifftn(allocator: std.mem.Allocator, input: *const NDArray(Complex(f32))) !NDArray(Complex(f32)) {
        var current = try input.copy(allocator);

        for (0..current.rank()) |axis| {
            const next = try fftAxis(allocator, &current, axis, true);
            current.deinit();
            current = next;
        }
        return current;
    }

    /// Computes the 2-dimensional FFT.
    pub fn fft2(allocator: std.mem.Allocator, input: *const NDArray(Complex(f32))) !NDArray(Complex(f32)) {
        if (input.rank() != 2) return core.Error.RankMismatch;
        return fftn(allocator, input);
    }

    /// Computes the 2-dimensional Inverse FFT.
    pub fn ifft2(allocator: std.mem.Allocator, input: *const NDArray(Complex(f32))) !NDArray(Complex(f32)) {
        if (input.rank() != 2) return core.Error.RankMismatch;
        return ifftn(allocator, input);
    }

    fn fftAxis(allocator: std.mem.Allocator, input: *const NDArray(Complex(f32)), axis: usize, inverse: bool) !NDArray(Complex(f32)) {
        if (axis >= input.rank()) return core.Error.IndexOutOfBounds;

        const n = input.shape[axis];
        if (!std.math.isPowerOfTwo(n)) return core.Error.InvalidShape; // Must be power of 2 for this simple implementation

        var result = try input.copy(allocator);

        // Iterate over all dimensions except axis
        var iter_shape = try allocator.alloc(usize, input.rank());
        defer allocator.free(iter_shape);
        @memcpy(iter_shape, input.shape);
        iter_shape[axis] = 1;

        var iter = try core.NdIterator.init(allocator, iter_shape);
        defer iter.deinit();

        var coords = try allocator.alloc(usize, input.rank());
        defer allocator.free(coords);

        var buffer = try allocator.alloc(Complex(f32), n);
        defer allocator.free(buffer);

        while (iter.next()) |iter_coords| {
            @memcpy(coords, iter_coords);

            // Read slice
            for (0..n) |i| {
                coords[axis] = i;
                buffer[i] = try result.get(coords);
            }

            // Apply FFT/IFFT in-place on buffer
            bit_reverse_permutation(buffer);

            var len: usize = 2;
            while (len <= n) : (len <<= 1) {
                const angle = (if (inverse) 2.0 else -2.0) * std.math.pi / @as(f32, @floatFromInt(len));
                const wlen = Complex(f32).init(std.math.cos(angle), std.math.sin(angle));

                var i: usize = 0;
                while (i < n) : (i += len) {
                    var w = Complex(f32).init(1.0, 0.0);
                    const half_len = len / 2;
                    for (0..half_len) |k| {
                        const u = buffer[i + k];
                        const v = buffer[i + k + half_len].mul(w);

                        buffer[i + k] = u.add(v);
                        buffer[i + k + half_len] = u.sub(v);

                        w = w.mul(wlen);
                    }
                }
            }

            if (inverse) {
                const scale = 1.0 / @as(f32, @floatFromInt(n));
                for (buffer) |*val| {
                    val.re *= scale;
                    val.im *= scale;
                }
            }

            // Write back
            for (0..n) |i| {
                coords[axis] = i;
                try result.set(coords, buffer[i]);
            }
        }

        return result;
    }
};

test "fft basic" {
    const allocator = std.testing.allocator;
    var a = try NDArray(f32).init(allocator, &.{4});
    defer a.deinit(allocator);
    // [1, 1, 1, 1] -> FFT -> [4, 0, 0, 0]
    a.fill(1.0);

    var res = try FFT.fft(allocator, &a);
    defer res.deinit(allocator);

    try std.testing.expectApproxEqAbs(res.data[0].re, 4.0, 1e-4);
    try std.testing.expectApproxEqAbs(res.data[1].re, 0.0, 1e-4);
}

test "fft ifft" {
    const allocator = std.testing.allocator;
    var a = try NDArray(Complex(f32)).init(allocator, &.{4});
    defer a.deinit(allocator);
    a.fill(Complex(f32).init(0, 0));
    a.data[0] = Complex(f32).init(4.0, 0.0);

    var res = try FFT.ifft(allocator, &a);
    defer res.deinit(allocator);

    // Should be [1, 1, 1, 1]
    try std.testing.expectApproxEqAbs(res.data[0].re, 1.0, 1e-4);
    try std.testing.expectApproxEqAbs(res.data[1].re, 1.0, 1e-4);
    try std.testing.expectApproxEqAbs(res.data[2].re, 1.0, 1e-4);
    try std.testing.expectApproxEqAbs(res.data[3].re, 1.0, 1e-4);
}
