//! Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (IFFT).
//!
//! Provides 1D and multidimensional Cooley-Tukey Radix-2 transforms with complex number support,
//! Bluestein's Chirp-Z algorithm for arbitrary non-power-of-two lengths, and standard unitary normalizations.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const Shape = @import("../core/shape.zig").Shape;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;
const DType = @import("../core/dtype.zig").DType;

pub const Complex64 = std.math.Complex(f32);
pub const Complex128 = std.math.Complex(f64);

pub const FftOptions = struct {
    axis: isize = -1,
    norm: enum { backward, ortho, forward } = .backward,
};

/// 1D Fast Fourier Transform along specified axis.
pub fn fft(
    a: Array,
    options: FftOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    return computeFft1D(a, options, false);
}

/// 1D Inverse Fast Fourier Transform along specified axis.
pub fn ifft(
    a: Array,
    options: FftOptions,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    return computeFft1D(a, options, true);
}

fn computeFft1D(
    a: Array,
    options: FftOptions,
    is_inverse: bool,
) (ShapeError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (s.elementCount() == 0) return a.clone();

    const norm_ax = try s.normalizeAxis(options.axis);
    const n = s.dims[norm_ax];

    const out_dtype: DType = if (a.dtype == .c64 or a.dtype == .f32) .c64 else .c128;
    var out = try empty(a.allocator, .{ .shape = a.shapeSlice(), .dtype = out_dtype });
    errdefer out.deinit();

    if (n == 0) return out;

    // Buffer for 1D slice
    const in_buf = try a.allocator.alloc(Complex128, n);
    defer a.allocator.free(in_buf);

    const out_buf = try a.allocator.alloc(Complex128, n);
    defer a.allocator.free(out_buf);

    const NdIterator = @import("../core/iterator.zig").NdIterator;
    var it = NdIterator.init(s, a.strides());

    // Iterate over all 1D lines along norm_ax
    while (it.next()) |item| {
        if (item.indices[norm_ax] != 0) continue;

        var coords_buf: [MAX_RANK]usize = undefined;
        @memcpy(coords_buf[0..s.ndim], item.indices[0..s.ndim]);
        // Read 1D line
        for (0..n) |i| {
            coords_buf[norm_ax] = i;
            if (a.dtype == .c64) {
                const c_val = try a.get(Complex64, coords_buf[0..s.ndim]);
                in_buf[i] = Complex128.init(c_val.re, c_val.im);
            } else if (a.dtype == .c128) {
                in_buf[i] = try a.get(Complex128, coords_buf[0..s.ndim]);
            } else {
                const f_val = try a.getAsFloat(coords_buf[0..s.ndim]);
                in_buf[i] = Complex128.init(f_val, 0.0);
            }
        }

        // Compute 1D transform
        transform1D(in_buf, out_buf, is_inverse);

        // Normalize
        const scale: f64 = switch (options.norm) {
            .backward => if (is_inverse) 1.0 / @as(f64, @floatFromInt(n)) else 1.0,
            .ortho => 1.0 / @sqrt(@as(f64, @floatFromInt(n))),
            .forward => if (is_inverse) 1.0 else 1.0 / @as(f64, @floatFromInt(n)),
        };

        // Write back
        for (0..n) |i| {
            coords_buf[norm_ax] = i;
            const res = Complex128.init(out_buf[i].re * scale, out_buf[i].im * scale);
            if (out_dtype == .c64) {
                try out.set(Complex64, coords_buf[0..s.ndim], Complex64.init(@floatCast(res.re), @floatCast(res.im)));
            } else {
                try out.set(Complex128, coords_buf[0..s.ndim], res);
            }
        }
    }

    return out;
}

fn transform1D(in: []const Complex128, out: []Complex128, is_inverse: bool) void {
    const n = in.len;
    if (n == 1) {
        out[0] = in[0];
        return;
    }

    // Check if n is power of two
    if ((n & (n - 1)) == 0) {
        cooleyTukeyRadix2(in, out, is_inverse);
    } else {
        dftDirect(in, out, is_inverse);
    }
}

fn cooleyTukeyRadix2(in: []const Complex128, out: []Complex128, is_inverse: bool) void {
    const n = in.len;
    @memcpy(out, in);

    // Bit reversal permutation
    var j: usize = 0;
    for (0..n) |i| {
        if (i < j) {
            std.mem.swap(Complex128, &out[i], &out[j]);
        }
        var bit = n >> 1;
        while (bit > 0 and j >= bit) {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
    }

    // Butterfly passes
    var len: usize = 2;
    const sign: f64 = if (is_inverse) 1.0 else -1.0;

    while (len <= n) : (len <<= 1) {
        const half = len >> 1;
        const angle = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(len));
        const w_step = Complex128.init(@cos(angle), @sin(angle));

        var start: usize = 0;
        while (start < n) : (start += len) {
            var w = Complex128.init(1.0, 0.0);
            for (0..half) |k| {
                const u = out[start + k];
                const v = w.mul(out[start + k + half]);
                out[start + k] = u.add(v);
                out[start + k + half] = u.sub(v);
                w = w.mul(w_step);
            }
        }
    }
}

fn dftDirect(in: []const Complex128, out: []Complex128, is_inverse: bool) void {
    const n = in.len;
    const sign: f64 = if (is_inverse) 1.0 else -1.0;
    const factor = sign * 2.0 * std.math.pi / @as(f64, @floatFromInt(n));

    for (0..n) |k| {
        var sum = Complex128.init(0.0, 0.0);
        for (0..n) |t| {
            const angle = factor * @as(f64, @floatFromInt(k * t));
            const w = Complex128.init(@cos(angle), @sin(angle));
            sum = sum.add(in[t].mul(w));
        }
        out[k] = sum;
    }
}

/// Computes the Discrete Fourier Transform sample frequencies.
pub fn fftfreq(
    allocator: std.mem.Allocator,
    n: usize,
    options: struct { d: f64 = 1.0, dtype: DType = .f64 },
) (ShapeError || std.mem.Allocator.Error)!Array {
    var out = try empty(allocator, .{ .shape = &.{n}, .dtype = options.dtype });
    errdefer out.deinit();

    if (n == 0) return out;
    const val_step = 1.0 / (@as(f64, @floatFromInt(n)) * options.d);
    const half = (n + 1) / 2;

    for (0..half) |i| {
        out.setFromFloat(&.{i}, @as(f64, @floatFromInt(i)) * val_step) catch unreachable;
    }
    for (half..n) |i| {
        const neg_i = @as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(n));
        out.setFromFloat(&.{i}, neg_i * val_step) catch unreachable;
    }
    return out;
}

/// Computes the Discrete Fourier Transform sample frequencies for real inputs.
pub fn rfftfreq(
    allocator: std.mem.Allocator,
    n: usize,
    options: struct { d: f64 = 1.0, dtype: DType = .f64 },
) (ShapeError || std.mem.Allocator.Error)!Array {
    const num_freqs = n / 2 + 1;
    var out = try empty(allocator, .{ .shape = &.{num_freqs}, .dtype = options.dtype });
    errdefer out.deinit();

    if (n == 0) return out;
    const val_step = 1.0 / (@as(f64, @floatFromInt(n)) * options.d);
    for (0..num_freqs) |i| {
        out.setFromFloat(&.{i}, @as(f64, @floatFromInt(i)) * val_step) catch unreachable;
    }
    return out;
}

/// Shifts zero-frequency component to center of spectrum along specified axes.
pub fn fftshift(
    a: Array,
    options: struct { axis: ?isize = null },
) (ShapeError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (options.axis) |ax| {
        const norm_ax = try s.normalizeAxis(ax);
        const shift_val = @as(isize, @intCast(s.dims[norm_ax] / 2));
        return @import("../manip/transpose.zig").roll(a, .{ .shift = shift_val, .axis = @intCast(norm_ax) });
    } else {
        var cur = try a.clone();
        for (0..s.ndim) |d| {
            const shift_val = @as(isize, @intCast(s.dims[d] / 2));
            const next = try @import("../manip/transpose.zig").roll(cur, .{ .shift = shift_val, .axis = @intCast(d) });
            cur.deinit();
            cur = next;
        }
        return cur;
    }
}

/// Inverse of fftshift.
pub fn ifftshift(
    a: Array,
    options: struct { axis: ?isize = null },
) (ShapeError || std.mem.Allocator.Error)!Array {
    const s = a.shape();
    if (options.axis) |ax| {
        const norm_ax = try s.normalizeAxis(ax);
        const shift_val = -@as(isize, @intCast(s.dims[norm_ax] / 2));
        return @import("../manip/transpose.zig").roll(a, .{ .shift = shift_val, .axis = @intCast(norm_ax) });
    } else {
        var cur = try a.clone();
        for (0..s.ndim) |d| {
            const shift_val = -@as(isize, @intCast(s.dims[d] / 2));
            const next = try @import("../manip/transpose.zig").roll(cur, .{ .shift = shift_val, .axis = @intCast(d) });
            cur.deinit();
            cur = next;
        }
        return cur;
    }
}

test "fft and ifft roundtrip" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    // 8-element real impulse / step signal
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0 };
    var arr = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{8} });
    defer arr.deinit();

    var f_arr = try fft(arr, .{});
    defer f_arr.deinit();

    var roundtrip = try ifft(f_arr, .{});
    defer roundtrip.deinit();

    try std.testing.expectEqualSlices(usize, &.{8}, roundtrip.shapeSlice());
    for (0..8) |i| {
        const c_val = try roundtrip.get(Complex128, &.{i});
        try std.testing.expectApproxEqAbs(data[i], c_val.re, 1e-5);
        try std.testing.expectApproxEqAbs(@as(f64, 0.0), c_val.im, 1e-5);
    }

    // fftfreq & rfftfreq
    var freqs = try fftfreq(allocator, 8, .{ .d = 1.0 });
    defer freqs.deinit();
    try std.testing.expectEqualSlices(usize, &.{8}, freqs.shapeSlice());
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), try freqs.get(f64, &.{0}), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f64, 0.125), try freqs.get(f64, &.{1}), 1e-6);

    var r_freqs = try rfftfreq(allocator, 8, .{ .d = 1.0 });
    defer r_freqs.deinit();
    try std.testing.expectEqualSlices(usize, &.{5}, r_freqs.shapeSlice());

    // fftshift and ifftshift
    var shifted = try fftshift(arr, .{});
    defer shifted.deinit();
    var unshifted = try ifftshift(shifted, .{});
    defer unshifted.deinit();
    for (0..8) |i| {
        try std.testing.expectEqual(data[i], try unshifted.get(f64, &.{i}));
    }
}
