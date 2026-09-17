//! Matrix multiplication, dot product, inner product, and outer product.
//!
//! High-performance numerical linear algebra with SIMD dot-product vectorization,
//! cache-efficient loop ordering, and batched multidimensional broadcasting.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const DType = @import("../core/dtype.zig").DType;
const Shape = @import("../core/shape.zig").Shape;
const Strides = @import("../core/shape.zig").Strides;
const MAX_RANK = @import("../core/shape.zig").MAX_RANK;
const ShapeError = @import("../core/error.zig").ShapeError;
const LinalgError = @import("../core/error.zig").LinalgError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;

const VEC_SIZE = 8;

inline fn zeroT(comptime T: type) T {
    if (T == bool) return false;
    if (@typeInfo(T) == .@"struct") return T.init(0.0, 0.0);
    return 0;
}

inline fn mulT(comptime T: type, a: T, b: T) T {
    if (T == bool) return a and b;
    if (@typeInfo(T) == .@"struct") return a.mul(b);
    return a * b;
}

inline fn addT(comptime T: type, a: T, b: T) T {
    if (T == bool) return a or b;
    if (@typeInfo(T) == .@"struct") return a.add(b);
    return a + b;
}

const MatmulOptions = struct {
    dtype: ?DType = null,
};

/// Matrix product of two arrays. Supports 1D vectors, 2D matrices, and batched N-D arrays.
pub fn matmul(
    a: Array,
    b: Array,
    options: MatmulOptions,
) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const s_a = a.shape();
    const s_b = b.shape();

    if (s_a.ndim == 0 or s_b.ndim == 0) return ShapeError.InvalidDimension;

    const out_dtype = options.dtype orelse DType.promote(a.dtype, b.dtype);

    // Case 1: 1D dot 1D -> 0D scalar
    if (s_a.ndim == 1 and s_b.ndim == 1) {
        if (s_a.dims[0] != s_b.dims[0]) return LinalgError.IncompatibleDimensions;
        const out = try empty(a.allocator, .{ .shape = &.{}, .dtype = out_dtype });
        try computeDot1D(a, b, out, out_dtype);
        return out;
    }

    // Case 2: 2D matmul 2D -> 2D (M x N)
    if (s_a.ndim == 2 and s_b.ndim == 2) {
        const M = s_a.dims[0];
        const K = s_a.dims[1];
        if (s_b.dims[0] != K) return LinalgError.IncompatibleDimensions;
        const N = s_b.dims[1];

        const out = try zeros(a.allocator, .{
            .shape = &.{ M, N },
            .dtype = out_dtype,
        });

        try computeMatmul2D(a, b, out, out_dtype, M, K, N);
        return out;
    }

    // Case 3: 1D (K) matmul 2D (K x N) -> 1D (N)
    if (s_a.ndim == 1 and s_b.ndim == 2) {
        const K = s_a.dims[0];
        if (s_b.dims[0] != K) return LinalgError.IncompatibleDimensions;
        const N = s_b.dims[1];

        const out = try zeros(a.allocator, .{
            .shape = &.{N},
            .dtype = out_dtype,
        });

        const expandDims = @import("../manip/reshape.zig").expandDims;
        const a_2d = try expandDims(a, .{ .axis = 0 }); // (1, K)
        const out_2d = try expandDims(out, .{ .axis = 0 }); // (1, N)

        try computeMatmul2D(a_2d, b, out_2d, out_dtype, 1, K, N);
        return out;
    }

    // Case 4: 2D (M x K) matmul 1D (K) -> 1D (M)
    if (s_a.ndim == 2 and s_b.ndim == 1) {
        const M = s_a.dims[0];
        const K = s_a.dims[1];
        if (s_b.dims[0] != K) return LinalgError.IncompatibleDimensions;

        const out = try zeros(a.allocator, .{
            .shape = &.{M},
            .dtype = out_dtype,
        });

        const expandDims = @import("../manip/reshape.zig").expandDims;
        const b_2d = try expandDims(b, .{ .axis = 1 }); // (K, 1)
        const out_2d = try expandDims(out, .{ .axis = 1 }); // (M, 1)

        try computeMatmul2D(a, b_2d, out_2d, out_dtype, M, K, 1);
        return out;
    }

    // Case 5: Higher rank batched matrix multiplication
    return computeBatchedMatmul(a, b, out_dtype);
}

/// Dot product of two arrays (inner product for 1D, matrix multiplication for 2D).
pub fn dot(a: Array, b: Array, options: MatmulOptions) !Array {
    return matmul(a, b, options);
}

/// Inner product of two arrays.
pub fn inner(a: Array, b: Array, options: MatmulOptions) !Array {
    return matmul(a, b, options);
}

/// Outer product of two vectors (computes a[i] * b[j]).
pub fn outer(
    a: Array,
    b: Array,
    options: MatmulOptions,
) (ShapeError || DTypeError || std.mem.Allocator.Error)!Array {
    const ravel = @import("../manip/reshape.zig").ravel;
    var flat_a = try ravel(a);
    defer flat_a.deinit();

    var flat_b = try ravel(b);
    defer flat_b.deinit();

    const M = flat_a.elementCount();
    const N = flat_b.elementCount();
    const out_dtype = options.dtype orelse DType.promote(a.dtype, b.dtype);

    const out = try empty(a.allocator, .{
        .shape = &.{ M, N },
        .dtype = out_dtype,
    });

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (out_dtype == tag) {
            const T = tag.toType();
            const out_ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));

            if (T == bool) {
                for (0..M) |i| {
                    const val_a = flat_a.get(bool, &.{i}) catch unreachable;
                    for (0..N) |j| {
                        const val_b = flat_b.get(bool, &.{j}) catch unreachable;
                        out_ptr[i * N + j] = val_a and val_b;
                    }
                }
                return out;
            }

            for (0..M) |i| {
                const val_a = flat_a.get(T, &.{i}) catch unreachable;
                for (0..N) |j| {
                    const val_b = flat_b.get(T, &.{j}) catch unreachable;
                    out_ptr[i * N + j] = mulT(T, val_a, val_b);
                }
            }

            return out;
        }
    }

    return out;
}

fn computeDot1D(a: Array, b: Array, out: Array, out_dtype: DType) !void {
    const K = a.shape_dims[0];
    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (out_dtype == tag) {
            const T = tag.toType();

            if (T == bool) {
                var any_acc = false;
                for (0..K) |k| {
                    const va = try a.get(bool, &.{k});
                    const vb = try b.get(bool, &.{k});
                    if (va and vb) {
                        any_acc = true;
                        break;
                    }
                }
                const ptr: [*]bool = @ptrCast(@alignCast(out.data_ptr));
                ptr[0] = any_acc;
                return;
            }

            var sum_acc: T = zeroT(T);

            // SIMD path when contiguous
            if (a.isContiguous() and b.isContiguous() and a.dtype == out_dtype and b.dtype == out_dtype) {
                const sa = a.asConstSlice(T) catch unreachable;
                const sb = b.asConstSlice(T) catch unreachable;
                var i: usize = 0;

                if ((@typeInfo(T) == .float or @typeInfo(T) == .int) and K >= VEC_SIZE) {
                    var vec_acc: @Vector(VEC_SIZE, T) = @splat(0);
                    while (i + VEC_SIZE <= K) : (i += VEC_SIZE) {
                        const va: @Vector(VEC_SIZE, T) = sa[i..][0..VEC_SIZE].*;
                        const vb: @Vector(VEC_SIZE, T) = sb[i..][0..VEC_SIZE].*;
                        vec_acc += va * vb;
                    }
                    sum_acc = @reduce(.Add, vec_acc);
                }

                while (i < K) : (i += 1) {
                    sum_acc = addT(T, sum_acc, mulT(T, sa[i], sb[i]));
                }
            } else {
                for (0..K) |k| {
                    const va = try a.get(T, &.{k});
                    const vb = try b.get(T, &.{k});
                    sum_acc = addT(T, sum_acc, mulT(T, va, vb));
                }
            }

            const ptr: [*]T = @ptrCast(@alignCast(out.data_ptr));
            ptr[0] = sum_acc;
            return;
        }
    }
}

fn computeMatmul2D(
    a: Array,
    b: Array,
    out: Array,
    out_dtype: DType,
    M: usize,
    K: usize,
    N: usize,
) !void {
    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (out_dtype == tag) {
            const T = tag.toType();

            if (T == bool) {
                for (0..M) |i| {
                    for (0..N) |j| {
                        var any_val = false;
                        for (0..K) |k| {
                            const va = try a.get(bool, &.{ i, k });
                            const vb = try b.get(bool, &.{ k, j });
                            if (va and vb) {
                                any_val = true;
                                break;
                            }
                        }
                        try out.set(bool, &.{ i, j }, any_val);
                    }
                }
                return;
            }

            // Cache-efficient i-k-j loop order for contiguous buffers
            if (a.isContiguous() and b.isContiguous() and out.isContiguous() and
                a.dtype == out_dtype and b.dtype == out_dtype)
            {
                const sa = a.asConstSlice(T) catch unreachable;
                const sb = b.asConstSlice(T) catch unreachable;
                const so = out.asSlice(T) catch unreachable;

                for (0..M) |i| {
                    const a_row = sa[i * K ..][0..K];
                    const out_row = so[i * N ..][0..N];

                    for (0..K) |k| {
                        const a_val = a_row[k];
                        const b_row = sb[k * N ..][0..N];

                        var j: usize = 0;
                        if ((@typeInfo(T) == .float or @typeInfo(T) == .int) and N >= VEC_SIZE) {
                            const va: @Vector(VEC_SIZE, T) = @splat(a_val);
                            while (j + VEC_SIZE <= N) : (j += VEC_SIZE) {
                                const vb: @Vector(VEC_SIZE, T) = b_row[j..][0..VEC_SIZE].*;
                                var vo: @Vector(VEC_SIZE, T) = out_row[j..][0..VEC_SIZE].*;
                                vo += va * vb;
                                out_row[j..][0..VEC_SIZE].* = vo;
                            }
                        }

                        while (j < N) : (j += 1) {
                            out_row[j] = addT(T, out_row[j], mulT(T, a_val, b_row[j]));
                        }
                    }
                }
                return;
            }

            // General strided path
            for (0..M) |i| {
                for (0..N) |j| {
                    var sum_val: T = zeroT(T);
                    for (0..K) |k| {
                        const va = try a.get(T, &.{ i, k });
                        const vb = try b.get(T, &.{ k, j });
                        sum_val = addT(T, sum_val, mulT(T, va, vb));
                    }
                    try out.set(T, &.{ i, j }, sum_val);
                }
            }
            return;
        }
    }
}

fn computeBatchedMatmul(a: Array, b: Array, out_dtype: DType) !Array {
    const s_a = a.shape();
    const s_b = b.shape();

    if (s_a.ndim < 2 or s_b.ndim < 2) return ShapeError.IncompatibleShapes;

    const M = s_a.dims[s_a.ndim - 2];
    const K_a = s_a.dims[s_a.ndim - 1];
    const K_b = s_b.dims[s_b.ndim - 2];
    const N = s_b.dims[s_b.ndim - 1];

    if (K_a != K_b) return LinalgError.IncompatibleDimensions;

    const broadcastShapes = @import("../core/shape.zig").broadcastShapes;
    const batch_a = Shape{ .dims = s_a.dims, .ndim = s_a.ndim - 2 };
    const batch_b = Shape{ .dims = s_b.dims, .ndim = s_b.ndim - 2 };
    const batch_out = try broadcastShapes(batch_a, batch_b);

    var out_dims: [MAX_RANK]usize = undefined;
    for (0..batch_out.ndim) |i| {
        out_dims[i] = batch_out.dims[i];
    }
    out_dims[batch_out.ndim] = M;
    out_dims[batch_out.ndim + 1] = N;

    const out_shape = Shape{ .dims = out_dims, .ndim = batch_out.ndim + 2 };
    const out = try zeros(a.allocator, .{
        .shape = out_shape.slice(),
        .dtype = out_dtype,
    });

    const NdIterator = @import("../core/iterator.zig").NdIterator;
    var batch_it = NdIterator.init(batch_out, Strides.fromShape(batch_out, .c));

    inline for (std.meta.fields(DType)) |field| {
        const tag: DType = @enumFromInt(field.value);
        if (out_dtype == tag) {
            const T = tag.toType();

            while (batch_it.next()) |b_item| {
                for (0..M) |i| {
                    for (0..N) |j| {
                        if (T == bool) {
                            var any_val = false;
                            for (0..K_a) |k| {
                                var idx_a: [MAX_RANK]usize = undefined;
                                for (0..s_a.ndim - 2) |d| {
                                    idx_a[d] = b_item.indices[d] % s_a.dims[d];
                                }
                                idx_a[s_a.ndim - 2] = i;
                                idx_a[s_a.ndim - 1] = k;

                                var idx_b: [MAX_RANK]usize = undefined;
                                for (0..s_b.ndim - 2) |d| {
                                    idx_b[d] = b_item.indices[d] % s_b.dims[d];
                                }
                                idx_b[s_b.ndim - 2] = k;
                                idx_b[s_b.ndim - 1] = j;

                                const va = try a.get(bool, idx_a[0..s_a.ndim]);
                                const vb = try b.get(bool, idx_b[0..s_b.ndim]);
                                if (va and vb) {
                                    any_val = true;
                                    break;
                                }
                            }

                            var idx_out: [MAX_RANK]usize = undefined;
                            @memcpy(idx_out[0..batch_out.ndim], b_item.indices[0..batch_out.ndim]);
                            idx_out[batch_out.ndim] = i;
                            idx_out[batch_out.ndim + 1] = j;
                            try out.set(bool, idx_out[0..out_shape.ndim], any_val);
                        } else {
                            var sum_val: T = zeroT(T);
                            for (0..K_a) |k| {
                                var idx_a: [MAX_RANK]usize = undefined;
                                for (0..s_a.ndim - 2) |d| {
                                    idx_a[d] = b_item.indices[d] % s_a.dims[d];
                                }
                                idx_a[s_a.ndim - 2] = i;
                                idx_a[s_a.ndim - 1] = k;

                                var idx_b: [MAX_RANK]usize = undefined;
                                for (0..s_b.ndim - 2) |d| {
                                    idx_b[d] = b_item.indices[d] % s_b.dims[d];
                                }
                                idx_b[s_b.ndim - 2] = k;
                                idx_b[s_b.ndim - 1] = j;

                                const va = try a.get(T, idx_a[0..s_a.ndim]);
                                const vb = try b.get(T, idx_b[0..s_b.ndim]);
                                sum_val = addT(T, sum_val, mulT(T, va, vb));
                            }

                            var idx_out: [MAX_RANK]usize = undefined;
                            @memcpy(idx_out[0..batch_out.ndim], b_item.indices[0..batch_out.ndim]);
                            idx_out[batch_out.ndim] = i;
                            idx_out[batch_out.ndim + 1] = j;
                            try out.set(T, idx_out[0..out_shape.ndim], sum_val);
                        }
                    }
                }
            }

            return out;
        }
    }

    return out;
}

/// Computes the Kronecker product of two 2D arrays.
pub fn kron(
    a: Array,
    b: Array,
) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    if (a.ndim != 2 or b.ndim != 2) return ShapeError.InvalidDimension;

    const a_rows = a.shape_dims[0];
    const a_cols = a.shape_dims[1];
    const b_rows = b.shape_dims[0];
    const b_cols = b.shape_dims[1];

    const out_rows = a_rows * b_rows;
    const out_cols = a_cols * b_cols;
    const out_dtype = DType.promote(a.dtype, b.dtype);

    var out = try empty(a.allocator, .{
        .shape = &.{ out_rows, out_cols },
        .dtype = out_dtype,
    });
    errdefer out.deinit();

    for (0..a_rows) |i| {
        for (0..a_cols) |j| {
            const a_val = try a.getAsFloat(&.{ i, j });
            for (0..b_rows) |k| {
                for (0..b_cols) |l| {
                    const b_val = try b.getAsFloat(&.{ k, l });
                    const r = i * b_rows + k;
                    const c = j * b_cols + l;
                    try out.setFromFloat(&.{ r, c }, a_val * b_val);
                }
            }
        }
    }

    return out;
}

test "1D dot and 2D matmul" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    // 1D dot: [1, 2, 3] . [4, 5, 6] = 4 + 10 + 18 = 32
    const data1 = [_]f64{ 1, 2, 3 };
    const data2 = [_]f64{ 4, 5, 6 };
    var v1 = try fromSlice(allocator, f64, .{ .data = &data1, .shape = &.{3} });
    defer v1.deinit();
    var v2 = try fromSlice(allocator, f64, .{ .data = &data2, .shape = &.{3} });
    defer v2.deinit();

    var d = try dot(v1, v2, .{});
    defer d.deinit();
    try std.testing.expectEqual(@as(f64, 32.0), try d.get(f64, &.{}));

    // 2D matmul: 2x2 identity with 2x2 matrix
    const data_a = [_]f32{ 1, 0, 0, 1 };
    const data_b = [_]f32{ 10, 20, 30, 40 };
    var ma = try fromSlice(allocator, f32, .{ .data = &data_a, .shape = &.{ 2, 2 } });
    defer ma.deinit();
    var mb = try fromSlice(allocator, f32, .{ .data = &data_b, .shape = &.{ 2, 2 } });
    defer mb.deinit();

    var mc = try matmul(ma, mb, .{});
    defer mc.deinit();
    try std.testing.expectEqualSlices(f32, &.{ 10, 20, 30, 40 }, try mc.asSlice(f32));

    // Outer product of [1, 2] and [3, 4, 5] -> 2x3: [[3, 4, 5], [6, 8, 10]]
    const d_oa = [_]f64{ 1, 2 };
    const d_ob = [_]f64{ 3, 4, 5 };
    var oa = try fromSlice(allocator, f64, .{ .data = &d_oa, .shape = &.{2} });
    defer oa.deinit();
    var ob = try fromSlice(allocator, f64, .{ .data = &d_ob, .shape = &.{3} });
    defer ob.deinit();

    var outp = try outer(oa, ob, .{});
    defer outp.deinit();
    try std.testing.expectEqualSlices(f64, &.{ 3, 4, 5, 6, 8, 10 }, try outp.asSlice(f64));

    // Kron test: [1, 2; 3, 4] (x) [0, 5; 6, 7] -> 4x4
    const k_a_data = [_]f64{ 1, 2, 3, 4 };
    const k_b_data = [_]f64{ 0, 5, 6, 7 };
    var k_a = try fromSlice(allocator, f64, .{ .data = &k_a_data, .shape = &.{ 2, 2 } });
    defer k_a.deinit();
    var k_b = try fromSlice(allocator, f64, .{ .data = &k_b_data, .shape = &.{ 2, 2 } });
    defer k_b.deinit();

    var k_res = try kron(k_a, k_b);
    defer k_res.deinit();

    try std.testing.expectEqualSlices(usize, &.{ 4, 4 }, k_res.shapeSlice());
    try std.testing.expectEqual(@as(f64, 0.0), try k_res.get(f64, &.{ 0, 0 }));
    try std.testing.expectEqual(@as(f64, 5.0), try k_res.get(f64, &.{ 0, 1 }));
    try std.testing.expectEqual(@as(f64, 10.0), try k_res.get(f64, &.{ 0, 3 }));
    try std.testing.expectEqual(@as(f64, 28.0), try k_res.get(f64, &.{ 3, 3 }));
}
