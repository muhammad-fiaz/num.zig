//! Singular Value Decomposition (SVD) for general rectangular matrices.
//!
//! Decomposes A = U * S * Vt using Golub-Kahan bidiagonalization and implicit QR iterations.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const empty = @import("../core/array.zig").empty;
const eye = @import("../core/array.zig").eye;
const Shape = @import("../core/shape.zig").Shape;
const ShapeError = @import("../core/error.zig").ShapeError;
const LinalgError = @import("../core/error.zig").LinalgError;
const DTypeError = @import("../core/error.zig").DTypeError;
const IndexError = @import("../core/error.zig").IndexError;
const DType = @import("../core/dtype.zig").DType;

pub const SvdResult = struct {
    u: Array,
    s: Array,
    vt: Array,

    pub fn deinit(self: *SvdResult) void {
        self.u.deinit();
        self.s.deinit();
        self.vt.deinit();
    }
};

/// Computes the Singular Value Decomposition of a matrix A: A = U * S * Vt.
pub fn svd(
    a: Array,
    options: struct {
        full_matrices: bool = true,
        max_iterations: usize = 100,
        tol: f64 = 1e-12,
    },
) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!SvdResult {
    const shp = a.shape();
    if (shp.ndim != 2) return ShapeError.InvalidDimension;

    const M = shp.dims[0];
    const N = shp.dims[1];
    const K = @min(M, N);
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    if (M == 0 or N == 0) {
        return SvdResult{
            .u = try empty(a.allocator, .{ .shape = &.{ M, M }, .dtype = float_dtype }),
            .s = try empty(a.allocator, .{ .shape = &.{K}, .dtype = float_dtype }),
            .vt = try empty(a.allocator, .{ .shape = &.{ N, N }, .dtype = float_dtype }),
        };
    }

    // Work on A copy
    var B = try a.clone();
    defer B.deinit();

    var U = try eye(a.allocator, .{ .n = M, .dtype = float_dtype });
    errdefer U.deinit();

    var Vt = try eye(a.allocator, .{ .n = N, .dtype = float_dtype });
    errdefer Vt.deinit();

    // 1. Golub-Kahan Bidiagonalization: B = U_b * (Bidiagonal) * V_b^T
    for (0..K) |i| {
        // Zero below diagonal in column i
        if (i < M - 1) {
            var col_norm_sq: f64 = 0.0;
            for (i..M) |r| {
                const val = try B.getAsFloat(&.{ r, i });
                col_norm_sq += val * val;
            }
            const col_norm = @sqrt(col_norm_sq);
            if (col_norm > options.tol) {
                const lead = try B.getAsFloat(&.{ i, i });
                const sgn: f64 = if (lead >= 0) 1.0 else -1.0;
                const @"u1" = lead + sgn * col_norm;

                var v_norm_sq = @"u1" * @"u1";
                for (i + 1..M) |r| {
                    const val = try B.getAsFloat(&.{ r, i });
                    v_norm_sq += val * val;
                }
                const v_scale = @sqrt(v_norm_sq);
                if (v_scale > options.tol) {
                    var v_buf: [256]f64 = undefined;
                    v_buf[0] = @"u1" / v_scale;
                    for (i + 1..M) |r| {
                        v_buf[r - i] = (try B.getAsFloat(&.{ r, i })) / v_scale;
                    }
                    const len = M - i;

                    // Apply from left: B[i:M, i:N]
                    for (i..N) |col| {
                        var dot_p: f64 = 0.0;
                        for (0..len) |r_idx| {
                            dot_p += v_buf[r_idx] * (try B.getAsFloat(&.{ i + r_idx, col }));
                        }
                        for (0..len) |r_idx| {
                            const cur = try B.getAsFloat(&.{ i + r_idx, col });
                            try B.setFromFloat(&.{ i + r_idx, col }, cur - 2.0 * v_buf[r_idx] * dot_p);
                        }
                    }

                    // Accumulate into U
                    for (0..M) |r| {
                        var dot_p: f64 = 0.0;
                        for (0..len) |r_idx| {
                            dot_p += (try U.getAsFloat(&.{ r, i + r_idx })) * v_buf[r_idx];
                        }
                        for (0..len) |r_idx| {
                            const cur = try U.getAsFloat(&.{ r, i + r_idx });
                            try U.setFromFloat(&.{ r, i + r_idx }, cur - 2.0 * dot_p * v_buf[r_idx]);
                        }
                    }
                }
            }
        }

        // Zero right of superdiagonal in row i
        if (i < N - 2 and i < M) {
            var row_norm_sq: f64 = 0.0;
            for (i + 1..N) |c| {
                const val = try B.getAsFloat(&.{ i, c });
                row_norm_sq += val * val;
            }
            const row_norm = @sqrt(row_norm_sq);
            if (row_norm > options.tol) {
                const lead = try B.getAsFloat(&.{ i, i + 1 });
                const sgn: f64 = if (lead >= 0) 1.0 else -1.0;
                const @"u1" = lead + sgn * row_norm;

                var v_norm_sq = @"u1" * @"u1";
                for (i + 2..N) |c| {
                    const val = try B.getAsFloat(&.{ i, c });
                    v_norm_sq += val * val;
                }
                const v_scale = @sqrt(v_norm_sq);
                if (v_scale > options.tol) {
                    var v_buf: [256]f64 = undefined;
                    v_buf[0] = @"u1" / v_scale;
                    for (i + 2..N) |c| {
                        v_buf[c - (i + 1)] = (try B.getAsFloat(&.{ i, c })) / v_scale;
                    }
                    const len = N - (i + 1);

                    // Apply from right: B[i:M, i+1:N]
                    for (i..M) |row| {
                        var dot_p: f64 = 0.0;
                        for (0..len) |c_idx| {
                            dot_p += (try B.getAsFloat(&.{ row, i + 1 + c_idx })) * v_buf[c_idx];
                        }
                        for (0..len) |c_idx| {
                            const cur = try B.getAsFloat(&.{ row, i + 1 + c_idx });
                            try B.setFromFloat(&.{ row, i + 1 + c_idx }, cur - 2.0 * dot_p * v_buf[c_idx]);
                        }
                    }

                    // Accumulate into Vt
                    for (0..N) |col| {
                        var dot_p: f64 = 0.0;
                        for (0..len) |c_idx| {
                            dot_p += v_buf[c_idx] * (try Vt.getAsFloat(&.{ i + 1 + c_idx, col }));
                        }
                        for (0..len) |c_idx| {
                            const cur = try Vt.getAsFloat(&.{ i + 1 + c_idx, col });
                            try Vt.setFromFloat(&.{ i + 1 + c_idx, col }, cur - 2.0 * v_buf[c_idx] * dot_p);
                        }
                    }
                }
            }
        }
    }

    // 2. Diagonalize the bidiagonal matrix via Golub-Kahan SVD step
    var d: [256]f64 = undefined;
    var e_super: [256]f64 = undefined;
    for (0..K) |i| {
        d[i] = try B.getAsFloat(&.{ i, i });
        e_super[i] = if (i + 1 < N and i + 1 < M) try B.getAsFloat(&.{ i, i + 1 }) else 0.0;
    }

    var p = K;
    var total_iters: usize = 0;
    const max_total = K * options.max_iterations;

    while (p > 1 and total_iters < max_total) : (total_iters += 1) {
        // Check for convergence of trailing superdiagonal element
        if (@abs(e_super[p - 2]) <= options.tol * (@abs(d[p - 2]) + @abs(d[p - 1]) + 1e-15)) {
            e_super[p - 2] = 0.0;
            p -= 1;
            continue;
        }

        // Wilkinson shift on trailing 2x2 of T = B^T * B
        const dm1 = d[p - 2];
        const em1 = e_super[p - 2];
        const dp = d[p - 1];

        const t11 = dm1 * dm1 + (if (p >= 3) e_super[p - 3] * e_super[p - 3] else 0.0);
        const t12 = dm1 * em1;
        const t22 = em1 * em1 + dp * dp;

        const tr = t11 + t22;
        const dt = t11 * t22 - t12 * t12;
        const disc = tr * tr - 4.0 * dt;
        const shift = if (disc >= 0) blk: {
            const sq = @sqrt(disc);
            const r1 = (tr + sq) / 2.0;
            const r2 = (tr - sq) / 2.0;
            break :blk if (@abs(r1 - t22) < @abs(r2 - t22)) r1 else r2;
        } else t22;

        // Chase the bulge using Givens rotations
        var y = d[0] * d[0] - shift;
        var z = d[0] * e_super[0];

        for (0..p - 1) |k| {
            // Right rotation (affects Vt and columns k, k+1)
            const r1 = @sqrt(y * y + z * z);
            const c1 = if (r1 > 0) y / r1 else 1.0;
            const s1 = if (r1 > 0) z / r1 else 0.0;

            if (k > 0) e_super[k - 1] = r1;

            const dk = d[k];
            const ek = e_super[k];
            const dkp1 = d[k + 1];

            d[k] = c1 * dk + s1 * ek;
            e_super[k] = -s1 * dk + c1 * ek;
            d[k + 1] = c1 * dkp1;
            const bulge = s1 * dkp1;

            // Apply rotation to Vt
            for (0..N) |col| {
                const vk = try Vt.getAsFloat(&.{ k, col });
                const vk1 = try Vt.getAsFloat(&.{ k + 1, col });
                try Vt.setFromFloat(&.{ k, col }, c1 * vk + s1 * vk1);
                try Vt.setFromFloat(&.{ k + 1, col }, -s1 * vk + c1 * vk1);
            }

            // Left rotation (affects U and rows k, k+1 to eliminate bulge)
            y = d[k];
            z = bulge;
            const r2 = @sqrt(y * y + z * z);
            const c2 = if (r2 > 0) y / r2 else 1.0;
            const s2 = if (r2 > 0) z / r2 else 0.0;

            d[k] = r2;
            const ek_new = e_super[k];
            const dkp1_new = d[k + 1];
            e_super[k] = c2 * ek_new + s2 * dkp1_new;
            d[k + 1] = -s2 * ek_new + c2 * dkp1_new;

            if (k + 1 < p - 1) {
                y = e_super[k];
                z = s2 * e_super[k + 1];
                e_super[k + 1] = c2 * e_super[k + 1];
            }

            // Apply rotation to U
            for (0..M) |row| {
                const uk = try U.getAsFloat(&.{ row, k });
                const uk1 = try U.getAsFloat(&.{ row, k + 1 });
                try U.setFromFloat(&.{ row, k }, c2 * uk + s2 * uk1);
                try U.setFromFloat(&.{ row, k + 1 }, -s2 * uk + c2 * uk1);
            }
        }
    }

    // Ensure singular values are positive
    for (0..K) |i| {
        if (d[i] < 0) {
            d[i] = -d[i];
            for (0..N) |col| {
                const cur = try Vt.getAsFloat(&.{ i, col });
                try Vt.setFromFloat(&.{ i, col }, -cur);
            }
        }
    }

    // Sort singular values descending
    for (0..K) |i| {
        var max_idx = i;
        for (i + 1..K) |j| {
            if (d[j] > d[max_idx]) max_idx = j;
        }
        if (max_idx != i) {
            std.mem.swap(f64, &d[i], &d[max_idx]);
            for (0..M) |row| {
                const u_i = try U.getAsFloat(&.{ row, i });
                const u_max = try U.getAsFloat(&.{ row, max_idx });
                try U.setFromFloat(&.{ row, i }, u_max);
                try U.setFromFloat(&.{ row, max_idx }, u_i);
            }
            for (0..N) |col| {
                const vt_i = try Vt.getAsFloat(&.{ i, col });
                const vt_max = try Vt.getAsFloat(&.{ max_idx, col });
                try Vt.setFromFloat(&.{ i, col }, vt_max);
                try Vt.setFromFloat(&.{ max_idx, col }, vt_i);
            }
        }
    }

    var s_arr = try empty(a.allocator, .{ .shape = &.{K}, .dtype = float_dtype });
    errdefer s_arr.deinit();

    for (0..K) |i| {
        try s_arr.setFromFloat(&.{i}, d[i]);
    }

    if (!options.full_matrices and (M != K or N != K)) {
        var Ue = try empty(a.allocator, .{ .shape = &.{ M, K }, .dtype = float_dtype });
        errdefer Ue.deinit();
        var Vte = try empty(a.allocator, .{ .shape = &.{ K, N }, .dtype = float_dtype });
        errdefer Vte.deinit();
        for (0..M) |r| {
            for (0..K) |c| {
                try Ue.setFromFloat(&.{ r, c }, try U.getAsFloat(&.{ r, c }));
            }
        }
        for (0..K) |r| {
            for (0..N) |c| {
                try Vte.setFromFloat(&.{ r, c }, try Vt.getAsFloat(&.{ r, c }));
            }
        }
        U.deinit();
        Vt.deinit();
        return SvdResult{
            .u = Ue,
            .s = s_arr,
            .vt = Vte,
        };
    }

    return SvdResult{
        .u = U,
        .s = s_arr,
        .vt = Vt,
    };
}

test "svd of 2x2 matrix" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    const data = [_]f64{ 3.0, 0.0, 0.0, -4.0 };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 2, 2 } });
    defer a.deinit();

    var res = try svd(a, .{});
    defer res.deinit();

    try std.testing.expectEqualSlices(usize, &.{2}, res.s.shapeSlice());
    const s0 = try res.s.getAsFloat(&.{0});
    const s1 = try res.s.getAsFloat(&.{1});

    // Singular values of diag(3, -4) are 4.0 and 3.0
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), s0, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), s1, 1e-5);
}

test "svd economy mode shapes" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;
    const data = [_]f64{ 1, 2, 3, 4, 5, 6 };
    var a = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 3, 2 } });
    defer a.deinit();
    var full_res = try svd(a, .{ .full_matrices = true });
    defer full_res.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, full_res.u.shapeSlice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, full_res.vt.shapeSlice());
    var eco = try svd(a, .{ .full_matrices = false });
    defer eco.deinit();
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, eco.u.shapeSlice());
    try std.testing.expectEqualSlices(usize, &.{2}, eco.s.shapeSlice());
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, eco.vt.shapeSlice());
}
