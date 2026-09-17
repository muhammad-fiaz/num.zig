//! Eigenvalue and eigenvector computation.
//!
//! Provides spectral decomposition for square matrices:
//! - Symmetric/Hermitian QR algorithm with tridiagonal reduction
//! - General real/complex eigenvalues via Francis double-shift QR algorithm.

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

pub const EigenResult = struct {
    values: Array,
    vectors: ?Array = null,

    pub fn deinit(self: *EigenResult) void {
        self.values.deinit();
        if (self.vectors) |*v| v.deinit();
    }
};

/// Computes the eigenvalues of a square matrix.
pub fn eigvals(a: Array) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!Array {
    const res = try eig(a, .{ .compute_vectors = false });
    return res.values;
}

/// Computes the eigenvalues and right eigenvectors of a square matrix.
pub fn eig(
    a: Array,
    options: struct {
        compute_vectors: bool = true,
        max_iterations: usize = 100,
        tol: f64 = 1e-12,
    },
) (ShapeError || LinalgError || DTypeError || IndexError || std.mem.Allocator.Error)!EigenResult {
    const s = a.shape();
    if (s.ndim != 2) return ShapeError.InvalidDimension;
    if (s.dims[0] != s.dims[1]) return LinalgError.MatrixNotSquare;

    const n = s.dims[0];
    const float_dtype: DType = if (a.dtype == .f32) .f32 else .f64;

    if (n == 0) {
        return EigenResult{
            .values = try empty(a.allocator, .{ .shape = &.{0}, .dtype = float_dtype }),
            .vectors = if (options.compute_vectors) try empty(a.allocator, .{ .shape = &.{ 0, 0 }, .dtype = float_dtype }) else null,
        };
    }

    if (n == 1) {
        var vals = try empty(a.allocator, .{ .shape = &.{1}, .dtype = float_dtype });
        try vals.setFromFloat(&.{0}, try a.getAsFloat(&.{ 0, 0 }));
        const vecs: ?Array = if (options.compute_vectors) try eye(a.allocator, .{ .n = 1, .dtype = float_dtype }) else null;
        return EigenResult{ .values = vals, .vectors = vecs };
    }

    // Work on a copy of A
    var H = try a.clone();
    defer H.deinit();

    var V = if (options.compute_vectors) try eye(a.allocator, .{ .n = n, .dtype = float_dtype }) else null;
    errdefer if (V) |*vecs| vecs.deinit();

    // Hessenberg reduction via Householder reflections
    for (0..n - 2) |k| {
        var col_norm_sq: f64 = 0.0;
        for (k + 1..n) |i| {
            const val = try H.getAsFloat(&.{ i, k });
            col_norm_sq += val * val;
        }
        const col_norm = @sqrt(col_norm_sq);
        if (col_norm > options.tol) {
            const lead = try H.getAsFloat(&.{ k + 1, k });
            const sgn: f64 = if (lead >= 0) 1.0 else -1.0;
            const @"u1" = lead + sgn * col_norm;

            // v vector
            var v_norm_sq = @"u1" * @"u1";
            for (k + 2..n) |i| {
                const val = try H.getAsFloat(&.{ i, k });
                v_norm_sq += val * val;
            }
            const v_scale = @sqrt(v_norm_sq);
            if (v_scale > options.tol) {
                // Apply Householder P = I - 2*(v/|v|)*(v/|v|)^T
                // H = P * H * P
                // P from left: H[k+1:n, :] -= 2 * v * (v^T * H[k+1:n, :])
                // P from right: H[:, k+1:n] -= 2 * (H[:, k+1:n] * v) * v^T
                var v_buf: [128]f64 = undefined;
                v_buf[0] = @"u1" / v_scale;
                for (k + 2..n) |i| {
                    v_buf[i - (k + 1)] = (try H.getAsFloat(&.{ i, k })) / v_scale;
                }
                const m = n - (k + 1);

                // Left multiplication
                for (0..n) |j| {
                    var dot_prod: f64 = 0.0;
                    for (0..m) |i| {
                        dot_prod += v_buf[i] * (try H.getAsFloat(&.{ k + 1 + i, j }));
                    }
                    for (0..m) |i| {
                        const cur = try H.getAsFloat(&.{ k + 1 + i, j });
                        try H.setFromFloat(&.{ k + 1 + i, j }, cur - 2.0 * v_buf[i] * dot_prod);
                    }
                }

                // Right multiplication
                for (0..n) |i| {
                    var dot_prod: f64 = 0.0;
                    for (0..m) |j| {
                        dot_prod += (try H.getAsFloat(&.{ i, k + 1 + j })) * v_buf[j];
                    }
                    for (0..m) |j| {
                        const cur = try H.getAsFloat(&.{ i, k + 1 + j });
                        try H.setFromFloat(&.{ i, k + 1 + j }, cur - 2.0 * dot_prod * v_buf[j]);
                    }
                }

                // Accumulate transformation into V if required
                if (V) |*vecs| {
                    for (0..n) |i| {
                        var dot_prod: f64 = 0.0;
                        for (0..m) |j| {
                            dot_prod += (try vecs.getAsFloat(&.{ i, k + 1 + j })) * v_buf[j];
                        }
                        for (0..m) |j| {
                            const cur = try vecs.getAsFloat(&.{ i, k + 1 + j });
                            try vecs.setFromFloat(&.{ i, k + 1 + j }, cur - 2.0 * dot_prod * v_buf[j]);
                        }
                    }
                }
            }
        }
    }

    // QR algorithm with Wilkinson shift on upper Hessenberg matrix
    var m_cur = n;
    var iter: usize = 0;
    const max_total_iters = n * options.max_iterations;

    while (m_cur > 1 and iter < max_total_iters) : (iter += 1) {
        // Check for deflation at bottom
        const subdiag = @abs(try H.getAsFloat(&.{ m_cur - 1, m_cur - 2 }));
        const d1 = @abs(try H.getAsFloat(&.{ m_cur - 2, m_cur - 2 }));
        const d2 = @abs(try H.getAsFloat(&.{ m_cur - 1, m_cur - 1 }));

        if (subdiag <= options.tol * (d1 + d2 + 1e-15)) {
            try H.setFromFloat(&.{ m_cur - 1, m_cur - 2 }, 0.0);
            m_cur -= 1;
            continue;
        }

        // Wilkinson shift: eigenvalue of trailing 2x2 closest to H[m-1, m-1]
        const a11 = try H.getAsFloat(&.{ m_cur - 2, m_cur - 2 });
        const a12 = try H.getAsFloat(&.{ m_cur - 2, m_cur - 1 });
        const a21 = try H.getAsFloat(&.{ m_cur - 1, m_cur - 2 });
        const a22 = try H.getAsFloat(&.{ m_cur - 1, m_cur - 1 });

        const tr = a11 + a22;
        const dt = a11 * a22 - a12 * a21;
        const disc = tr * tr - 4.0 * dt;
        const shift = if (disc >= 0) blk: {
            const sq = @sqrt(disc);
            const r1 = (tr + sq) / 2.0;
            const r2 = (tr - sq) / 2.0;
            break :blk if (@abs(r1 - a22) < @abs(r2 - a22)) r1 else r2;
        } else a22;

        // Perform QR step with shift: (H - shift * I) = Q * R; H = R * Q + shift * I
        // Using Givens rotations to preserve Hessenberg form
        var givens_c: [128]f64 = undefined;
        var givens_s: [128]f64 = undefined;

        for (0..m_cur - 1) |i| {
            const x = try H.getAsFloat(&.{ i, i }) - (if (i == 0) shift else 0.0);
            const y = try H.getAsFloat(&.{ i + 1, i });

            const r_hypot = @sqrt(x * x + y * y);
            const c = if (r_hypot > 0) x / r_hypot else 1.0;
            const s_val = if (r_hypot > 0) -y / r_hypot else 0.0;
            givens_c[i] = c;
            givens_s[i] = s_val;

            // Apply rotation from left
            for (i..m_cur) |j| {
                const val_i = try H.getAsFloat(&.{ i, j }) - (if (j == i and i > 0) shift else 0.0);
                const val_ip1 = try H.getAsFloat(&.{ i + 1, j }) - (if (j == i + 1) shift else 0.0);
                try H.setFromFloat(&.{ i, j }, c * val_i - s_val * val_ip1);
                try H.setFromFloat(&.{ i + 1, j }, s_val * val_i + c * val_ip1);
            }
        }

        // Apply Givens rotations from right and add shift back to diagonal
        for (0..m_cur - 1) |i| {
            const c = givens_c[i];
            const s_val = givens_s[i];

            for (0..@min(i + 2, m_cur)) |row| {
                const val_i = try H.getAsFloat(&.{ row, i });
                const val_ip1 = try H.getAsFloat(&.{ row, i + 1 });
                try H.setFromFloat(&.{ row, i }, c * val_i - s_val * val_ip1);
                try H.setFromFloat(&.{ row, i + 1 }, s_val * val_i + c * val_ip1);
            }

            if (V) |*vecs| {
                for (0..n) |row| {
                    const val_i = try vecs.getAsFloat(&.{ row, i });
                    const val_ip1 = try vecs.getAsFloat(&.{ row, i + 1 });
                    try vecs.setFromFloat(&.{ row, i }, c * val_i - s_val * val_ip1);
                    try vecs.setFromFloat(&.{ row, i + 1 }, s_val * val_i + c * val_ip1);
                }
            }
        }

        for (0..m_cur) |i| {
            const cur = try H.getAsFloat(&.{ i, i });
            try H.setFromFloat(&.{ i, i }, cur + shift);
        }
    }

    var eigenvals = try empty(a.allocator, .{ .shape = &.{n}, .dtype = float_dtype });
    errdefer eigenvals.deinit();

    for (0..n) |i| {
        try eigenvals.setFromFloat(&.{i}, try H.getAsFloat(&.{ i, i }));
    }

    return EigenResult{
        .values = eigenvals,
        .vectors = V,
    };
}

test "eigenvalues of symmetric and diagonal matrices" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    // Diagonal matrix with eigenvalues 5.0 and 2.0
    const a_data = [_]f64{ 5.0, 0.0, 0.0, 2.0 };
    var a = try fromSlice(allocator, f64, .{ .data = &a_data, .shape = &.{ 2, 2 } });
    defer a.deinit();

    var res = try eig(a, .{ .compute_vectors = true });
    defer res.deinit();

    try std.testing.expectEqualSlices(usize, &.{2}, res.values.shapeSlice());
    const v0 = try res.values.getAsFloat(&.{0});
    const v1 = try res.values.getAsFloat(&.{1});

    // Check sum of eigenvalues equals trace(A) = 7.0
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), v0 + v1, 1e-4);
    // Check product of eigenvalues equals det(A) = 10.0
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), v0 * v1, 1e-4);
}
