//! Sparse matrix formats: Compressed Sparse Row (CSR) and Compressed Sparse Column (CSC).
//!
//! Provides memory-efficient sparse representations with explicit allocator discipline,
//! matrix-vector multiplication, dense conversion, and indexing.

const std = @import("std");
const Array = @import("../core/array.zig").Array;
const zeros = @import("../core/array.zig").zeros;
const DType = @import("../core/dtype.zig").DType;
const ShapeError = @import("../core/error.zig").ShapeError;
const IndexError = @import("../core/error.zig").IndexError;

/// Compressed Sparse Row (CSR) matrix representation.
pub const CsrMatrix = struct {
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    data: []f64,
    indices: []usize,
    indptr: []usize,

    pub fn deinit(self: *CsrMatrix) void {
        self.allocator.free(self.data);
        self.allocator.free(self.indices);
        self.allocator.free(self.indptr);
        self.* = undefined;
    }

    /// Number of stored non-zero elements.
    pub fn nnz(self: CsrMatrix) usize {
        return self.data.len;
    }

    /// Constructs a CSR matrix from a dense Array.
    pub fn fromDense(allocator: std.mem.Allocator, dense: Array, tol: f64) !CsrMatrix {
        if (dense.ndim != 2) return ShapeError.InvalidDimension;
        const rows = dense.shape_dims[0];
        const cols = dense.shape_dims[1];

        var nnz_count: usize = 0;
        for (0..rows) |r| {
            for (0..cols) |c| {
                const val = try dense.getAsFloat(&.{ r, c });
                if (@abs(val) > tol) nnz_count += 1;
            }
        }

        const data = try allocator.alloc(f64, nnz_count);
        errdefer allocator.free(data);

        const indices = try allocator.alloc(usize, nnz_count);
        errdefer allocator.free(indices);

        const indptr = try allocator.alloc(usize, rows + 1);
        errdefer allocator.free(indptr);

        var idx: usize = 0;
        indptr[0] = 0;
        for (0..rows) |r| {
            for (0..cols) |c| {
                const val = try dense.getAsFloat(&.{ r, c });
                if (@abs(val) > tol) {
                    data[idx] = val;
                    indices[idx] = c;
                    idx += 1;
                }
            }
            indptr[r + 1] = idx;
        }

        return CsrMatrix{
            .allocator = allocator,
            .rows = rows,
            .cols = cols,
            .data = data,
            .indices = indices,
            .indptr = indptr,
        };
    }

    /// Converts the CSR matrix back to a dense 2D Array.
    pub fn toDense(self: CsrMatrix) !Array {
        var arr = try zeros(self.allocator, .{ .shape = &.{ self.rows, self.cols }, .dtype = .f64 });
        errdefer arr.deinit();

        for (0..self.rows) |r| {
            const start = self.indptr[r];
            const end = self.indptr[r + 1];
            for (start..end) |i| {
                const c = self.indices[i];
                const val = self.data[i];
                try arr.setFromFloat(&.{ r, c }, val);
            }
        }

        return arr;
    }

    /// Matrix-vector multiplication: y = A * x.
    pub fn dotVector(self: CsrMatrix, x: Array) !Array {
        if (x.ndim != 1 or x.shape_dims[0] != self.cols) return ShapeError.IncompatibleShapes;

        var y = try zeros(self.allocator, .{ .shape = &.{self.rows}, .dtype = .f64 });
        errdefer y.deinit();

        for (0..self.rows) |r| {
            var sum: f64 = 0.0;
            const start = self.indptr[r];
            const end = self.indptr[r + 1];
            for (start..end) |i| {
                const col = self.indices[i];
                const x_val = try x.getAsFloat(&.{col});
                sum += self.data[i] * x_val;
            }
            try y.setFromFloat(&.{r}, sum);
        }

        return y;
    }
};

/// Compressed Sparse Column (CSC) matrix representation.
pub const CscMatrix = struct {
    allocator: std.mem.Allocator,
    rows: usize,
    cols: usize,
    data: []f64,
    indices: []usize, // Row indices
    indptr: []usize, // Column pointers

    pub fn deinit(self: *CscMatrix) void {
        self.allocator.free(self.data);
        self.allocator.free(self.indices);
        self.allocator.free(self.indptr);
        self.* = undefined;
    }

    pub fn nnz(self: CscMatrix) usize {
        return self.data.len;
    }

    pub fn fromDense(allocator: std.mem.Allocator, dense: Array, tol: f64) !CscMatrix {
        if (dense.ndim != 2) return ShapeError.InvalidDimension;
        const rows = dense.shape_dims[0];
        const cols = dense.shape_dims[1];

        var nnz_count: usize = 0;
        for (0..cols) |c| {
            for (0..rows) |r| {
                const val = try dense.getAsFloat(&.{ r, c });
                if (@abs(val) > tol) nnz_count += 1;
            }
        }

        const data = try allocator.alloc(f64, nnz_count);
        errdefer allocator.free(data);

        const indices = try allocator.alloc(usize, nnz_count);
        errdefer allocator.free(indices);

        const indptr = try allocator.alloc(usize, cols + 1);
        errdefer allocator.free(indptr);

        var idx: usize = 0;
        indptr[0] = 0;
        for (0..cols) |c| {
            for (0..rows) |r| {
                const val = try dense.getAsFloat(&.{ r, c });
                if (@abs(val) > tol) {
                    data[idx] = val;
                    indices[idx] = r;
                    idx += 1;
                }
            }
            indptr[c + 1] = idx;
        }

        return CscMatrix{
            .allocator = allocator,
            .rows = rows,
            .cols = cols,
            .data = data,
            .indices = indices,
            .indptr = indptr,
        };
    }

    pub fn toDense(self: CscMatrix) !Array {
        var arr = try zeros(self.allocator, .{ .shape = &.{ self.rows, self.cols }, .dtype = .f64 });
        errdefer arr.deinit();

        for (0..self.cols) |c| {
            const start = self.indptr[c];
            const end = self.indptr[c + 1];
            for (start..end) |i| {
                const r = self.indices[i];
                const val = self.data[i];
                try arr.setFromFloat(&.{ r, c }, val);
            }
        }

        return arr;
    }
};

test "csr and csc matrix roundtrip and matvec" {
    const allocator = std.testing.allocator;
    const fromSlice = @import("../core/array.zig").fromSlice;

    // 3x3 sparse matrix with 3 non-zeros
    const data = [_]f64{
        10.0, 0.0,  0.0,
        0.0,  20.0, 0.0,
        0.0,  0.0,  30.0,
    };
    var dense = try fromSlice(allocator, f64, .{ .data = &data, .shape = &.{ 3, 3 } });
    defer dense.deinit();

    var csr = try CsrMatrix.fromDense(allocator, dense, 1e-10);
    defer csr.deinit();

    try std.testing.expectEqual(@as(usize, 3), csr.nnz());
    try std.testing.expectEqual(@as(usize, 3), csr.rows);
    try std.testing.expectEqual(@as(usize, 3), csr.cols);

    var recovered = try csr.toDense();
    defer recovered.deinit();

    for (0..3) |r| {
        for (0..3) |c| {
            try std.testing.expectEqual(try dense.getAsFloat(&.{ r, c }), try recovered.getAsFloat(&.{ r, c }));
        }
    }

    // Multiply by vector [1, 2, 3] -> [10, 40, 90]
    const vec_data = [_]f64{ 1.0, 2.0, 3.0 };
    var x = try fromSlice(allocator, f64, .{ .data = &vec_data, .shape = &.{3} });
    defer x.deinit();

    var y = try csr.dotVector(x);
    defer y.deinit();

    try std.testing.expectEqual(@as(f64, 10.0), try y.getAsFloat(&.{0}));
    try std.testing.expectEqual(@as(f64, 40.0), try y.getAsFloat(&.{1}));
    try std.testing.expectEqual(@as(f64, 90.0), try y.getAsFloat(&.{2}));

    // CSC check
    var csc = try CscMatrix.fromDense(allocator, dense, 1e-10);
    defer csc.deinit();

    var csc_dense = try csc.toDense();
    defer csc_dense.deinit();

    try std.testing.expectEqual(@as(f64, 20.0), try csc_dense.getAsFloat(&.{ 1, 1 }));
}
