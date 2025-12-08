const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core.zig");
const NDArray = core.NDArray;

/// Compressed Sparse Row (CSR) Matrix.
pub fn CSRMatrix(comptime T: type) type {
    return struct {
        const Self = @This();

        values: std.ArrayListUnmanaged(T),
        col_indices: std.ArrayListUnmanaged(usize),
        row_ptr: std.ArrayListUnmanaged(usize),
        shape: [2]usize,

        pub fn init(allocator: Allocator, rows: usize, cols: usize) !Self {
            var row_ptr = std.ArrayListUnmanaged(usize){};
            errdefer row_ptr.deinit(allocator);

            try row_ptr.ensureTotalCapacity(allocator, rows + 1);
            try row_ptr.append(allocator, 0); // Start with 0

            var i: usize = 0;
            while (i < rows) : (i += 1) {
                try row_ptr.append(allocator, 0);
            }

            return Self{
                .values = std.ArrayListUnmanaged(T){},
                .col_indices = std.ArrayListUnmanaged(usize){},
                .row_ptr = row_ptr,
                .shape = .{ rows, cols },
            };
        }

        pub fn deinit(self: *Self, allocator: Allocator) void {
            self.values.deinit(allocator);
            self.col_indices.deinit(allocator);
            self.row_ptr.deinit(allocator);
        }

        /// Converts a dense NDArray to CSR format.
        pub fn fromDense(allocator: Allocator, arr: NDArray(T)) !Self {
            if (arr.rank() != 2) return core.Error.RankMismatch;
            const rows = arr.shape[0];
            const cols = arr.shape[1];

            var csr = try Self.init(allocator, rows, cols);
            errdefer csr.deinit(allocator);

            // Reset row_ptr to just [0]
            csr.row_ptr.clearRetainingCapacity();
            try csr.row_ptr.append(allocator, 0);

            var nnz_count: usize = 0;
            for (0..rows) |i| {
                for (0..cols) |j| {
                    const val = try arr.get(&.{ i, j });
                    if (val != 0) {
                        try csr.values.append(allocator, val);
                        try csr.col_indices.append(allocator, j);
                        nnz_count += 1;
                    }
                }
                try csr.row_ptr.append(allocator, nnz_count);
            }
            return csr;
        }

        /// Converts CSR to dense NDArray.
        pub fn toDense(self: Self, allocator: Allocator) !NDArray(T) {
            var arr = try NDArray(T).zeros(allocator, &self.shape);

            for (0..self.shape[0]) |i| {
                const start = self.row_ptr.items[i];
                const end = self.row_ptr.items[i + 1];
                for (start..end) |k| {
                    const col = self.col_indices.items[k];
                    const val = self.values.items[k];
                    try arr.set(&.{ i, col }, val);
                }
            }
            return arr;
        }
    };
}

test "sparse csr matrix" {
    const allocator = std.testing.allocator;
    var dense = try NDArray(f32).init(allocator, &.{ 3, 3 });
    defer dense.deinit(allocator);
    dense.fill(0);
    try dense.set(&.{ 0, 0 }, 1.0);
    try dense.set(&.{ 1, 1 }, 2.0);
    try dense.set(&.{ 2, 2 }, 3.0);

    var csr = try CSRMatrix(f32).fromDense(allocator, dense);
    defer csr.deinit(allocator);

    try std.testing.expectEqual(csr.values.items.len, 3);
    try std.testing.expectEqual(csr.values.items[0], 1.0);
    try std.testing.expectEqual(csr.values.items[1], 2.0);
    try std.testing.expectEqual(csr.values.items[2], 3.0);

    var dense2 = try csr.toDense(allocator);
    defer dense2.deinit(allocator);

    try std.testing.expectEqual(try dense2.get(&.{ 0, 0 }), 1.0);
    try std.testing.expectEqual(try dense2.get(&.{ 1, 1 }), 2.0);
    try std.testing.expectEqual(try dense2.get(&.{ 2, 2 }), 3.0);
    try std.testing.expectEqual(try dense2.get(&.{ 0, 1 }), 0.0);
}
