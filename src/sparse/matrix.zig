const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core.zig");
const NDArray = core.NDArray;

/// Compressed Sparse Row (CSR) Matrix.
pub fn CSRMatrix(comptime T: type) type {
    return struct {
        const Self = @This();

        allocator: Allocator,
        values: std.ArrayList(T),
        col_indices: std.ArrayList(usize),
        row_ptr: std.ArrayList(usize),
        shape: [2]usize,

        pub fn init(allocator: Allocator, rows: usize, cols: usize) !Self {
            var row_ptr = std.ArrayList(usize){};
            errdefer row_ptr.deinit(allocator);

            try row_ptr.ensureTotalCapacity(allocator, rows + 1);
            try row_ptr.append(allocator, 0); // Start with 0

            var i: usize = 0;
            while (i < rows) : (i += 1) {
                try row_ptr.append(allocator, 0);
            }

            return Self{
                .allocator = allocator,
                .values = std.ArrayList(T){},
                .col_indices = std.ArrayList(usize){},
                .row_ptr = row_ptr,
                .shape = .{ rows, cols },
            };
        }

        pub fn deinit(self: *Self) void {
            self.values.deinit(self.allocator);
            self.col_indices.deinit(self.allocator);
            self.row_ptr.deinit(self.allocator);
        }

        /// Converts a dense NDArray to CSR format.
        pub fn fromDense(allocator: Allocator, arr: NDArray(T)) !Self {
            if (arr.rank() != 2) return core.Error.RankMismatch;
            const rows = arr.shape[0];
            const cols = arr.shape[1];

            var csr = try Self.init(allocator, rows, cols);
            errdefer csr.deinit();

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
        pub fn toDense(self: Self) !NDArray(T) {
            var arr = try NDArray(T).zeros(self.allocator, &self.shape);

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
