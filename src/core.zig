const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{
    ShapeMismatch,
    RankMismatch,
    AllocationFailed,
    IndexOutOfBounds,
    UnsupportedType,
    DimensionMismatch,
    NotImplemented,
    SingularMatrix,
    InvalidFormat,
    UnsupportedVersion,
};

/// N-dimensional array structure.
///
/// Represents a multi-dimensional container of items of the same type and size.
/// The number of dimensions and items in an array is defined by its shape, which is a tuple of N non-negative integers that specify the sizes of each dimension.
/// The type of items in the array is specified by a separate data-type object (one of the primitive types or a derived type).
///
/// Memory Layout:
/// The array uses a row-major (C-style) memory layout by default.
/// It manages its own memory for data, shape, and strides.
pub fn NDArray(comptime T: type) type {
    return struct {
        const Self = @This();

        data: []T,
        shape: []const usize,
        strides: []const usize,
        owns_data: bool,

        pub const Flags = struct {
            c_contiguous: bool = true,
            f_contiguous: bool = false,
            writeable: bool = true,
            aligned: bool = true,
            owndata: bool = true,
        };

        /// Returns the memory layout flags for the array.
        ///
        /// Returns:
        ///     A Flags struct indicating the contiguity and ownership status of the array.
        pub fn flags(self: Self) Flags {
            // Check C-contiguity (row-major)
            var is_c = true;
            var current_stride: usize = 1;
            var i: usize = self.shape.len;
            while (i > 0) {
                i -= 1;
                if (self.shape[i] > 1) {
                    if (self.strides[i] != current_stride) {
                        is_c = false;
                        break;
                    }
                    current_stride *= self.shape[i];
                }
            }

            // Check F-contiguity (column-major)
            var is_f = true;
            current_stride = 1;
            for (0..self.shape.len) |j| {
                if (self.shape[j] > 1) {
                    if (self.strides[j] != current_stride) {
                        is_f = false;
                        break;
                    }
                    current_stride *= self.shape[j];
                }
            }

            // Check alignment (basic check for now)
            const ptr_addr = @intFromPtr(self.data.ptr);
            const align_of_t = @alignOf(T);
            const is_aligned = (ptr_addr % align_of_t) == 0;

            return Flags{
                .c_contiguous = is_c,
                .f_contiguous = is_f,
                .writeable = true, // Default to true for now
                .aligned = is_aligned,
                .owndata = self.owns_data,
            };
        }

        /// Initializes a new NDArray with the specified shape.
        ///
        /// Allocates memory for the data, shape, and strides. The data is uninitialized.
        ///
        /// Arguments:
        ///     allocator: The allocator to use for memory allocation.
        ///     shape: A slice representing the dimensions of the array.
        ///
        /// Returns:
        ///     A new NDArray instance.
        ///
        /// Example:
        /// ```zig
        /// var arr = try NDArray(f32).init(allocator, &.{2, 3});
        /// defer arr.deinit(allocator);
        /// ```
        pub fn init(allocator: Allocator, shape: []const usize) !Self {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            const data = try allocator.alloc(T, total_size);
            errdefer allocator.free(data);

            const shape_copy = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(shape_copy);
            @memcpy(shape_copy, shape);

            const strides = try allocator.alloc(usize, shape.len);
            errdefer allocator.free(strides);

            // Compute row-major strides
            if (shape.len > 0) {
                strides[shape.len - 1] = 1;
                var i: usize = shape.len - 1;
                while (i > 0) {
                    i -= 1;
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
            }

            return Self{
                .data = data,
                .shape = shape_copy,
                .strides = strides,
                .owns_data = true,
            };
        }

        /// Frees all resources associated with this array.
        ///
        /// This includes the data (if owned), shape, and strides.
        pub fn deinit(self: *Self, allocator: Allocator) void {
            if (self.owns_data) {
                allocator.free(self.data);
            }
            allocator.free(self.shape);
            allocator.free(self.strides);
        }

        /// Prints the array to stderr.
        pub fn print(self: Self) !void {
            std.debug.print("NDArray(shape={any}, data={any})\n", .{ self.shape, self.data });
        }

        /// Create a deep copy of the array.
        pub fn copy(self: Self, allocator: Allocator) !Self {
            const result = try init(allocator, self.shape);
            @memcpy(result.data, self.data);
            return result;
        }

        /// Fill the array with a scalar value.
        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        /// Cast the array to a different type.
        pub fn astype(self: Self, allocator: Allocator, comptime DestT: type) !NDArray(DestT) {
            var result = try NDArray(DestT).init(allocator, self.shape);
            for (self.data, 0..) |val, i| {
                result.data[i] = switch (@typeInfo(DestT)) {
                    .int => @as(DestT, @intFromFloat(val)), // Assuming source is float, need generic cast
                    .float => @as(DestT, @floatFromInt(val)), // Assuming source is int
                    else => @as(DestT, val), // Try direct cast
                };
                // The above switch is too simplistic for generic T.
                // Better to use a cast helper or just @as if compatible.
                // For now, let's assume simple numeric conversions.
                if (@typeInfo(T) == .float and @typeInfo(DestT) == .int) {
                    result.data[i] = @intFromFloat(val);
                } else if (@typeInfo(T) == .int and @typeInfo(DestT) == .float) {
                    result.data[i] = @floatFromInt(val);
                } else {
                    result.data[i] = @as(DestT, @intCast(val)); // Fallback
                }
            }
            return result;
        }

        /// Create a new array filled with zeros.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     shape: The shape of the array.
        ///
        /// Returns:
        ///     A new array filled with zeros.
        ///
        /// Example:
        /// ```zig
        /// var z = try NDArray(f32).zeros(allocator, &.{2, 2});
        /// defer z.deinit(allocator);
        /// ```
        pub fn zeros(allocator: Allocator, shape: []const usize) !Self {
            const self = try init(allocator, shape);
            @memset(self.data, 0);
            return self;
        }

        /// Create a 2D identity matrix.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     n: The number of rows and columns.
        ///
        /// Returns:
        ///     A new (n, n) identity matrix.
        ///
        /// Example:
        /// ```zig
        /// var i = try NDArray(f32).eye(allocator, 3);
        /// defer i.deinit(allocator);
        /// ```
        pub fn eye(allocator: Allocator, n: usize) !Self {
            const self = try zeros(allocator, &.{ n, n });
            const one: T = switch (@typeInfo(T)) {
                .int => 1,
                .float => 1.0,
                else => @compileError("Unsupported type for eye()"),
            };

            var i: usize = 0;
            while (i < n) : (i += 1) {
                // Direct access is safe here as we know the shape
                // stride[0] = n, stride[1] = 1
                // index = i * n + i
                self.data[i * n + i] = one;
            }
            return self;
        }

        /// Create a new array filled with ones.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     shape: The shape of the array.
        ///
        /// Returns:
        ///     A new array filled with ones.
        ///
        /// Example:
        /// ```zig
        /// var o = try NDArray(f32).ones(allocator, &.{2, 2});
        /// defer o.deinit(allocator);
        /// ```
        pub fn ones(allocator: Allocator, shape: []const usize) !Self {
            const self = try init(allocator, shape);
            // Handle different types for 1
            const one: T = switch (@typeInfo(T)) {
                .int => 1,
                .float => 1.0,
                else => @compileError("Unsupported type for ones()"),
            };
            @memset(self.data, one);
            return self;
        }

        /// Create a new array filled with a specific value.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     shape: The shape of the array.
        ///     value: The value to fill the array with.
        ///
        /// Returns:
        ///     A new array filled with the specified value.
        ///
        /// Example:
        /// ```zig
        /// var f = try NDArray(f32).full(allocator, &.{2, 2}, 3.14);
        /// defer f.deinit(allocator);
        /// ```
        pub fn full(allocator: Allocator, shape: []const usize, value: T) !Self {
            const self = try init(allocator, shape);
            @memset(self.data, value);
            return self;
        }

        /// Get the element at the specified indices.
        /// Returns IndexOutOfBounds if indices are invalid.
        pub fn get(self: Self, indices: []const usize) !T {
            if (indices.len != self.shape.len) return Error.RankMismatch;

            var offset: usize = 0;
            for (indices, 0..) |idx, i| {
                if (idx >= self.shape[i]) return Error.IndexOutOfBounds;
                offset += idx * self.strides[i];
            }
            return self.data[offset];
        }

        /// Set the element at the specified indices.
        /// Returns IndexOutOfBounds if indices are invalid.
        pub fn set(self: *Self, indices: []const usize, value: T) !void {
            if (indices.len != self.shape.len) return Error.RankMismatch;

            var offset: usize = 0;
            for (indices, 0..) |idx, i| {
                if (idx >= self.shape[i]) return Error.IndexOutOfBounds;
                offset += idx * self.strides[i];
            }
            self.data[offset] = value;
        }

        /// Return the total number of elements.
        pub fn size(self: Self) usize {
            var s: usize = 1;
            for (self.shape) |dim| {
                s *= dim;
            }
            return s;
        }

        /// Return the rank (number of dimensions).
        pub fn rank(self: Self) usize {
            return self.shape.len;
        }

        /// Return a new 1D array with values within a given interval.
        /// Values are generated within the half-open interval [start, stop).
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     start: The starting value of the sequence.
        ///     stop: The end value of the sequence (exclusive).
        ///     step: The step size between values.
        ///
        /// Returns:
        ///     A new 1D array.
        ///
        /// Example:
        /// ```zig
        /// var r = try NDArray(f32).arange(allocator, 0, 10, 1);
        /// defer r.deinit(allocator);
        /// ```
        pub fn arange(allocator: Allocator, start: T, stop: T, step: T) !Self {
            if (step == 0) return Error.DimensionMismatch;

            var count: usize = 0;
            if (@typeInfo(T) == .float) {
                const diff = stop - start;
                const steps = @ceil(diff / step);
                if (steps <= 0) return try init(allocator, &.{0});
                count = @intFromFloat(steps);
            } else {
                // Integer logic
                if (step > 0) {
                    if (start >= stop) return try init(allocator, &.{0});
                    count = @intCast(@divTrunc(stop - start + step - 1, step));
                } else {
                    if (@typeInfo(T) == .int and @typeInfo(T).int.signedness == .unsigned) {
                        // Unsigned types cannot have negative step
                        return Error.DimensionMismatch;
                    }
                    if (start <= stop) return try init(allocator, &.{0});
                    count = @intCast(@divTrunc(start - stop - step - 1, -step));
                }
            }

            const self = try init(allocator, &.{count});
            var val = start;
            for (self.data) |*d| {
                d.* = val;
                val += step;
            }
            return self;
        }

        /// Return a new 1D array with num evenly spaced samples, calculated over the interval [start, stop].
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     start: The starting value of the sequence.
        ///     stop: The end value of the sequence.
        ///     num: The number of samples to generate.
        ///
        /// Returns:
        ///     A new 1D array.
        ///
        /// Example:
        /// ```zig
        /// var l = try NDArray(f32).linspace(allocator, 0, 1, 11);
        /// defer l.deinit(allocator);
        /// ```
        pub fn linspace(allocator: Allocator, start: T, stop: T, num: usize) !Self {
            if (num == 0) return try init(allocator, &.{0});
            if (num == 1) {
                const self = try init(allocator, &.{1});
                self.data[0] = start;
                return self;
            }

            const self = try init(allocator, &.{num});

            if (@typeInfo(T) == .float) {
                const step = (stop - start) / @as(T, @floatFromInt(num - 1));
                for (self.data, 0..) |*d, i| {
                    d.* = start + step * @as(T, @floatFromInt(i));
                }
            } else {
                // For integers, this is approximate
                const diff = stop - start;
                for (self.data, 0..) |*d, i| {
                    const ratio = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(num - 1));
                    d.* = start + @as(T, @intFromFloat(@as(f64, @floatFromInt(diff)) * ratio));
                }
            }
            return self;
        }

        /// Return a contiguous copy of the array.
        pub fn asContiguous(self: Self, allocator: Allocator) !Self {
            if (self.flags().c_contiguous and self.owns_data) {
                return self.copy(allocator);
            }
            // Reuse flatten logic but keep shape?
            // Or just implement copy logic that respects strides.

            const result = try init(allocator, self.shape);

            // Iterate and copy
            var coords = try allocator.alloc(usize, self.rank());
            defer allocator.free(coords);
            @memset(coords, 0);

            var i: usize = 0;
            while (i < self.size()) : (i += 1) {
                result.data[i] = try self.get(coords);

                if (self.rank() > 0) {
                    var dim = self.rank() - 1;
                    while (true) {
                        coords[dim] += 1;
                        if (coords[dim] < self.shape[dim]) break;
                        coords[dim] = 0;
                        if (dim == 0) break;
                        dim -= 1;
                    }
                }
            }
            return result;
        }

        /// Expand the shape of an array.
        pub fn expandDims(self: Self, allocator: Allocator, axis: usize) !Self {
            if (axis > self.rank()) return Error.IndexOutOfBounds;

            const new_rank = self.rank() + 1;
            const new_shape = try allocator.alloc(usize, new_rank);
            defer allocator.free(new_shape);

            var j: usize = 0;
            for (0..new_rank) |i| {
                if (i == axis) {
                    new_shape[i] = 1;
                } else {
                    new_shape[i] = self.shape[j];
                    j += 1;
                }
            }

            const result = try init(allocator, new_shape);
            @memcpy(result.data, self.data);
            return result;
        }

        /// Join a sequence of arrays along an existing axis.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     arrays: A slice of arrays to concatenate.
        ///     axis: The axis along which the arrays will be joined.
        ///
        /// Returns:
        ///     A new concatenated array.
        ///
        /// Example:
        /// ```zig
        /// var c = try NDArray(f32).concatenate(allocator, &.{a, b}, 0);
        /// defer c.deinit(allocator);
        /// ```
        pub fn concatenate(allocator: Allocator, arrays: []const Self, axis: usize) !Self {
            if (arrays.len == 0) return Error.DimensionMismatch;
            const r = arrays[0].rank();
            if (axis >= r) return Error.IndexOutOfBounds;

            // Verify shapes
            var total_dim: usize = 0;
            for (arrays) |arr| {
                if (arr.rank() != r) return Error.RankMismatch;
                for (arr.shape, 0..) |d, i| {
                    if (i != axis and d != arrays[0].shape[i]) return Error.ShapeMismatch;
                }
                total_dim += arr.shape[axis];
            }

            // Result shape
            const new_shape = try allocator.alloc(usize, r);
            defer allocator.free(new_shape);
            @memcpy(new_shape, arrays[0].shape);
            new_shape[axis] = total_dim;

            var result = try init(allocator, new_shape);

            // Copy data
            if (axis == 0) {
                var offset: usize = 0;
                for (arrays) |arr| {
                    @memcpy(result.data[offset..][0..arr.size()], arr.data);
                    offset += arr.size();
                }
            } else {
                var axis_offset: usize = 0;
                for (arrays) |arr| {
                    var coords = try allocator.alloc(usize, r);
                    defer allocator.free(coords);
                    @memset(coords, 0);

                    var i: usize = 0;
                    while (i < arr.size()) : (i += 1) {
                        var res_offset: usize = 0;
                        for (coords, 0..) |c, dim| {
                            const pos = if (dim == axis) c + axis_offset else c;
                            res_offset += pos * result.strides[dim];
                        }

                        result.data[res_offset] = arr.data[i];

                        if (r > 0) {
                            var dim = r - 1;
                            while (true) {
                                coords[dim] += 1;
                                if (coords[dim] < arr.shape[dim]) break;
                                coords[dim] = 0;
                                if (dim == 0) break;
                                dim -= 1;
                            }
                        }
                    }
                    axis_offset += arr.shape[axis];
                }
            }
            return result;
        }

        /// Stack arrays in sequence along a new axis.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     arrays: A slice of arrays to stack.
        ///     axis: The axis in the result array along which the input arrays are stacked.
        ///
        /// Returns:
        ///     A new stacked array.
        ///
        /// Example:
        /// ```zig
        /// var s = try NDArray(f32).stack(allocator, &.{a, b}, 0);
        /// defer s.deinit(allocator);
        /// ```
        pub fn stack(allocator: Allocator, arrays: []const Self, axis: usize) !Self {
            if (arrays.len == 0) return Error.DimensionMismatch;
            const r = arrays[0].rank();
            if (axis > r) return Error.IndexOutOfBounds;

            const expanded_arrays = try allocator.alloc(Self, arrays.len);
            defer allocator.free(expanded_arrays);

            // We need to clean up if something fails, but for now let's assume success or leak on error (prototype)
            // Better: use errdefer

            for (arrays, 0..) |arr, i| {
                expanded_arrays[i] = try arr.expandDims(allocator, axis);
            }
            defer {
                for (expanded_arrays) |*arr| arr.deinit(allocator);
            }

            return concatenate(allocator, expanded_arrays, axis);
        }

        /// Broadcast array to a new shape. Returns a view (does not copy data).
        /// The returned array does NOT own the data.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     new_shape: The shape to broadcast to.
        ///
        /// Returns:
        ///     A view of the array with the new shape.
        ///
        /// Example:
        /// ```zig
        /// var b = try arr.broadcastTo(allocator, &.{2, 3});
        /// ```
        pub fn broadcastTo(self: Self, allocator: Allocator, new_shape: []const usize) !Self {
            if (new_shape.len < self.rank()) return Error.ShapeMismatch;

            const new_strides = try allocator.alloc(usize, new_shape.len);
            errdefer allocator.free(new_strides);

            const offset = new_shape.len - self.rank();

            // Check compatibility and compute strides
            for (0..new_shape.len) |i| {
                if (i < offset) {
                    // New dimensions must be broadcastable (usually 1 in source, but here source doesn't exist)
                    // Actually, new dimensions are prepended.
                    // If we are broadcasting, the new outer dims have stride 0 if we want to repeat?
                    // No, if we prepend dims, we are repeating the whole block.
                    // Stride should be 0.
                    new_strides[i] = 0;
                } else {
                    const self_i = i - offset;
                    if (self.shape[self_i] == new_shape[i]) {
                        new_strides[i] = self.strides[self_i];
                    } else if (self.shape[self_i] == 1) {
                        new_strides[i] = 0; // Broadcast dimension
                    } else {
                        return Error.ShapeMismatch;
                    }
                }
            }

            const shape_copy = try allocator.alloc(usize, new_shape.len);
            errdefer allocator.free(shape_copy);
            @memcpy(shape_copy, new_shape);

            return Self{
                .data = self.data,
                .shape = shape_copy,
                .strides = new_strides,
                .owns_data = false,
            };
        }

        /// Permute the axes of an array. Returns a view.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     axes: The new order of axes.
        ///
        /// Returns:
        ///     A view of the array with permuted axes.
        ///
        /// Example:
        /// ```zig
        /// var p = try arr.permute(allocator, &.{1, 0});
        /// ```
        pub fn permute(self: Self, allocator: Allocator, axes: []const usize) !Self {
            if (axes.len != self.rank()) return Error.RankMismatch;

            const new_shape = try allocator.alloc(usize, self.rank());
            errdefer allocator.free(new_shape);

            const new_strides = try allocator.alloc(usize, self.rank());
            errdefer allocator.free(new_strides);

            for (axes, 0..) |axis, i| {
                if (axis >= self.rank()) return Error.IndexOutOfBounds;
                new_shape[i] = self.shape[axis];
                new_strides[i] = self.strides[axis];
            }

            return Self{
                .data = self.data,
                .shape = new_shape,
                .strides = new_strides,
                .owns_data = false,
            };
        }

        /// Swap two axes. Returns a view.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     axis1: The first axis.
        ///     axis2: The second axis.
        ///
        /// Returns:
        ///     A view of the array with swapped axes.
        ///
        /// Example:
        /// ```zig
        /// var s = try arr.swapaxes(allocator, 0, 1);
        /// ```
        pub fn swapaxes(self: Self, allocator: Allocator, axis1: usize, axis2: usize) !Self {
            if (axis1 >= self.rank() or axis2 >= self.rank()) return Error.IndexOutOfBounds;

            var axes = try allocator.alloc(usize, self.rank());
            defer allocator.free(axes);

            for (0..self.rank()) |i| axes[i] = i;
            axes[axis1] = axis2;
            axes[axis2] = axis1;

            return self.permute(allocator, axes);
        }

        /// Return a new array with the same data but a new shape.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///     new_shape: The new shape.
        ///
        /// Returns:
        ///     A new array with the new shape.
        ///
        /// Example:
        /// ```zig
        /// var r = try arr.reshape(allocator, &.{2, 3});
        /// ```
        pub fn reshape(self: Self, allocator: Allocator, new_shape: []const usize) !Self {
            var new_size: usize = 1;
            for (new_shape) |dim| new_size *= dim;
            if (new_size != self.size()) return Error.DimensionMismatch;

            if (self.flags().c_contiguous) {
                var new_strides = try allocator.alloc(usize, new_shape.len);
                errdefer allocator.free(new_strides);

                if (new_shape.len > 0) {
                    new_strides[new_shape.len - 1] = 1;
                    var i: usize = new_shape.len - 1;
                    while (i > 0) {
                        i -= 1;
                        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
                    }
                }

                const new_shape_copy = try allocator.alloc(usize, new_shape.len);
                @memcpy(new_shape_copy, new_shape);

                return Self{
                    .data = self.data,
                    .shape = new_shape_copy,
                    .strides = new_strides,
                    .owns_data = false,
                };
            } else {
                var contig_arr = try self.asContiguous(allocator);
                var res = try contig_arr.reshape(allocator, new_shape);

                // Transfer ownership from contig_arr to res
                // contig_arr owns data. res is a view of contig_arr.
                // We want res to own the data.

                // We need to manually set res.owns_data = true
                // And ensure contig_arr doesn't free data when deinit is called.
                // But we can't easily modify contig_arr state if we don't have a pointer to it?
                // contig_arr is a local variable.

                // Hack: We modify res to own data, and we manually free contig_arr's metadata but NOT data.

                res.owns_data = true;

                // Free contig_arr metadata
                allocator.free(contig_arr.shape);
                allocator.free(contig_arr.strides);
                // Do NOT free contig_arr.data

                return res;
            }
        }

        /// Flatten the array into 1D.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///
        /// Returns:
        ///     A new 1D array.
        ///
        /// Example:
        /// ```zig
        /// var f = try arr.flatten(allocator);
        /// defer f.deinit(allocator);
        /// ```
        pub fn flatten(self: Self, allocator: Allocator) !Self {
            return self.reshape(allocator, &.{self.size()});
        }

        /// Transpose the array (reverse dimensions).
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///
        /// Returns:
        ///     A view of the array with reversed axes.
        ///
        /// Example:
        /// ```zig
        /// var t = try arr.transpose(allocator);
        /// ```
        pub fn transpose(self: Self, allocator: Allocator) !Self {
            var axes = try allocator.alloc(usize, self.rank());
            defer allocator.free(axes);
            for (0..self.rank()) |i| {
                axes[i] = self.rank() - 1 - i;
            }
            return self.permute(allocator, axes);
        }

        /// Remove single-dimensional entries from the shape.
        ///
        /// Arguments:
        ///     allocator: The allocator to use.
        ///
        /// Returns:
        ///     A view of the array with single-dimensional entries removed.
        ///
        /// Example:
        /// ```zig
        /// var s = try arr.squeeze(allocator);
        /// ```
        pub fn squeeze(self: Self, allocator: Allocator) !Self {
            var new_rank: usize = 0;
            for (self.shape) |dim| {
                if (dim != 1) new_rank += 1;
            }

            var new_shape = try allocator.alloc(usize, new_rank);
            var new_strides = try allocator.alloc(usize, new_rank);

            var j: usize = 0;
            for (self.shape, 0..) |dim, i| {
                if (dim != 1) {
                    new_shape[j] = dim;
                    new_strides[j] = self.strides[i];
                    j += 1;
                }
            }

            return Self{
                .data = self.data,
                .shape = new_shape,
                .strides = new_strides,
                .owns_data = false,
            };
        }

        /// Compute the sum of all elements.
        ///
        /// Returns:
        ///     The sum of all elements.
        ///
        /// Example:
        /// ```zig
        /// const s = try arr.sum(allocator);
        /// ```
        pub fn sum(self: Self, allocator: Allocator) !T {
            var s: T = 0;
            // Use iterator to handle non-contiguous
            var iter = try NdIterator.init(allocator, self.shape);
            defer iter.deinit(allocator);
            while (iter.next()) |coords| {
                s += try self.get(coords);
            }
            return s;
        }

        /// Compute the mean of all elements.
        ///
        /// Returns:
        ///     The mean of all elements.
        ///
        /// Example:
        /// ```zig
        /// const m = try arr.mean(allocator);
        /// ```
        pub fn mean(self: Self, allocator: Allocator) !T {
            const s = try self.sum(allocator);
            if (@typeInfo(T) == .float) {
                return s / @as(T, @floatFromInt(self.size()));
            } else {
                return @divTrunc(s, @as(T, @intCast(self.size())));
            }
        }

        /// Find the minimum value.
        ///
        /// Returns:
        ///     The minimum value in the array.
        ///
        /// Example:
        /// ```zig
        /// const m = try arr.min(allocator);
        /// ```
        pub fn min(self: Self, allocator: Allocator) !T {
            if (self.size() == 0) return Error.DimensionMismatch;
            var iter = try NdIterator.init(allocator, self.shape);
            defer iter.deinit(allocator);

            var m: T = undefined;
            if (iter.next()) |coords| {
                m = try self.get(coords);
            } else {
                return Error.DimensionMismatch;
            }

            while (iter.next()) |coords| {
                const val = try self.get(coords);
                if (val < m) m = val;
            }
            return m;
        }

        /// Find the maximum value.
        ///
        /// Returns:
        ///     The maximum value in the array.
        ///
        /// Example:
        /// ```zig
        /// const m = try arr.max(allocator);
        /// ```
        pub fn max(self: Self, allocator: Allocator) !T {
            if (self.size() == 0) return Error.DimensionMismatch;
            var iter = try NdIterator.init(allocator, self.shape);
            defer iter.deinit(allocator);

            var m: T = undefined;
            if (iter.next()) |coords| {
                m = try self.get(coords);
            } else {
                return Error.DimensionMismatch;
            }

            while (iter.next()) |coords| {
                const val = try self.get(coords);
                if (val > m) m = val;
            }
            return m;
        }
    };
}

/// N-dimensional iterator.
/// Iterates over coordinates of a given shape.
pub const NdIterator = struct {
    shape: []const usize,
    coords: []usize,
    first: bool,

    pub fn init(allocator: Allocator, shape: []const usize) !NdIterator {
        const coords = try allocator.alloc(usize, shape.len);
        @memset(coords, 0);
        return NdIterator{
            .shape = shape,
            .coords = coords,
            .first = true,
        };
    }

    pub fn deinit(self: *NdIterator, allocator: Allocator) void {
        allocator.free(self.coords);
    }

    pub fn reset(self: *NdIterator) void {
        @memset(self.coords, 0);
        self.first = true;
    }

    pub fn next(self: *NdIterator) ?[]const usize {
        if (self.shape.len == 0) {
            if (self.first) {
                self.first = false;
                return self.coords;
            }
            return null;
        }

        if (self.first) {
            self.first = false;
            // Check if shape has any zero dimension
            for (self.shape) |d| {
                if (d == 0) return null;
            }
            return self.coords;
        }

        var dim = self.shape.len - 1;
        while (true) {
            self.coords[dim] += 1;
            if (self.coords[dim] < self.shape[dim]) {
                return self.coords;
            }
            self.coords[dim] = 0;
            if (dim == 0) return null;
            dim -= 1;
        }
    }
};

/// Compute the broadcasted shape of two arrays.
/// Caller owns the returned slice.
pub fn broadcastShape(allocator: Allocator, shape_a: []const usize, shape_b: []const usize) ![]usize {
    const rank_a = shape_a.len;
    const rank_b = shape_b.len;
    const max_rank = @max(rank_a, rank_b);

    const result_shape = try allocator.alloc(usize, max_rank);
    errdefer allocator.free(result_shape);

    var i: usize = 0;
    while (i < max_rank) : (i += 1) {
        // i=0 means last dimension.

        const idx_a_rev = i;
        const idx_b_rev = i;

        const val_a = if (idx_a_rev < rank_a) shape_a[rank_a - 1 - idx_a_rev] else 1;
        const val_b = if (idx_b_rev < rank_b) shape_b[rank_b - 1 - idx_b_rev] else 1;

        if (val_a != val_b and val_a != 1 and val_b != 1) {
            return Error.ShapeMismatch;
        }

        result_shape[max_rank - 1 - i] = @max(val_a, val_b);
    }
    return result_shape;
}

test "core advanced 3d manipulation" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).init(allocator, &.{ 2, 3, 4 });
    defer arr.deinit(allocator);
    arr.fill(1.0);

    try std.testing.expectEqual(arr.rank(), 3);
    try std.testing.expectEqual(arr.size(), 24);

    // Test reshape
    var reshaped = try arr.reshape(allocator, &.{ 4, 6 });
    defer reshaped.deinit(allocator);
    try std.testing.expectEqual(reshaped.rank(), 2);
    try std.testing.expectEqual(reshaped.shape[0], 4);
    try std.testing.expectEqual(reshaped.shape[1], 6);

    // Test transpose
    var transposed = try arr.permute(allocator, &.{ 2, 0, 1 });
    defer transposed.deinit(allocator);
    try std.testing.expectEqual(transposed.shape[0], 4);
    try std.testing.expectEqual(transposed.shape[1], 2);
    try std.testing.expectEqual(transposed.shape[2], 3);
}

test "core broadcast shape complex" {
    const allocator = std.testing.allocator;
    const s1 = [_]usize{ 1, 3, 1 };
    const s2 = [_]usize{ 2, 1, 4 };
    const res = try broadcastShape(allocator, &s1, &s2);
    defer allocator.free(res);

    try std.testing.expectEqual(res.len, 3);
    try std.testing.expectEqual(res[0], 2);
    try std.testing.expectEqual(res[1], 3);
    try std.testing.expectEqual(res[2], 4);
}

test "core primitive types and high dimensions" {
    const allocator = std.testing.allocator;

    // Test f128
    var arr_f128 = try NDArray(f128).zeros(allocator, &.{ 2, 2 });
    defer arr_f128.deinit(allocator);
    try arr_f128.set(&.{ 0, 0 }, 1.23456789012345678901234567890123456789);
    const val = try arr_f128.get(&.{ 0, 0 });
    try std.testing.expectApproxEqAbs(val, 1.23456789012345678901234567890123456789, 1e-30);

    // Test i128
    var arr_i128 = try NDArray(i128).zeros(allocator, &.{2});
    defer arr_i128.deinit(allocator);
    try arr_i128.set(&.{0}, 123456789012345678901234567890);
    try std.testing.expectEqual(try arr_i128.get(&.{0}), 123456789012345678901234567890);

    // Test 5D array
    var arr_5d = try NDArray(u8).zeros(allocator, &.{ 2, 2, 2, 2, 2 });
    defer arr_5d.deinit(allocator);
    try std.testing.expectEqual(arr_5d.rank(), 5);
    try std.testing.expectEqual(arr_5d.size(), 32);
    try arr_5d.set(&.{ 1, 1, 1, 1, 1 }, 255);
    try std.testing.expectEqual(try arr_5d.get(&.{ 1, 1, 1, 1, 1 }), 255);
}

test "core flags" {
    const allocator = std.testing.allocator;
    var arr = try NDArray(f32).zeros(allocator, &.{ 2, 3 });
    defer arr.deinit(allocator);

    const f = arr.flags();
    try std.testing.expect(f.c_contiguous);
    try std.testing.expect(!f.f_contiguous);
    try std.testing.expect(f.owndata);
    try std.testing.expect(f.writeable);
    try std.testing.expect(f.aligned);
}
